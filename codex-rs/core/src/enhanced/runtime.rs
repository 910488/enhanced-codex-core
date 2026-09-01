//! Codex-only loader for `$CODEX_HOME/enhanced-runtime.json`.
//!
//! This file is not part of the Vellum portable crate. If the file is absent,
//! every hook stays `DeferToUpstream`.

use std::path::Path;

use serde::Deserialize;

use super::bounded_continuation::ReservedContinuation;
use super::config::{AblationProfile, EnhancedRuntimeFeatures};
use super::context_pruner::ModelVisibleSurface;
use super::hooks::EnhancedTurnHooks;
use super::telemetry::MemoryTelemetry;

const CONFIG_FILE: &str = "enhanced-runtime.json";
const ABLATION_ENV: &str = "VELLUM_ENHANCED_ABLATION_PROFILE";

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EnhancedRuntimeFile {
    ablation_profile: Option<String>,
    feature_flags: Option<EnhancedRuntimeFeatures>,
    compiled_feature_defaults: Option<EnhancedRuntimeFeatures>,
}

#[derive(Debug)]
pub struct EnhancedSessionRuntime {
    pub hooks: EnhancedTurnHooks,
    pub telemetry: MemoryTelemetry,
    pub last_surface: Option<ModelVisibleSurface>,
    pub pending_continuation: Option<ReservedContinuation>,
    pub ledger_restored: bool,
}

impl EnhancedSessionRuntime {
    pub fn load(codex_home: impl AsRef<Path>) -> Self {
        Self {
            hooks: EnhancedTurnHooks::new(load_features(codex_home.as_ref())),
            telemetry: MemoryTelemetry::default(),
            last_surface: None,
            pending_continuation: None,
            ledger_restored: false,
        }
    }

    pub fn all_off() -> Self {
        Self {
            hooks: EnhancedTurnHooks::new(EnhancedRuntimeFeatures::all_off()),
            telemetry: MemoryTelemetry::default(),
            last_surface: None,
            pending_continuation: None,
            ledger_restored: false,
        }
    }
}

pub fn load_features(codex_home: &Path) -> EnhancedRuntimeFeatures {
    if let Ok(profile) = std::env::var(ABLATION_ENV) {
        if let Some(parsed) = AblationProfile::parse(&profile) {
            return parsed.features();
        }
    }
    let path = codex_home.join(CONFIG_FILE);
    let Ok(bytes) = std::fs::read(&path) else {
        return EnhancedRuntimeFeatures::all_off();
    };
    let Ok(file) = serde_json::from_slice::<EnhancedRuntimeFile>(&bytes) else {
        return EnhancedRuntimeFeatures::all_off();
    };
    if let Some(profile) = file
        .ablation_profile
        .as_deref()
        .and_then(AblationProfile::parse)
    {
        return profile.features();
    }
    if let Some(flags) = file.feature_flags {
        return flags;
    }
    file.compiled_feature_defaults
        .unwrap_or_else(EnhancedRuntimeFeatures::all_off)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn missing_file_is_all_off() {
        let dir = TempDir::new().unwrap();
        let features = load_features(dir.path());
        assert!(!features.any_enabled());
    }

    #[test]
    fn ablation_profile_selects_ports() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(CONFIG_FILE),
            r#"{"ablationProfile":"E1"}"#,
        )
        .unwrap();
        let features = load_features(dir.path());
        assert!(features.qwen_tool_reliability);
        assert!(!features.deepseek_context_recovery);
        assert!(!features.qwen_bounded_continuation);
    }
}
