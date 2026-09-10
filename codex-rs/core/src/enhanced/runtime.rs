//! Codex-only loader for `$CODEX_HOME/enhanced-runtime.json`.
//!
//! This file is not part of the Vellum portable crate. If the file is absent,
//! every hook stays `DeferToUpstream`.

use std::path::Path;

use serde::Deserialize;

use super::bounded_continuation::ReservedContinuation;
use super::config::AblationProfile;
use super::config::EnhancedRuntimeFeatures;
use super::context_projection::ContextProjectionStore;
use super::context_pruner::ModelVisibleSurface;
use super::hooks::EnhancedTurnHooks;
use super::telemetry::EnhancedEvent;
use super::telemetry::EnhancedEventFields;
use super::telemetry::EnhancedEventKind;
use super::telemetry::MemoryTelemetry;

const CONFIG_FILE: &str = "enhanced-runtime.json";
const ABLATION_ENV: &str = "VELLUM_ENHANCED_ABLATION_PROFILE";
const FEATURE_PROFILE_ENV: &str = "VELLUM_ENHANCED_FEATURE_PROFILE";

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
    pub context_projections: ContextProjectionStore,
}

impl EnhancedSessionRuntime {
    pub fn load(codex_home: impl AsRef<Path>, thread_id: &str) -> Self {
        let features = load_features(codex_home.as_ref());
        let mut telemetry = MemoryTelemetry::for_thread(thread_id);
        let applied = EnhancedEvent::new(
            EnhancedEventKind::SessionFeaturesApplied,
            EnhancedEventFields {
                feature_profile: Some(profile_label(features).to_string()),
                ..EnhancedEventFields::default()
            },
        );
        telemetry.emit(applied);
        super::reporting::publish_event(&telemetry.events[0]);
        Self {
            hooks: EnhancedTurnHooks::new(features),
            telemetry,
            last_surface: None,
            pending_continuation: None,
            ledger_restored: false,
            context_projections: ContextProjectionStore::load(codex_home.as_ref(), thread_id),
        }
    }

    pub fn all_off() -> Self {
        Self {
            hooks: EnhancedTurnHooks::new(EnhancedRuntimeFeatures::all_off()),
            telemetry: MemoryTelemetry::default(),
            last_surface: None,
            pending_continuation: None,
            ledger_restored: false,
            context_projections: ContextProjectionStore::in_memory(),
        }
    }
}

pub fn load_features(codex_home: &Path) -> EnhancedRuntimeFeatures {
    if let Ok(profile) = std::env::var(ABLATION_ENV)
        && let Some(parsed) = AblationProfile::parse(&profile)
    {
        return parsed.features();
    }
    if let Ok(profile) = std::env::var(FEATURE_PROFILE_ENV)
        && let Some(features) = parse_feature_profile(&profile)
    {
        return features;
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

fn profile_label(features: EnhancedRuntimeFeatures) -> &'static str {
    [
        AblationProfile::E0,
        AblationProfile::E1,
        AblationProfile::E2,
        AblationProfile::E3,
        AblationProfile::E4,
        AblationProfile::E5,
    ]
    .into_iter()
    .find(|profile| profile.features() == features)
    .map(AblationProfile::as_str)
    .unwrap_or("custom")
}

fn parse_feature_profile(value: &str) -> Option<EnhancedRuntimeFeatures> {
    serde_json::from_str(value).ok()
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
        std::fs::write(dir.path().join(CONFIG_FILE), r#"{"ablationProfile":"E1"}"#).unwrap();
        let features = load_features(dir.path());
        assert!(features.qwen_tool_reliability);
        assert!(!features.deepseek_context_recovery);
        assert!(!features.qwen_bounded_continuation);
    }

    #[test]
    fn bridge_feature_profile_uses_the_same_shape_as_session_hooks() {
        let features = parse_feature_profile(
            r#"{"qwenToolReliability":true,"deepseekContextRecovery":false,"qwenBoundedContinuation":true}"#,
        )
        .unwrap();
        assert!(features.qwen_tool_reliability);
        assert!(!features.deepseek_context_recovery);
        assert!(features.qwen_bounded_continuation);
    }
}
