use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Independently gated ports and experiments. Eval must select these
/// explicitly; model names never imply a hidden profile.
///
/// Unknown JSON keys are refused so a new flag cannot be silently dropped.
/// Missing keys default to `false` so older configs still load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct EnhancedRuntimeFeatures {
    pub qwen_tool_reliability: bool,
    pub deepseek_context_recovery: bool,
    pub qwen_bounded_continuation: bool,
    /// Model-visible repetition notice. Default off: diagnostics only.
    pub repetition_notice: bool,
    /// Conservative English trailing-continue on natural-stop final text.
    /// Default off; not implied by E5.
    pub intent_continuation: bool,
}

pub const FEATURE_FLAG_KEYS: &[&str] = &[
    "qwenToolReliability",
    "deepseekContextRecovery",
    "qwenBoundedContinuation",
    "repetitionNotice",
    "intentContinuation",
];

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum FeatureFlagError {
    #[error("feature flags must be a JSON object")]
    NotObject,
    #[error("unknown feature flag {0} is not silently ignored")]
    Unknown(String),
    #[error("feature flag {0} must be a boolean")]
    NotBoolean(String),
}

impl EnhancedRuntimeFeatures {
    pub const fn all_off() -> Self {
        Self {
            qwen_tool_reliability: false,
            deepseek_context_recovery: false,
            qwen_bounded_continuation: false,
            repetition_notice: false,
            intent_continuation: false,
        }
    }

    /// Historical E5 mapping: the three original ports on. New experiments
    /// stay off so E5 cannot smuggle them.
    pub const fn all_on() -> Self {
        Self {
            qwen_tool_reliability: true,
            deepseek_context_recovery: true,
            qwen_bounded_continuation: true,
            repetition_notice: false,
            intent_continuation: false,
        }
    }

    pub fn any_enabled(self) -> bool {
        self.qwen_tool_reliability
            || self.deepseek_context_recovery
            || self.qwen_bounded_continuation
            || self.repetition_notice
            || self.intent_continuation
    }

    pub fn experimental_any_enabled(self) -> bool {
        self.repetition_notice || self.intent_continuation
    }

    /// Parse a JSON object, refusing unknown keys instead of dropping them.
    pub fn parse_strict(value: &Value) -> Result<Self, FeatureFlagError> {
        let object = value.as_object().ok_or(FeatureFlagError::NotObject)?;
        for key in object.keys() {
            if !FEATURE_FLAG_KEYS.contains(&key.as_str()) {
                return Err(FeatureFlagError::Unknown(key.clone()));
            }
            if !object[key].is_boolean() {
                return Err(FeatureFlagError::NotBoolean(key.clone()));
            }
        }
        serde_json::from_value(value.clone()).map_err(|_| FeatureFlagError::NotObject)
    }
}

/// Ablation profiles from the Enhanced Codex MVP plan. Eval runners must pass
/// one of these rather than deriving features from a model id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum AblationProfile {
    /// Unmodified Enhanced Codex fork baseline (all ports off).
    E0,
    /// Tool reliability only.
    E1,
    /// Context recovery only.
    E2,
    /// Bounded continuation only.
    E3,
    /// Tool reliability + context recovery.
    E4,
    /// All ports on.
    E5,
}

impl AblationProfile {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_uppercase().as_str() {
            "E0" | "B0" => Some(Self::E0),
            "E1" => Some(Self::E1),
            "E2" => Some(Self::E2),
            "E3" => Some(Self::E3),
            "E4" => Some(Self::E4),
            "E5" => Some(Self::E5),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::E0 => "E0",
            Self::E1 => "E1",
            Self::E2 => "E2",
            Self::E3 => "E3",
            Self::E4 => "E4",
            Self::E5 => "E5",
        }
    }

    pub fn features(self) -> EnhancedRuntimeFeatures {
        match self {
            Self::E0 => EnhancedRuntimeFeatures::all_off(),
            Self::E1 => EnhancedRuntimeFeatures {
                qwen_tool_reliability: true,
                ..EnhancedRuntimeFeatures::all_off()
            },
            Self::E2 => EnhancedRuntimeFeatures {
                deepseek_context_recovery: true,
                ..EnhancedRuntimeFeatures::all_off()
            },
            Self::E3 => EnhancedRuntimeFeatures {
                qwen_bounded_continuation: true,
                ..EnhancedRuntimeFeatures::all_off()
            },
            Self::E4 => EnhancedRuntimeFeatures {
                qwen_tool_reliability: true,
                deepseek_context_recovery: true,
                qwen_bounded_continuation: false,
                repetition_notice: false,
                intent_continuation: false,
            },
            Self::E5 => EnhancedRuntimeFeatures::all_on(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_do_not_branch_on_model_names() {
        assert_eq!(AblationProfile::parse("qwen"), None);
        assert_eq!(AblationProfile::parse("deepseek"), None);
        assert_eq!(AblationProfile::parse("grok"), None);
        assert!(
            AblationProfile::parse("e5")
                .unwrap()
                .features()
                .any_enabled()
        );
        assert!(!AblationProfile::E0.features().any_enabled());
    }

    #[test]
    fn e0_through_e5_do_not_enable_new_experiments() {
        for profile in [
            AblationProfile::E0,
            AblationProfile::E1,
            AblationProfile::E2,
            AblationProfile::E3,
            AblationProfile::E4,
            AblationProfile::E5,
        ] {
            let features = profile.features();
            assert!(
                !features.repetition_notice,
                "{} must not imply repetitionNotice",
                profile.as_str()
            );
            assert!(
                !features.intent_continuation,
                "{} must not imply intentContinuation",
                profile.as_str()
            );
        }
        assert!(!EnhancedRuntimeFeatures::all_on().experimental_any_enabled());
        assert_eq!(
            AblationProfile::E5.features(),
            EnhancedRuntimeFeatures::all_on()
        );
        assert_eq!(
            AblationProfile::E0.features(),
            EnhancedRuntimeFeatures::all_off()
        );
        assert!(AblationProfile::E1.features().qwen_tool_reliability);
        assert!(AblationProfile::E2.features().deepseek_context_recovery);
        assert!(AblationProfile::E3.features().qwen_bounded_continuation);
        assert!(AblationProfile::E4.features().qwen_tool_reliability);
        assert!(AblationProfile::E4.features().deepseek_context_recovery);
        assert!(!AblationProfile::E4.features().qwen_bounded_continuation);
    }

    #[test]
    fn old_three_flag_configs_still_load_and_unknown_flags_are_refused() {
        let old = serde_json::json!({
            "qwenToolReliability": true,
            "deepseekContextRecovery": false,
            "qwenBoundedContinuation": true
        });
        let parsed = EnhancedRuntimeFeatures::parse_strict(&old).unwrap();
        assert!(parsed.qwen_tool_reliability);
        assert!(!parsed.deepseek_context_recovery);
        assert!(parsed.qwen_bounded_continuation);
        assert!(!parsed.repetition_notice);
        assert!(!parsed.intent_continuation);

        let unknown = serde_json::json!({
            "qwenToolReliability": false,
            "mysteryFlag": true
        });
        assert!(matches!(
            EnhancedRuntimeFeatures::parse_strict(&unknown),
            Err(FeatureFlagError::Unknown(key)) if key == "mysteryFlag"
        ));
        assert!(serde_json::from_value::<EnhancedRuntimeFeatures>(unknown).is_err());
    }
}
