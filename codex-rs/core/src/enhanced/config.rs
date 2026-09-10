use serde::Deserialize;
use serde::Serialize;

/// Independent feature gates. Eval must select these explicitly; model names
/// never imply a hidden profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct EnhancedRuntimeFeatures {
    pub qwen_tool_reliability: bool,
    pub deepseek_context_recovery: bool,
    pub qwen_bounded_continuation: bool,
}

impl EnhancedRuntimeFeatures {
    pub const fn all_off() -> Self {
        Self {
            qwen_tool_reliability: false,
            deepseek_context_recovery: false,
            qwen_bounded_continuation: false,
        }
    }

    pub const fn all_on() -> Self {
        Self {
            qwen_tool_reliability: true,
            deepseek_context_recovery: true,
            qwen_bounded_continuation: true,
        }
    }

    pub fn any_enabled(self) -> bool {
        self.qwen_tool_reliability
            || self.deepseek_context_recovery
            || self.qwen_bounded_continuation
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
}
