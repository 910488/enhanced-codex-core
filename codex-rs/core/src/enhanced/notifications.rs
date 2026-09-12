//! The two App Server notifications the Enhanced fork is allowed to send.
//!
//! Codex Desktop is the client on the other end of that pipe and knows nothing
//! about Vellum, so these are addressed to the bridge and stop there. They exist
//! for one reason: something has to be able to say *which* runtime actually ran
//! a turn, from inside that runtime, rather than inferring it from a file on
//! disk or a model name.
//!
//! Both payloads are built here, in the portable crate, so the fork cannot
//! invent a field the bridge will refuse — and so the contract can be tested
//! without a fork build.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::config::EnhancedRuntimeFeatures;
use super::telemetry::{field_name_is_forbidden, EnhancedEvent};

pub const ENHANCED_IDENTITY_NOTIFICATION: &str = "vellum/enhancedRuntimeIdentity";
pub const ENHANCED_EVENT_NOTIFICATION: &str = "vellum/enhancedEvent";

/// Sent once, right after the child answers `initialize`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnhancedRuntimeIdentityParams {
    pub enhanced_commit: String,
    pub runtime_digest: String,
    pub feature_profile: String,
    pub ports: EnhancedPortFlags,
}

/// The three independently gated ports, reported as the running binary has them
/// compiled and configured — not as the launch manifest hoped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnhancedPortFlags {
    pub qwen_tool_reliability: bool,
    pub deepseek_context_recovery: bool,
    pub qwen_bounded_continuation: bool,
    #[serde(default)]
    pub repetition_notice: bool,
    #[serde(default)]
    pub intent_continuation: bool,
}

impl From<EnhancedRuntimeFeatures> for EnhancedPortFlags {
    fn from(value: EnhancedRuntimeFeatures) -> Self {
        Self {
            qwen_tool_reliability: value.qwen_tool_reliability,
            deepseek_context_recovery: value.deepseek_context_recovery,
            qwen_bounded_continuation: value.qwen_bounded_continuation,
            repetition_notice: value.repetition_notice,
            intent_continuation: value.intent_continuation,
        }
    }
}

/// Builds the identity notification the fork writes to stdout.
pub fn identity_notification(
    enhanced_commit: impl Into<String>,
    runtime_digest: impl Into<String>,
    feature_profile: impl Into<String>,
    features: EnhancedRuntimeFeatures,
) -> Value {
    let params = EnhancedRuntimeIdentityParams {
        enhanced_commit: enhanced_commit.into(),
        runtime_digest: runtime_digest.into(),
        feature_profile: feature_profile.into(),
        ports: features.into(),
    };
    serde_json::json!({
        "method": ENHANCED_IDENTITY_NOTIFICATION,
        "params": serde_json::to_value(params).unwrap_or(Value::Null),
    })
}

/// Builds an event notification from an `enhanced.*` telemetry event.
///
/// Null fields are dropped rather than sent: the receiver allowlists field
/// names, and an absent measurement is not a measurement of zero.
pub fn event_notification(event: &EnhancedEvent) -> Value {
    let mut fields = Map::new();
    if let Ok(Value::Object(encoded)) = serde_json::to_value(&event.fields) {
        for (key, value) in encoded {
            if value.is_null() || field_name_is_forbidden(&key) {
                continue;
            }
            fields.insert(key, value);
        }
    }
    serde_json::json!({
        "method": ENHANCED_EVENT_NOTIFICATION,
        "params": {"name": event.name, "fields": Value::Object(fields)},
    })
}

#[cfg(test)]
mod tests {
    use super::super::config::AblationProfile;
    use super::super::telemetry::{EnhancedEventFields, EnhancedEventKind};
    use super::*;

    #[test]
    fn identity_reports_every_port_flag_explicitly() {
        let notification = identity_notification(
            "c".repeat(40),
            "sha256:digest",
            AblationProfile::E4.as_str(),
            AblationProfile::E4.features(),
        );
        assert_eq!(notification["method"], ENHANCED_IDENTITY_NOTIFICATION);
        let ports = &notification["params"]["ports"];
        assert_eq!(ports["qwenToolReliability"], true);
        assert_eq!(ports["deepseekContextRecovery"], true);
        assert_eq!(ports["qwenBoundedContinuation"], false);
        assert_eq!(ports["repetitionNotice"], false);
        assert_eq!(ports["intentContinuation"], false);
        assert_eq!(notification["params"]["featureProfile"], "E4");
    }

    #[test]
    fn identity_reports_experimental_flags_explicitly_when_set() {
        let mut features = AblationProfile::E5.features();
        features.repetition_notice = true;
        features.intent_continuation = true;
        let notification = identity_notification("c".repeat(40), "sha256:digest", "custom", features);
        let ports = &notification["params"]["ports"];
        assert_eq!(ports["repetitionNotice"], true);
        assert_eq!(ports["intentContinuation"], true);
        assert_eq!(ports["qwenBoundedContinuation"], true);
    }

    #[test]
    fn event_notifications_carry_only_measured_fields() {
        let event = EnhancedEvent::new(
            EnhancedEventKind::ContextOverflowRetry,
            EnhancedEventFields {
                retry_index: Some(1),
                before_token_estimate: Some(900),
                after_token_estimate: Some(120),
                ..EnhancedEventFields::default()
            },
        );
        let notification = event_notification(&event);
        assert_eq!(notification["method"], ENHANCED_EVENT_NOTIFICATION);
        assert_eq!(
            notification["params"]["name"],
            "enhanced.context.overflow_retry"
        );
        let fields = notification["params"]["fields"].as_object().unwrap();
        assert_eq!(fields.len(), 3);
        assert!(!fields.contains_key("threadIdHash"));
        assert_eq!(fields["retryIndex"], 1);
    }

    #[test]
    fn no_event_field_name_can_be_a_secret() {
        for name in ["threadIdHash", "callIdHash", "runtimeDigest", "modelId"] {
            assert!(!field_name_is_forbidden(name), "{name} must stay allowed");
        }
        for name in ["authorization", "apiKey", "rawPrompt", "toolOutput"] {
            assert!(field_name_is_forbidden(name), "{name} must stay refused");
        }
    }
}
