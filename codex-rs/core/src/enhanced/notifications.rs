use serde::Serialize;
use serde_json::Map;
use serde_json::Value;

use super::config::EnhancedRuntimeFeatures;
use super::telemetry::EnhancedEvent;
use super::telemetry::field_name_is_forbidden;

pub const ENHANCED_IDENTITY_NOTIFICATION: &str = "vellum/enhancedRuntimeIdentity";
pub const ENHANCED_EVENT_NOTIFICATION: &str = "vellum/enhancedEvent";

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EnhancedRuntimeIdentityParams {
    enhanced_commit: String,
    runtime_digest: String,
    feature_profile: String,
    ports: EnhancedPortFlags,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EnhancedPortFlags {
    qwen_tool_reliability: bool,
    deepseek_context_recovery: bool,
    qwen_bounded_continuation: bool,
}

impl From<EnhancedRuntimeFeatures> for EnhancedPortFlags {
    fn from(value: EnhancedRuntimeFeatures) -> Self {
        Self {
            qwen_tool_reliability: value.qwen_tool_reliability,
            deepseek_context_recovery: value.deepseek_context_recovery,
            qwen_bounded_continuation: value.qwen_bounded_continuation,
        }
    }
}

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

pub fn event_notification(event: &EnhancedEvent) -> Value {
    let mut fields = Map::new();
    if let Ok(Value::Object(encoded)) = serde_json::to_value(&event.fields) {
        for (key, value) in encoded {
            if !value.is_null() && !field_name_is_forbidden(&key) {
                fields.insert(key, value);
            }
        }
    }
    serde_json::json!({
        "method": ENHANCED_EVENT_NOTIFICATION,
        "params": {"name": event.name, "fields": Value::Object(fields)},
    })
}
