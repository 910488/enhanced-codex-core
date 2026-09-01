use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::config::AblationProfile;

/// Telemetry namespace is `enhanced.*`. Events are for ablation, debug, and
/// regression — never a control plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnhancedEventKind {
    ToolDuplicateDetected,
    ToolDuplicateSuppressed,
    ToolCallIdCollision,
    ContextPruneStarted,
    ContextPruneCompleted,
    ContextCompactionAvoided,
    ContextOverflowRetry,
    ContextOverflowRetryRefused,
    ContinuationAllowed,
    ContinuationExhausted,
}

impl EnhancedEventKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::ToolDuplicateDetected => "enhanced.tool.duplicate_detected",
            Self::ToolDuplicateSuppressed => "enhanced.tool.duplicate_suppressed",
            Self::ToolCallIdCollision => "enhanced.tool.call_id_collision",
            Self::ContextPruneStarted => "enhanced.context.prune_started",
            Self::ContextPruneCompleted => "enhanced.context.prune_completed",
            Self::ContextCompactionAvoided => "enhanced.context.compaction_avoided",
            Self::ContextOverflowRetry => "enhanced.context.overflow_retry",
            Self::ContextOverflowRetryRefused => "enhanced.context.overflow_retry_refused",
            Self::ContinuationAllowed => "enhanced.continuation.allowed",
            Self::ContinuationExhausted => "enhanced.continuation.exhausted",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct EnhancedEventFields {
    pub thread_id_hash: Option<String>,
    pub runtime_digest: Option<String>,
    pub feature_profile: Option<String>,
    pub provider_id: Option<String>,
    pub model_id: Option<String>,
    pub request_index: Option<u64>,
    pub call_id_hash: Option<String>,
    pub before_token_estimate: Option<u64>,
    pub after_token_estimate: Option<u64>,
    pub chars_removed: Option<u64>,
    pub retry_index: Option<u8>,
    pub continuation_index: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnhancedEvent {
    pub kind: EnhancedEventKind,
    pub name: String,
    pub fields: EnhancedEventFields,
}

impl EnhancedEvent {
    pub fn new(kind: EnhancedEventKind, fields: EnhancedEventFields) -> Self {
        Self {
            kind,
            name: kind.name().to_string(),
            fields,
        }
    }
}

pub fn hash_identifier(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    format!("sha256:{}", hex::encode(digest))
}

const FORBIDDEN_FIELD_NAMES: &[&str] = &[
    "api_key",
    "api-key",
    "apikey",
    "authorization",
    "auth",
    "cookie",
    "prompt",
    "raw_prompt",
    "raw_tool_output",
    "tool_output",
    "user_text",
    "reasoning",
    "encrypted_content",
];

pub fn field_name_is_forbidden(name: &str) -> bool {
    let normalized = name.trim().to_ascii_lowercase().replace('-', "_");
    FORBIDDEN_FIELD_NAMES
        .iter()
        .any(|forbidden| normalized.contains(forbidden))
}

#[derive(Debug, Default)]
pub struct MemoryTelemetry {
    pub events: Vec<EnhancedEvent>,
}

impl MemoryTelemetry {
    pub fn emit(&mut self, event: EnhancedEvent) {
        self.events.push(event);
    }

    pub fn count(&self, kind: EnhancedEventKind) -> usize {
        self.events.iter().filter(|event| event.kind == kind).count()
    }
}

pub fn profile_label(profile: AblationProfile) -> String {
    profile.as_str().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_identifiers_and_rejects_secret_field_names() {
        assert!(hash_identifier("thread-1").starts_with("sha256:"));
        assert_ne!(hash_identifier("thread-1"), "thread-1");
        assert!(field_name_is_forbidden("Authorization"));
        assert!(field_name_is_forbidden("rawPrompt"));
        assert!(!field_name_is_forbidden("request_index"));
    }
}
