use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::config::AblationProfile;

/// Telemetry namespace is `enhanced.*`. Events are for ablation, debug, and
/// regression — never a control plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnhancedEventKind {
    SessionFeaturesApplied,
    ToolCallAdmitted,
    ToolCallCompleted,
    ToolDuplicateDetected,
    ToolDuplicateSuppressed,
    ToolCallIdCollision,
    ContextPruneStarted,
    ContextPruneCompleted,
    ContextCompactionAvoided,
    ContextOverflowRetry,
    ContextOverflowRetryRefused,
    ContextPressureChecked,
    ContextProjectionApplied,
    ContextProjectionRestored,
    ContextProjectionCleared,
    ContinuationEvaluated,
    ContinuationAllowed,
    ContinuationExhausted,
    ToolRepetitionObserved,
    ToolRepetitionNoticeAppended,
    IntentContinuationDetected,
}

impl EnhancedEventKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::SessionFeaturesApplied => "enhanced.session.features_applied",
            Self::ToolCallAdmitted => "enhanced.tool.call_admitted",
            Self::ToolCallCompleted => "enhanced.tool.call_completed",
            Self::ToolDuplicateDetected => "enhanced.tool.duplicate_detected",
            Self::ToolDuplicateSuppressed => "enhanced.tool.duplicate_suppressed",
            Self::ToolCallIdCollision => "enhanced.tool.call_id_collision",
            Self::ContextPruneStarted => "enhanced.context.prune_started",
            Self::ContextPruneCompleted => "enhanced.context.prune_completed",
            Self::ContextCompactionAvoided => "enhanced.context.compaction_avoided",
            Self::ContextOverflowRetry => "enhanced.context.overflow_retry",
            Self::ContextOverflowRetryRefused => "enhanced.context.overflow_retry_refused",
            Self::ContextProjectionApplied => "enhanced.context.projection_applied",
            Self::ContextProjectionRestored => "enhanced.context.projection_restored",
            Self::ContextProjectionCleared => "enhanced.context.projection_cleared",
            Self::ContextPressureChecked => "enhanced.context.pressure_checked",
            Self::ContinuationEvaluated => "enhanced.continuation.evaluated",
            Self::ContinuationAllowed => "enhanced.continuation.allowed",
            Self::ContinuationExhausted => "enhanced.continuation.exhausted",
            Self::ToolRepetitionObserved => "enhanced.tool.repetition_observed",
            Self::ToolRepetitionNoticeAppended => "enhanced.tool.repetition_notice_appended",
            Self::IntentContinuationDetected => "enhanced.continuation.intent_detected",
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
    pub unfinished_signal_count: Option<u64>,
    pub outcome: Option<String>,
    pub succeeded: Option<bool>,
    pub consecutive_count: Option<u8>,
    pub input_fingerprint_hash: Option<String>,
    pub result_fingerprint_hash: Option<String>,
    pub observation_reason: Option<String>,
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
    "patch",
    "raw_patch",
];

pub fn field_name_is_forbidden(name: &str) -> bool {
    let normalized = normalize_field_name(name);
    FORBIDDEN_FIELD_NAMES
        .iter()
        .any(|forbidden| normalized.contains(forbidden))
}

/// The forbidden list is written in snake_case, but the wire format is
/// camelCase — so `toolOutput` has to normalize to `tool_output` or the guard
/// only catches the spelling nobody sends.
fn normalize_field_name(name: &str) -> String {
    let characters = name.trim().replace('-', "_").chars().collect::<Vec<_>>();
    let mut normalized = String::with_capacity(characters.len() + 4);
    for (index, character) in characters.iter().enumerate() {
        let previous = index.checked_sub(1).and_then(|index| characters.get(index));
        let next = characters.get(index + 1);
        let starts_word = character.is_ascii_uppercase()
            && previous.is_some_and(|previous| {
                previous.is_ascii_lowercase()
                    || previous.is_ascii_digit()
                    || (previous.is_ascii_uppercase() && next.is_some_and(char::is_ascii_lowercase))
            });
        if starts_word && !normalized.ends_with('_') {
            normalized.push('_');
        }
        normalized.push(character.to_ascii_lowercase());
    }
    normalized
}

#[derive(Debug, Default)]
pub struct MemoryTelemetry {
    pub events: Vec<EnhancedEvent>,
    thread_id_hash: Option<String>,
}

impl MemoryTelemetry {
    pub fn for_thread(thread_id: &str) -> Self {
        Self {
            events: Vec::new(),
            thread_id_hash: Some(hash_identifier(thread_id)),
        }
    }

    pub fn emit(&mut self, mut event: EnhancedEvent) {
        if event.fields.thread_id_hash.is_none() {
            event.fields.thread_id_hash.clone_from(&self.thread_id_hash);
        }
        self.events.push(event);
    }

    pub fn count(&self, kind: EnhancedEventKind) -> usize {
        self.events
            .iter()
            .filter(|event| event.kind == kind)
            .count()
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
        // camelCase is the wire spelling; the guard has to catch it too.
        assert!(field_name_is_forbidden("toolOutput"));
        assert!(field_name_is_forbidden("apiKey"));
        assert!(field_name_is_forbidden("APIKey"));
        assert!(!field_name_is_forbidden("request_index"));
        assert!(!field_name_is_forbidden("beforeTokenEstimate"));
        assert!(!field_name_is_forbidden("threadIdHash"));
        assert!(field_name_is_forbidden("patch"));
        assert!(field_name_is_forbidden("rawPatch"));
        assert!(!field_name_is_forbidden("inputFingerprintHash"));
        assert!(!field_name_is_forbidden("consecutiveCount"));
        assert!(!field_name_is_forbidden("observationReason"));
    }
}
