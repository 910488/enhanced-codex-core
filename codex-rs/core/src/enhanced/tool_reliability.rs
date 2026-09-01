use std::collections::HashMap;

use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use sha2::Digest;
use sha2::Sha256;

use super::telemetry::EnhancedEvent;
use super::telemetry::EnhancedEventFields;
use super::telemetry::EnhancedEventKind;
use super::telemetry::hash_identifier;

/// Identity of a provider-issued tool call. The ledger is owned by a Codex
/// thread/session and must be restored on resume.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderToolCallIdentity {
    pub provider_call_id: String,
    pub tool_name: String,
    pub argument_fingerprint: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ToolCallResolution {
    InFlight,
    Executed,
    Failed,
    Cancelled,
    SyntheticDuplicate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HandledToolCall {
    pub identity: ProviderToolCallIdentity,
    pub resolution: ToolCallResolution,
    pub original_admitted: bool,
    pub synthetic_duplicate_emitted: bool,
    pub original_result_delivered: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallLedger {
    handled: HashMap<String, HandledToolCall>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmitDecision {
    Execute,
    SuppressDuplicate { message: String },
    FailClosed { message: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LateResultDecision {
    Accept,
    Suppress,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolReliabilityOutcome {
    pub decision: AdmitDecision,
    pub events: Vec<EnhancedEvent>,
}

impl ToolCallLedger {
    pub fn new() -> Self {
        Self {
            handled: HashMap::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.handled.is_empty()
    }

    pub fn get(&self, provider_call_id: &str) -> Option<&HandledToolCall> {
        self.handled.get(provider_call_id)
    }

    pub fn fingerprint_arguments(tool_name: &str, arguments: &Value) -> String {
        let canonical = canonical_json(arguments);
        let mut hasher = Sha256::new();
        hasher.update(tool_name.as_bytes());
        hasher.update([0]);
        hasher.update(canonical.as_bytes());
        format!("sha256:{}", hex::encode(hasher.finalize()))
    }

    pub fn identity(
        provider_call_id: impl Into<String>,
        tool_name: &str,
        arguments: &Value,
    ) -> ProviderToolCallIdentity {
        let tool_name = tool_name.to_string();
        ProviderToolCallIdentity {
            argument_fingerprint: Self::fingerprint_arguments(&tool_name, arguments),
            provider_call_id: provider_call_id.into(),
            tool_name,
        }
    }

    /// First observation reserves the call as in-flight so a resume cannot
    /// re-run a side-effecting tool. It is not marked Executed until the
    /// original attempt completes.
    pub fn admit(&mut self, identity: ProviderToolCallIdentity) -> ToolReliabilityOutcome {
        if let Some(existing) = self.handled.get_mut(&identity.provider_call_id) {
            if existing.identity.tool_name != identity.tool_name
                || existing.identity.argument_fingerprint != identity.argument_fingerprint
            {
                let events = vec![event(
                    EnhancedEventKind::ToolCallIdCollision,
                    &identity.provider_call_id,
                )];
                return ToolReliabilityOutcome {
                    decision: AdmitDecision::FailClosed {
                        message: format!(
                            "provider protocol collision: call id {} reused with a different tool or argument fingerprint",
                            identity.provider_call_id
                        ),
                    },
                    events,
                };
            }
            existing.synthetic_duplicate_emitted = true;
            let events = vec![
                event(
                    EnhancedEventKind::ToolDuplicateDetected,
                    &identity.provider_call_id,
                ),
                event(
                    EnhancedEventKind::ToolDuplicateSuppressed,
                    &identity.provider_call_id,
                ),
            ];
            return ToolReliabilityOutcome {
                decision: AdmitDecision::SuppressDuplicate {
                    message: format!(
                        "duplicate provider tool call {} was not executed",
                        identity.provider_call_id
                    ),
                },
                events,
            };
        }

        self.handled.insert(
            identity.provider_call_id.clone(),
            HandledToolCall {
                identity,
                resolution: ToolCallResolution::InFlight,
                original_admitted: true,
                synthetic_duplicate_emitted: false,
                original_result_delivered: false,
            },
        );
        ToolReliabilityOutcome {
            decision: AdmitDecision::Execute,
            events: Vec::new(),
        }
    }

    pub fn mark_resolution(&mut self, provider_call_id: &str, resolution: ToolCallResolution) {
        if matches!(resolution, ToolCallResolution::SyntheticDuplicate) {
            // Synthetic duplicate is a result sent to the model for a replayed
            // call. It must not rewrite the original execution record.
            if let Some(existing) = self.handled.get_mut(provider_call_id) {
                existing.synthetic_duplicate_emitted = true;
            }
            return;
        }
        if let Some(existing) = self.handled.get_mut(provider_call_id) {
            existing.resolution = resolution;
        }
    }

    pub fn complete_original(&mut self, provider_call_id: &str) -> LateResultDecision {
        match self.handled.get_mut(provider_call_id) {
            Some(handled) if handled.original_admitted && !handled.original_result_delivered => {
                handled.original_result_delivered = true;
                LateResultDecision::Accept
            }
            Some(_) => LateResultDecision::Suppress,
            None => LateResultDecision::Suppress,
        }
    }

    pub fn ingest_late_result(&self, provider_call_id: &str) -> LateResultDecision {
        match self.handled.get(provider_call_id) {
            Some(handled) if handled.synthetic_duplicate_emitted => LateResultDecision::Suppress,
            Some(handled) if handled.original_result_delivered => LateResultDecision::Suppress,
            Some(_) => LateResultDecision::Accept,
            None => LateResultDecision::Suppress,
        }
    }

    /// Results the model is allowed to see. Success is never synthesized.
    pub fn synthetic_duplicate_result(message: &str) -> SyntheticToolResult {
        SyntheticToolResult::failure(SyntheticResultKind::Duplicate, message)
    }

    pub fn synthetic_aborted_result() -> SyntheticToolResult {
        SyntheticToolResult::failure(SyntheticResultKind::Aborted, "aborted")
    }

    pub fn to_durable_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or_else(|_| Value::Object(Default::default()))
    }

    pub fn from_durable_json(value: &Value) -> Result<Self, LedgerError> {
        serde_json::from_value(value.clone())
            .map_err(|error| LedgerError::Invalid(error.to_string()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SyntheticResultKind {
    Duplicate,
    Aborted,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyntheticToolResult {
    pub kind: SyntheticResultKind,
    pub message: String,
}

impl SyntheticToolResult {
    pub fn failure(kind: SyntheticResultKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Synthetic results are never success. The field is intentionally not
    /// settable so callers cannot construct a false-success payload.
    pub fn success(&self) -> bool {
        false
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LedgerError {
    #[error("tool-call ledger is invalid: {0}")]
    Invalid(String),
}

fn event(kind: EnhancedEventKind, call_id: &str) -> EnhancedEvent {
    EnhancedEvent::new(
        kind,
        EnhancedEventFields {
            call_id_hash: Some(hash_identifier(call_id)),
            ..EnhancedEventFields::default()
        },
    )
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut keys = map.keys().cloned().collect::<Vec<_>>();
            keys.sort();
            let mut out = String::from("{");
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key).unwrap_or_else(|_| "\"\"".into()));
                out.push(':');
                out.push_str(&canonical_json(&map[key]));
            }
            out.push('}');
            out
        }
        Value::Array(items) => {
            let mut out = String::from("[");
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&canonical_json(item));
            }
            out.push(']');
            out
        }
        other => serde_json::to_string(other).unwrap_or_else(|_| "null".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args() -> Value {
        json!({"path": "a.rs", "n": 1})
    }

    #[test]
    fn duplicate_call_same_id_same_args_executes_once() {
        let mut ledger = ToolCallLedger::new();
        let identity = ToolCallLedger::identity("call-1", "shell", &args());
        assert_eq!(
            ledger.admit(identity.clone()).decision,
            AdmitDecision::Execute
        );
        let second = ledger.admit(identity);
        assert!(matches!(
            second.decision,
            AdmitDecision::SuppressDuplicate { .. }
        ));
        assert_eq!(
            ledger.get("call-1").unwrap().resolution,
            ToolCallResolution::InFlight
        );
        ledger.mark_resolution("call-1", ToolCallResolution::Executed);
        assert_eq!(
            ledger.get("call-1").unwrap().resolution,
            ToolCallResolution::Executed
        );
        assert!(!ToolCallLedger::synthetic_duplicate_result("duplicate").success());
    }

    #[test]
    fn duplicate_call_same_id_different_args_fails_closed() {
        let mut ledger = ToolCallLedger::new();
        ledger.admit(ToolCallLedger::identity("call-1", "shell", &args()));
        let other = ToolCallLedger::identity("call-1", "shell", &json!({"path": "b.rs"}));
        let outcome = ledger.admit(other);
        assert!(matches!(outcome.decision, AdmitDecision::FailClosed { .. }));
        assert!(
            outcome
                .events
                .iter()
                .any(|event| event.kind == EnhancedEventKind::ToolCallIdCollision)
        );
    }

    #[test]
    fn late_real_result_after_synthetic_duplicate_is_suppressed() {
        let mut ledger = ToolCallLedger::new();
        let identity = ToolCallLedger::identity("call-1", "shell", &args());
        ledger.admit(identity.clone());
        ledger.admit(identity);
        assert_eq!(
            ledger.ingest_late_result("call-1"),
            LateResultDecision::Suppress
        );
        assert_eq!(
            ledger.complete_original("call-1"),
            LateResultDecision::Accept
        );
        assert_eq!(
            ledger.complete_original("call-1"),
            LateResultDecision::Suppress
        );
    }

    #[test]
    fn resume_does_not_reexecute_handled_provider_call() {
        let mut ledger = ToolCallLedger::new();
        ledger.admit(ToolCallLedger::identity("call-1", "shell", &args()));
        let restored = ToolCallLedger::from_durable_json(&ledger.to_durable_json()).unwrap();
        let mut restored = restored;
        let outcome = restored.admit(ToolCallLedger::identity("call-1", "shell", &args()));
        assert!(matches!(
            outcome.decision,
            AdmitDecision::SuppressDuplicate { .. }
        ));
    }

    #[test]
    fn parallel_calls_with_unique_ids_are_unchanged() {
        let mut ledger = ToolCallLedger::new();
        let first = ledger.admit(ToolCallLedger::identity("a", "shell", &args()));
        let second = ledger.admit(ToolCallLedger::identity("b", "shell", &args()));
        assert_eq!(first.decision, AdmitDecision::Execute);
        assert_eq!(second.decision, AdmitDecision::Execute);
        assert_eq!(ledger.handled.len(), 2);
    }

    #[test]
    fn fingerprint_ignores_object_key_order() {
        let left = ToolCallLedger::fingerprint_arguments("shell", &json!({"b": 1, "a": 2}));
        let right = ToolCallLedger::fingerprint_arguments("shell", &json!({"a": 2, "b": 1}));
        assert_eq!(left, right);
    }
}
