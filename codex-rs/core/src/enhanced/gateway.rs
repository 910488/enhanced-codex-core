use serde::{Deserialize, Serialize};

/// Operations a thin Vellum provider gateway may perform. Agent-loop
/// authority stays in Codex / Enhanced Codex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GatewayOperation {
    AuthInjection,
    EndpointMapping,
    ResponsesChatTranslation,
    SseNormalization,
    ProviderErrorMapping,
    UsageAccounting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ForbiddenGatewayOperation {
    SemanticCompaction,
    CanonicalCheckpoint,
    HistoryTaskRecovery,
    ActionRecoveryPrompt,
    AutoContinuation,
    LoopDecision,
    SubagentDecision,
    SemanticFrontier,
    RollingLocal,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GatewayMutationGuard {
    pub allowed: Vec<GatewayOperation>,
    pub forbidden_attempts: Vec<ForbiddenGatewayOperation>,
}

impl GatewayMutationGuard {
    pub fn record_allowed(&mut self, operation: GatewayOperation) {
        self.allowed.push(operation);
    }

    pub fn record_forbidden(&mut self, operation: ForbiddenGatewayOperation) {
        self.forbidden_attempts.push(operation);
    }

    pub fn legacy_harness_mutation_count(&self) -> u64 {
        self.forbidden_attempts.len() as u64
    }

    pub fn assert_isolated(&self) -> Result<(), GatewayIsolationError> {
        let count = self.legacy_harness_mutation_count();
        if count == 0 {
            Ok(())
        } else {
            Err(GatewayIsolationError::LegacyMutation { count })
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GatewayIsolationError {
    #[error("legacy Vellum harness mutation count is {count}, expected 0")]
    LegacyMutation { count: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enhanced_path_requires_zero_legacy_mutations() {
        let mut guard = GatewayMutationGuard::default();
        guard.record_allowed(GatewayOperation::AuthInjection);
        guard.record_allowed(GatewayOperation::ProviderErrorMapping);
        assert_eq!(guard.legacy_harness_mutation_count(), 0);
        assert!(guard.assert_isolated().is_ok());
        guard.record_forbidden(ForbiddenGatewayOperation::CanonicalCheckpoint);
        assert_eq!(guard.legacy_harness_mutation_count(), 1);
        assert!(guard.assert_isolated().is_err());
    }
}
