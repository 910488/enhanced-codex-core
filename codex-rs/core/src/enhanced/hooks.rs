use serde_json::Value;

use super::bounded_continuation::AutoContinuationBudget;
use super::bounded_continuation::ContinuationDecision;
use super::bounded_continuation::ContinuationPlan;
use super::bounded_continuation::ReservedContinuation;
use super::bounded_continuation::TurnStopContext;
use super::bounded_continuation::commit_continuation;
use super::bounded_continuation::plan_turn_stop;
use super::bounded_continuation::release_continuation;
use super::config::EnhancedRuntimeFeatures;
use super::context_pruner::ModelVisibleSurface;
use super::context_pruner::ToolResultPrunePolicy;
use super::context_recovery::OverflowAttempt;
use super::context_recovery::OverflowPlan;
use super::context_recovery::PressurePlan;
use super::context_recovery::plan_context_pressure;
use super::context_recovery::plan_overflow_retry;
use super::telemetry::EnhancedEvent;
use super::telemetry::EnhancedEventFields;
use super::telemetry::EnhancedEventKind;
use super::telemetry::MemoryTelemetry;
use super::telemetry::hash_identifier;
use super::tool_reliability::AdmitDecision;
use super::tool_reliability::LateResultDecision;
use super::tool_reliability::ProviderToolCallIdentity;
use super::tool_reliability::ToolCallLedger;
use super::tool_reliability::ToolCallResolution;
use super::tool_reliability::ToolReliabilityOutcome;

/// When a feature is off the hook must not invent a Codex decision. The
/// caller falls through to unmodified upstream behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookDecision<T> {
    DeferToUpstream,
    Handled(T),
}

/// Seams Codex should call. E0 is a strict no-op: every method returns
/// `DeferToUpstream` and does not mutate ledgers, budgets, or telemetry.
#[derive(Debug, Clone)]
pub struct EnhancedTurnHooks {
    pub features: EnhancedRuntimeFeatures,
    pub ledger: ToolCallLedger,
    pub continuation: AutoContinuationBudget,
    pub prune_policy: ToolResultPrunePolicy,
    pub overflow_retries_used: u8,
}

impl EnhancedTurnHooks {
    pub fn new(features: EnhancedRuntimeFeatures) -> Self {
        Self {
            features,
            ledger: ToolCallLedger::new(),
            continuation: AutoContinuationBudget::default(),
            prune_policy: ToolResultPrunePolicy::default(),
            overflow_retries_used: 0,
        }
    }

    pub fn on_new_user_input(&mut self) {
        self.continuation.reset_for_user_input();
        self.overflow_retries_used = 0;
    }

    pub fn on_assistant_success(&mut self) {
        self.overflow_retries_used = 0;
    }

    pub fn on_turn_idle(&mut self) {
        self.overflow_retries_used = 0;
        release_continuation(&mut self.continuation);
    }

    pub fn admit_tool_call(
        &mut self,
        identity: ProviderToolCallIdentity,
        telemetry: &mut MemoryTelemetry,
    ) -> HookDecision<AdmitDecision> {
        if !self.features.qwen_tool_reliability {
            return HookDecision::DeferToUpstream;
        }
        let call_id_hash = hash_identifier(&identity.provider_call_id);
        let ToolReliabilityOutcome { decision, events } = self.ledger.admit(identity);
        for event in events {
            telemetry.emit(event);
        }
        if matches!(decision, AdmitDecision::Execute) {
            telemetry.emit(EnhancedEvent::new(
                EnhancedEventKind::ToolCallAdmitted,
                EnhancedEventFields {
                    call_id_hash: Some(call_id_hash),
                    outcome: Some("execute".into()),
                    ..EnhancedEventFields::default()
                },
            ));
        }
        HookDecision::Handled(decision)
    }

    pub fn complete_original_tool_result(
        &mut self,
        provider_call_id: &str,
    ) -> HookDecision<LateResultDecision> {
        if !self.features.qwen_tool_reliability {
            return HookDecision::DeferToUpstream;
        }
        HookDecision::Handled(self.ledger.complete_original(provider_call_id))
    }

    pub fn ingest_late_tool_result(
        &self,
        provider_call_id: &str,
    ) -> HookDecision<LateResultDecision> {
        if !self.features.qwen_tool_reliability {
            return HookDecision::DeferToUpstream;
        }
        HookDecision::Handled(self.ledger.ingest_late_result(provider_call_id))
    }

    pub fn mark_tool_resolution(&mut self, provider_call_id: &str, resolution: ToolCallResolution) {
        if self.features.qwen_tool_reliability {
            self.ledger.mark_resolution(provider_call_id, resolution);
        }
    }

    pub fn plan_pressure(
        &self,
        surface: &ModelVisibleSurface,
        context_window_tokens: u64,
        compact_threshold_tokens: u64,
        telemetry: &mut MemoryTelemetry,
    ) -> HookDecision<PressurePlan> {
        if !self.features.deepseek_context_recovery {
            return HookDecision::DeferToUpstream;
        }
        let plan = plan_context_pressure(
            surface,
            self.prune_policy,
            context_window_tokens,
            compact_threshold_tokens,
        );
        telemetry.emit(EnhancedEvent::new(
            EnhancedEventKind::ContextPressureChecked,
            EnhancedEventFields {
                before_token_estimate: Some(plan.prune.before_token_estimate),
                after_token_estimate: Some(plan.prune.after_token_estimate),
                chars_removed: Some(plan.prune.bytes_removed),
                outcome: Some(
                    match (plan.prune.rewritten, plan.compact) {
                        (
                            _,
                            super::context_recovery::CompactDecision::RunNativeCodexLocalCompact,
                        ) => "native_compact",
                        (true, super::context_recovery::CompactDecision::Skip) => "pruned",
                        (false, super::context_recovery::CompactDecision::Skip) => "unchanged",
                    }
                    .into(),
                ),
                ..EnhancedEventFields::default()
            },
        ));
        for event in &plan.events {
            telemetry.emit(event.clone());
        }
        HookDecision::Handled(plan)
    }

    pub fn plan_overflow(
        &mut self,
        before: &ModelVisibleSurface,
        after: &ModelVisibleSurface,
        cancelled: bool,
        telemetry: &mut MemoryTelemetry,
    ) -> HookDecision<OverflowPlan> {
        if !self.features.deepseek_context_recovery {
            return HookDecision::DeferToUpstream;
        }
        let plan = plan_overflow_retry(
            OverflowAttempt {
                generation_before: before.generation,
                retries_used: self.overflow_retries_used,
                cancelled,
            },
            before,
            after,
        );
        if let super::context_recovery::OverflowDecision::Retry { retry_index } = plan.decision {
            self.overflow_retries_used = retry_index;
        }
        for event in &plan.events {
            telemetry.emit(event.clone());
        }
        HookDecision::Handled(plan)
    }

    pub fn on_turn_stop(
        &mut self,
        context: &TurnStopContext,
        telemetry: &mut MemoryTelemetry,
    ) -> HookDecision<ContinuationPlan> {
        if !self.features.qwen_bounded_continuation {
            return HookDecision::DeferToUpstream;
        }
        if context.user_steer_pending {
            self.overflow_retries_used = 0;
        }
        let plan = plan_turn_stop(&mut self.continuation, context);
        telemetry.emit(EnhancedEvent::new(
            EnhancedEventKind::ContinuationEvaluated,
            EnhancedEventFields {
                unfinished_signal_count: Some(context.unfinished.len() as u64),
                outcome: Some(
                    match plan.decision {
                        ContinuationDecision::AllowStop => "allow_stop",
                        ContinuationDecision::Continue { .. } => "continue",
                        ContinuationDecision::StopCancelled => "cancelled",
                        ContinuationDecision::StopUserSteer => "user_steer",
                        ContinuationDecision::StopBudgetExhausted => "budget_exhausted",
                    }
                    .into(),
                ),
                ..EnhancedEventFields::default()
            },
        ));
        for event in &plan.events {
            telemetry.emit(event.clone());
        }
        HookDecision::Handled(plan)
    }

    pub fn commit_continuation(
        &mut self,
        reserved: ReservedContinuation,
    ) -> Result<(), super::bounded_continuation::ContinuationCommitError> {
        commit_continuation(&mut self.continuation, reserved)
    }

    pub fn release_continuation(&mut self) {
        release_continuation(&mut self.continuation);
    }

    pub fn restore_ledger(
        &mut self,
        value: &Value,
    ) -> Result<(), super::tool_reliability::LedgerError> {
        self.ledger = ToolCallLedger::from_durable_json(value)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::AblationProfile;
    use super::super::context_recovery::OverflowDecision;
    use super::super::context_recovery::OverflowPlan;
    use super::super::tool_reliability::ToolCallLedger;
    use super::*;
    use serde_json::json;

    #[test]
    fn e0_is_strict_defer_to_upstream() {
        let mut hooks = EnhancedTurnHooks::new(AblationProfile::E0.features());
        let mut telemetry = MemoryTelemetry::default();
        let identity = ToolCallLedger::identity("c", "shell", &json!({"a":1}));
        assert_eq!(
            hooks.admit_tool_call(identity.clone(), &mut telemetry),
            HookDecision::DeferToUpstream
        );
        assert_eq!(
            hooks.admit_tool_call(identity, &mut telemetry),
            HookDecision::DeferToUpstream
        );
        assert!(hooks.ledger.is_empty());
        assert!(telemetry.events.is_empty());
        assert!(matches!(
            hooks.plan_pressure(&ModelVisibleSurface::default(), 100, 10, &mut telemetry),
            HookDecision::DeferToUpstream
        ));
    }

    #[test]
    fn e1_suppresses_duplicates() {
        let mut hooks = EnhancedTurnHooks::new(AblationProfile::E1.features());
        let mut telemetry = MemoryTelemetry::default();
        let identity = ToolCallLedger::identity("c", "shell", &json!({"a":1}));
        hooks.admit_tool_call(identity.clone(), &mut telemetry);
        assert!(matches!(
            hooks.admit_tool_call(identity, &mut telemetry),
            HookDecision::Handled(AdmitDecision::SuppressDuplicate { .. })
        ));
    }

    #[test]
    fn overflow_budget_resets_on_new_user_input() {
        let mut hooks = EnhancedTurnHooks::new(AblationProfile::E2.features());
        let mut telemetry = MemoryTelemetry::default();
        let before = ModelVisibleSurface {
            item_token_estimates: Vec::new(),
            generation: 1,
            items: vec![],
        };
        let after = ModelVisibleSurface {
            item_token_estimates: Vec::new(),
            generation: 2,
            items: vec![],
        };
        hooks.overflow_retries_used = 1;
        let refused = hooks.plan_overflow(&before, &after, false, &mut telemetry);
        assert!(matches!(
            refused,
            HookDecision::Handled(OverflowPlan {
                decision: OverflowDecision::PreserveOriginalError,
                ..
            })
        ));
        hooks.on_new_user_input();
        assert_eq!(hooks.overflow_retries_used, 0);
        assert_eq!(hooks.continuation.used, 0);
    }

    #[test]
    fn overflow_budget_resets_after_successful_assistant_turn() {
        let mut hooks = EnhancedTurnHooks::new(AblationProfile::E2.features());
        hooks.overflow_retries_used = 1;
        hooks.on_assistant_success();
        assert_eq!(hooks.overflow_retries_used, 0);
        hooks.overflow_retries_used = 1;
        hooks.on_turn_idle();
        assert_eq!(hooks.overflow_retries_used, 0);
    }
}
