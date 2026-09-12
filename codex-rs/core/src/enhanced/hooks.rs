use serde_json::Value;

use super::bounded_continuation::{
    commit_continuation, plan_turn_stop, release_continuation, AutoContinuationBudget,
    ContinuationDecision, ContinuationPlan, ReservedContinuation, TurnStopContext,
};
use super::config::EnhancedRuntimeFeatures;
use super::context_pruner::{ModelVisibleSurface, ToolResultPrunePolicy};
use super::context_recovery::{
    plan_context_pressure, plan_overflow_retry, OverflowAttempt, OverflowPlan, PressurePlan,
};
use super::telemetry::{
    hash_identifier, EnhancedEvent, EnhancedEventFields, EnhancedEventKind, MemoryTelemetry,
};
use super::tool_observation::{
    observation_events, ObservationDecision, ObservationInput, ObservationState,
};
use super::tool_reliability::{
    AdmitDecision, LateResultDecision, ProviderToolCallIdentity, ToolCallLedger,
    ToolCallResolution, ToolReliabilityOutcome,
};

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
    pub observation: ObservationState,
}

impl EnhancedTurnHooks {
    pub fn new(features: EnhancedRuntimeFeatures) -> Self {
        Self {
            features,
            ledger: ToolCallLedger::new(),
            continuation: AutoContinuationBudget::default(),
            prune_policy: ToolResultPrunePolicy::default(),
            overflow_retries_used: 0,
            observation: ObservationState::new(),
        }
    }

    pub fn on_new_user_input(&mut self) {
        self.continuation.reset_for_user_input();
        self.overflow_retries_used = 0;
        self.observation.reset();
    }

    pub fn on_assistant_success(&mut self) {
        self.overflow_retries_used = 0;
    }

    pub fn on_turn_idle(&mut self) {
        self.overflow_retries_used = 0;
        release_continuation(&mut self.continuation);
        self.observation.reset();
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

    /// Non-blocking observation at original-result completion. Never stops
    /// the turn, never caps tool count, and never wraps up.
    pub fn observe_tool_result(
        &mut self,
        call_id: &str,
        input: ObservationInput,
        telemetry: &mut MemoryTelemetry,
    ) -> ObservationDecision {
        let mut decision = self
            .observation
            .observe(input, self.features.repetition_notice);
        for event in observation_events(&decision, call_id) {
            telemetry.emit(event);
        }
        for deferred in self.observation.take_deferred_notices() {
            for event in observation_events(&deferred, call_id) {
                telemetry.emit(event);
            }
            decision.emit_diagnostic |= deferred.emit_diagnostic;
            decision.standalone_notice |= deferred.standalone_notice;
        }
        decision
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
            self.observation.reset();
        }
        if context.cancelled {
            self.observation.reset();
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
        // Resume does not restore ended-turn observation counts.
        self.observation.reset();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::AblationProfile;
    use super::super::context_recovery::{OverflowDecision, OverflowPlan};
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

    #[test]
    fn observation_is_diagnostic_only_unless_repetition_notice_is_on() {
        use super::super::tool_observation::{
            ObservationInput, ObservationResultStatus, ObservationReason,
        };
        use super::super::tool_reliability::ToolKind;

        fn sample(index: u64) -> ObservationInput {
            ObservationInput {
                tool_name: "apply_patch".into(),
                tool_kind: ToolKind::Function,
                namespace: String::new(),
                input_fingerprint: "in".into(),
                original_result_fingerprint: "ok".into(),
                dispatch_index: index,
                result_status: ObservationResultStatus::Success,
                native_wait_poll: false,
            }
        }

        let mut hooks = EnhancedTurnHooks::new(AblationProfile::E5.features());
        let mut telemetry = MemoryTelemetry::default();
        hooks.observe_tool_result("c1", sample(0), &mut telemetry);
        hooks.observe_tool_result("c2", sample(1), &mut telemetry);
        let third = hooks.observe_tool_result("c3", sample(2), &mut telemetry);
        assert!(third.emit_diagnostic);
        assert!(!third.emit_model_notice);
        assert!(!hooks.features.repetition_notice);

        hooks.features.repetition_notice = true;
        hooks.observation.reset();
        hooks.observe_tool_result("c1", sample(0), &mut telemetry);
        hooks.observe_tool_result("c2", sample(1), &mut telemetry);
        let noticed = hooks.observe_tool_result("c3", sample(2), &mut telemetry);
        assert!(noticed.emit_model_notice);
        assert_eq!(noticed.reason, ObservationReason::RepetitionObserved);

        let restored = ToolCallLedger::identity("c", "shell", &json!({"a": 1}));
        let mut hooks = EnhancedTurnHooks::new(AblationProfile::E1.features());
        hooks.admit_tool_call(restored, &mut telemetry);
        let snapshot = hooks.ledger.to_durable_json();
        hooks.observe_tool_result("c1", sample(0), &mut telemetry);
        hooks.observe_tool_result("c2", sample(1), &mut telemetry);
        hooks.restore_ledger(&snapshot).unwrap();
        let after_resume = hooks.observe_tool_result("c3", sample(2), &mut telemetry);
        assert_eq!(after_resume.consecutive_count, 1);
    }

    #[test]
    fn out_of_order_observe_path_drains_a_standalone_notice() {
        use super::super::tool_observation::{
            ObservationInput, ObservationResultStatus, ObservationReason,
        };
        use super::super::tool_reliability::ToolKind;

        fn sample(index: u64) -> ObservationInput {
            ObservationInput {
                tool_name: "apply_patch".into(),
                tool_kind: ToolKind::Function,
                namespace: String::new(),
                input_fingerprint: "in".into(),
                original_result_fingerprint: "ok".into(),
                dispatch_index: index,
                result_status: ObservationResultStatus::Success,
                native_wait_poll: false,
            }
        }

        let mut hooks = EnhancedTurnHooks::new(AblationProfile::E5.features());
        hooks.features.repetition_notice = true;
        let mut telemetry = MemoryTelemetry::default();
        hooks.observe_tool_result("c0", sample(0), &mut telemetry);
        let late_high = hooks.observe_tool_result("c2", sample(2), &mut telemetry);
        assert!(!late_high.emit_model_notice);
        let middle = hooks.observe_tool_result("c1", sample(1), &mut telemetry);
        assert!(
            !middle.emit_model_notice,
            "must not attach the notice to the late-completing middle result"
        );
        assert!(
            middle.standalone_notice,
            "observe_tool_result must surface the deferred standalone notice"
        );
        assert!(
            middle.emit_diagnostic,
            "the deferred diagnostic must reach the observe path"
        );
        assert!(hooks.observation.take_deferred_notices().is_empty());
        assert_eq!(
            telemetry.count(super::super::telemetry::EnhancedEventKind::ToolRepetitionObserved),
            1
        );
        assert_eq!(
            telemetry.count(super::super::telemetry::EnhancedEventKind::ToolRepetitionNoticeAppended),
            1
        );
        assert_eq!(middle.reason, ObservationReason::BelowThreshold);
    }
}
