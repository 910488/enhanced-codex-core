use serde::{Deserialize, Serialize};

use super::telemetry::{EnhancedEvent, EnhancedEventFields, EnhancedEventKind};

pub const MAX_AUTO_CONTINUATIONS: u8 = 2;

/// Deterministic unfinished-work signals already owned by Codex. Natural
/// language and LLM-as-judge are forbidden.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum UnfinishedSignal {
    NativePlanIncomplete,
    NativeSubagentWorkRemaining,
    StructuredPendingWork,
    IntentToContinue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutoContinuationBudget {
    pub used: u8,
    pub max: u8,
    reserved: Option<u8>,
}

impl Default for AutoContinuationBudget {
    fn default() -> Self {
        Self {
            used: 0,
            max: MAX_AUTO_CONTINUATIONS,
            reserved: None,
        }
    }
}

impl AutoContinuationBudget {
    pub fn reset_for_user_input(&mut self) {
        self.used = 0;
        self.reserved = None;
    }

    pub fn reserved_index(&self) -> Option<u8> {
        self.reserved
    }
}

/// A continuation slot that is not counted until the model stream starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReservedContinuation {
    pub index: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TurnStopContext {
    pub cancelled: bool,
    pub user_steer_pending: bool,
    pub unfinished: Vec<UnfinishedSignal>,
    /// Experimental. Default false; not implied by E5.
    pub intent_continuation_enabled: bool,
    /// True only for a natural model stop. Tool-call stops must not consult
    /// assistant text.
    pub natural_stop: bool,
    /// Assistant final text for the natural stop. Reasoning, tool results,
    /// and quoted content are not supplied here.
    pub assistant_final_text: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuationDecision {
    AllowStop,
    Continue { index: u8 },
    StopCancelled,
    StopUserSteer,
    StopBudgetExhausted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContinuationPlan {
    pub decision: ContinuationDecision,
    pub reservation: Option<ReservedContinuation>,
    pub events: Vec<EnhancedEvent>,
}

pub const CONTINUE_NUDGE: &str =
    "Continue the unfinished deterministic work. Do not wait for a new user message.";

/// Plan a continuation without consuming the budget. The caller must
/// `commit_continuation` after the model stream is created, or
/// `release_continuation` if stream creation fails.
pub fn plan_turn_stop(
    budget: &mut AutoContinuationBudget,
    context: &TurnStopContext,
) -> ContinuationPlan {
    if context.cancelled {
        budget.reserved = None;
        return ContinuationPlan {
            decision: ContinuationDecision::StopCancelled,
            reservation: None,
            events: Vec::new(),
        };
    }
    if context.user_steer_pending {
        budget.reset_for_user_input();
        return ContinuationPlan {
            decision: ContinuationDecision::StopUserSteer,
            reservation: None,
            events: Vec::new(),
        };
    }
    let mut unfinished = context
        .unfinished
        .iter()
        .copied()
        .filter(|signal| *signal != UnfinishedSignal::NativeSubagentWorkRemaining)
        .collect::<Vec<_>>();
    let intent_detected = context.intent_continuation_enabled
        && context.natural_stop
        && context
            .assistant_final_text
            .as_deref()
            .is_some_and(trailing_intent_to_continue);
    if intent_detected && !unfinished.contains(&UnfinishedSignal::IntentToContinue) {
        unfinished.push(UnfinishedSignal::IntentToContinue);
    }
    if unfinished.is_empty() {
        return ContinuationPlan {
            decision: ContinuationDecision::AllowStop,
            reservation: None,
            events: Vec::new(),
        };
    }
    let consumed = budget.used.max(budget.reserved.unwrap_or(0));
    if consumed >= budget.max {
        return ContinuationPlan {
            decision: ContinuationDecision::StopBudgetExhausted,
            reservation: None,
            events: vec![EnhancedEvent::new(
                EnhancedEventKind::ContinuationExhausted,
                EnhancedEventFields {
                    continuation_index: Some(consumed),
                    ..EnhancedEventFields::default()
                },
            )],
        };
    }
    let index = consumed.saturating_add(1);
    budget.reserved = Some(index);
    let mut events = Vec::new();
    if intent_detected {
        events.push(EnhancedEvent::new(
            EnhancedEventKind::IntentContinuationDetected,
            EnhancedEventFields {
                continuation_index: Some(index),
                observation_reason: Some("trailing_continue_immediately".into()),
                ..EnhancedEventFields::default()
            },
        ));
    }
    events.push(EnhancedEvent::new(
        EnhancedEventKind::ContinuationAllowed,
        EnhancedEventFields {
            continuation_index: Some(index),
            ..EnhancedEventFields::default()
        },
    ));
    ContinuationPlan {
        decision: ContinuationDecision::Continue { index },
        reservation: Some(ReservedContinuation { index }),
        events,
    }
}

/// Conservative English trailing-continue detector. First version ports only
/// the upstream short-sentence "will continue immediately" family. No Chinese
/// semantic classes and no LLM judge.
pub fn trailing_intent_to_continue(text: &str) -> bool {
    let last = last_unquoted_sentence(text);
    if last.is_empty() || last.len() > 80 {
        return false;
    }
    matches!(
        normalize_sentence(&last).as_str(),
        "i will continue immediately"
            | "i'll continue immediately"
            | "i will continue now"
            | "i'll continue now"
            | "let me continue immediately"
            | "continuing immediately"
    )
}

fn last_unquoted_sentence(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let last_line = trimmed
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("")
        .trim();
    if is_quoted(last_line) {
        return String::new();
    }
    let mut start = 0;
    for (index, character) in last_line.char_indices() {
        if matches!(character, '.' | '!' | '?') {
            let next = index + character.len_utf8();
            if !last_line[next..].trim().is_empty() {
                start = next;
            }
        }
    }
    last_line[start..].trim().to_string()
}

fn is_quoted(text: &str) -> bool {
    let trimmed = text.trim();
    let bytes = trimmed.as_bytes();
    bytes.len() >= 2
        && ((bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\''))
}

fn normalize_sentence(text: &str) -> String {
    text.trim()
        .trim_end_matches(['.', '!', ','])
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

pub fn commit_continuation(
    budget: &mut AutoContinuationBudget,
    reserved: ReservedContinuation,
) -> Result<(), ContinuationCommitError> {
    if budget.reserved != Some(reserved.index) {
        return Err(ContinuationCommitError::NoReservation);
    }
    budget.used = reserved.index;
    budget.reserved = None;
    Ok(())
}

pub fn release_continuation(budget: &mut AutoContinuationBudget) {
    budget.reserved = None;
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ContinuationCommitError {
    #[error("no reserved continuation attempt to commit")]
    NoReservation,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unfinished() -> TurnStopContext {
        TurnStopContext {
            unfinished: vec![UnfinishedSignal::StructuredPendingWork],
            ..TurnStopContext::default()
        }
    }

    fn commit_plan(
        budget: &mut AutoContinuationBudget,
        context: &TurnStopContext,
    ) -> ContinuationDecision {
        let plan = plan_turn_stop(budget, context);
        if let Some(reserved) = plan.reservation {
            commit_continuation(budget, reserved).unwrap();
        }
        plan.decision
    }

    #[test]
    fn finished_turn_does_not_continue() {
        let mut budget = AutoContinuationBudget::default();
        let plan = plan_turn_stop(&mut budget, &TurnStopContext::default());
        assert_eq!(plan.decision, ContinuationDecision::AllowStop);
        assert_eq!(budget.used, 0);
        assert!(plan.reservation.is_none());
    }

    #[test]
    fn unfinished_turn_continues_once() {
        let mut budget = AutoContinuationBudget::default();
        let plan = plan_turn_stop(&mut budget, &unfinished());
        assert_eq!(plan.decision, ContinuationDecision::Continue { index: 1 });
        assert_eq!(budget.used, 0);
        commit_continuation(&mut budget, plan.reservation.unwrap()).unwrap();
        assert_eq!(budget.used, 1);
    }

    #[test]
    fn stream_failure_does_not_consume_budget() {
        let mut budget = AutoContinuationBudget::default();
        let plan = plan_turn_stop(&mut budget, &unfinished());
        assert_eq!(plan.decision, ContinuationDecision::Continue { index: 1 });
        release_continuation(&mut budget);
        assert_eq!(budget.used, 0);
        let again = plan_turn_stop(&mut budget, &unfinished());
        assert_eq!(again.decision, ContinuationDecision::Continue { index: 1 });
        commit_continuation(&mut budget, again.reservation.unwrap()).unwrap();
        assert_eq!(budget.used, 1);
    }

    #[test]
    fn continuation_budget_never_exceeds_two() {
        let mut budget = AutoContinuationBudget::default();
        let ctx = unfinished();
        assert_eq!(
            commit_plan(&mut budget, &ctx),
            ContinuationDecision::Continue { index: 1 }
        );
        assert_eq!(
            commit_plan(&mut budget, &ctx),
            ContinuationDecision::Continue { index: 2 }
        );
        assert_eq!(
            commit_plan(&mut budget, &ctx),
            ContinuationDecision::StopBudgetExhausted
        );
        assert_eq!(budget.used, 2);
    }

    #[test]
    fn new_user_input_resets_without_deciding_stop() {
        let mut budget = AutoContinuationBudget::default();
        commit_plan(&mut budget, &unfinished());
        commit_plan(&mut budget, &unfinished());
        budget.reset_for_user_input();
        assert_eq!(budget.used, 0);
        assert_eq!(
            commit_plan(&mut budget, &unfinished()),
            ContinuationDecision::Continue { index: 1 }
        );
    }

    #[test]
    fn user_steer_resets_budget_and_does_not_continue_the_old_stage() {
        let mut budget = AutoContinuationBudget::default();
        commit_plan(&mut budget, &unfinished());
        let plan = plan_turn_stop(
            &mut budget,
            &TurnStopContext {
                user_steer_pending: true,
                unfinished: vec![UnfinishedSignal::StructuredPendingWork],
                ..TurnStopContext::default()
            },
        );
        assert_eq!(plan.decision, ContinuationDecision::StopUserSteer);
        assert_eq!(budget.used, 0);
    }

    #[test]
    fn cancel_prevents_auto_continuation() {
        let mut budget = AutoContinuationBudget::default();
        let plan = plan_turn_stop(
            &mut budget,
            &TurnStopContext {
                cancelled: true,
                unfinished: vec![UnfinishedSignal::NativeSubagentWorkRemaining],
                ..TurnStopContext::default()
            },
        );
        assert_eq!(plan.decision, ContinuationDecision::StopCancelled);
        assert_eq!(budget.used, 0);
    }

    #[test]
    fn subagent_completion_does_not_create_infinite_continue() {
        let mut budget = AutoContinuationBudget::default();
        let running = TurnStopContext {
            unfinished: vec![UnfinishedSignal::NativeSubagentWorkRemaining],
            ..TurnStopContext::default()
        };
        plan_turn_stop(&mut budget, &running);
        let completed = TurnStopContext::default();
        let plan = plan_turn_stop(&mut budget, &completed);
        assert_eq!(plan.decision, ContinuationDecision::AllowStop);
        let again = plan_turn_stop(&mut budget, &completed);
        assert_eq!(again.decision, ContinuationDecision::AllowStop);
        assert!(budget.used <= MAX_AUTO_CONTINUATIONS);
    }

    fn intent_stop(text: &str) -> TurnStopContext {
        TurnStopContext {
            intent_continuation_enabled: true,
            natural_stop: true,
            assistant_final_text: Some(text.into()),
            ..TurnStopContext::default()
        }
    }

    #[test]
    fn intent_continuation_default_off_does_not_add_text_continuation() {
        let mut budget = AutoContinuationBudget::default();
        let plan = plan_turn_stop(
            &mut budget,
            &TurnStopContext {
                natural_stop: true,
                assistant_final_text: Some("I will continue immediately.".into()),
                ..TurnStopContext::default()
            },
        );
        assert_eq!(plan.decision, ContinuationDecision::AllowStop);
        assert_eq!(budget.used, 0);
    }

    #[test]
    fn intent_continuation_matches_conservative_english_and_shares_budget() {
        assert!(trailing_intent_to_continue(
            "The tests passed.\nI will continue immediately."
        ));
        assert!(trailing_intent_to_continue("I'll continue immediately."));
        assert!(!trailing_intent_to_continue("我将立即继续。"));
        assert!(!trailing_intent_to_continue(
            "\"I will continue immediately.\""
        ));
        assert!(!trailing_intent_to_continue(
            "I will continue working on this tomorrow."
        ));
        let mut budget = AutoContinuationBudget::default();
        assert_eq!(
            commit_plan(&mut budget, &intent_stop("I will continue immediately.")),
            ContinuationDecision::Continue { index: 1 }
        );
        assert_eq!(
            commit_plan(&mut budget, &unfinished()),
            ContinuationDecision::Continue { index: 2 }
        );
        assert_eq!(
            commit_plan(&mut budget, &intent_stop("I'll continue now.")),
            ContinuationDecision::StopBudgetExhausted
        );
        assert_eq!(budget.used, 2);
    }

    #[test]
    fn intent_counts_only_after_stream_create_and_yields_to_cancel_and_steer() {
        let mut budget = AutoContinuationBudget::default();
        let plan = plan_turn_stop(&mut budget, &intent_stop("I will continue immediately."));
        assert_eq!(plan.decision, ContinuationDecision::Continue { index: 1 });
        assert_eq!(budget.used, 0);
        release_continuation(&mut budget);
        assert_eq!(budget.used, 0);
        let again = plan_turn_stop(&mut budget, &intent_stop("I will continue immediately."));
        commit_continuation(&mut budget, again.reservation.unwrap()).unwrap();
        assert_eq!(budget.used, 1);

        let cancelled = plan_turn_stop(
            &mut budget,
            &TurnStopContext {
                cancelled: true,
                intent_continuation_enabled: true,
                natural_stop: true,
                assistant_final_text: Some("I will continue immediately.".into()),
                ..TurnStopContext::default()
            },
        );
        assert_eq!(cancelled.decision, ContinuationDecision::StopCancelled);

        let steered = plan_turn_stop(
            &mut budget,
            &TurnStopContext {
                user_steer_pending: true,
                intent_continuation_enabled: true,
                natural_stop: true,
                assistant_final_text: Some("I will continue immediately.".into()),
                ..TurnStopContext::default()
            },
        );
        assert_eq!(steered.decision, ContinuationDecision::StopUserSteer);
        assert_eq!(budget.used, 0);
    }

    #[test]
    fn subagent_wait_alone_does_not_continue_even_with_intent_flag() {
        let mut budget = AutoContinuationBudget::default();
        let plan = plan_turn_stop(
            &mut budget,
            &TurnStopContext {
                unfinished: vec![UnfinishedSignal::NativeSubagentWorkRemaining],
                intent_continuation_enabled: true,
                natural_stop: true,
                assistant_final_text: Some("waiting on the subagent".into()),
                ..TurnStopContext::default()
            },
        );
        assert_eq!(plan.decision, ContinuationDecision::AllowStop);
    }

    #[test]
    fn reasoning_or_non_natural_stop_text_is_not_consulted() {
        let mut budget = AutoContinuationBudget::default();
        let tool_stop = plan_turn_stop(
            &mut budget,
            &TurnStopContext {
                intent_continuation_enabled: true,
                natural_stop: false,
                assistant_final_text: Some("I will continue immediately.".into()),
                ..TurnStopContext::default()
            },
        );
        assert_eq!(tool_stop.decision, ContinuationDecision::AllowStop);
    }
}
