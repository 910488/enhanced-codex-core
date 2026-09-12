use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use super::telemetry::{hash_identifier, EnhancedEvent, EnhancedEventFields, EnhancedEventKind};
use super::tool_reliability::ToolKind;

/// Native wait/poll tools excluded from repetition observation. Matching is
/// by native tool semantics, not a name suffix on unknown tools.
pub const NATIVE_WAIT_POLL_TOOLS: &[(&str, ToolKind)] = &[("wait", ToolKind::Function)];

pub const REPETITION_NOTICE: &str = "Notice: the same tool, input, and original result have now occurred three times in a row. Consider a different approach.";

pub const REPETITION_THRESHOLD: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObservationResultStatus {
    Success,
    Failed,
    Cancelled,
    Synthetic,
}

impl ObservationResultStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Synthetic => "synthetic",
        }
    }
}

/// Explicit inputs at tool-result completion. Tests feed recorded fixture
/// events through this shape without a live model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationInput {
    pub tool_name: String,
    pub tool_kind: ToolKind,
    pub namespace: String,
    pub input_fingerprint: String,
    pub original_result_fingerprint: String,
    pub dispatch_index: u64,
    pub result_status: ObservationResultStatus,
    pub native_wait_poll: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationReason {
    BelowThreshold,
    RepetitionObserved,
    NoticeAlreadyEmitted,
    NativeWaitPollExcluded,
    StatusNotOriginal,
    TurnReset,
}

impl ObservationReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BelowThreshold => "below_threshold",
            Self::RepetitionObserved => "repetition_observed",
            Self::NoticeAlreadyEmitted => "notice_already_emitted",
            Self::NativeWaitPollExcluded => "native_wait_poll_excluded",
            Self::StatusNotOriginal => "status_not_original",
            Self::TurnReset => "turn_reset",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationDecision {
    pub consecutive_count: u8,
    pub emit_diagnostic: bool,
    pub emit_model_notice: bool,
    /// True when a repetition notice was decided for a later-dispatch
    /// completion that had already been delivered. Callers must not attach
    /// that notice to the current result.
    pub standalone_notice: bool,
    pub reason: ObservationReason,
    pub input_fingerprint_hash: String,
    pub result_fingerprint_hash: String,
    pub dispatch_index: u64,
}

impl ObservationDecision {
    fn skipped(reason: ObservationReason, input: &ObservationInput) -> Self {
        Self {
            consecutive_count: 0,
            emit_diagnostic: false,
            emit_model_notice: false,
            standalone_notice: false,
            reason,
            input_fingerprint_hash: hash_identifier(&input.input_fingerprint),
            result_fingerprint_hash: hash_identifier(&input.original_result_fingerprint),
            dispatch_index: input.dispatch_index,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObservationRecord {
    tool_name: String,
    tool_kind: ToolKind,
    namespace: String,
    input_fingerprint: String,
    original_result_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ObservationSlot {
    Countable(ObservationRecord),
    /// Native wait/poll: occupies dispatch order so it is not an in-flight
    /// hole, but does not count toward or break a repetition streak.
    Skip,
}

impl ObservationRecord {
    fn from_input(input: &ObservationInput) -> Self {
        Self {
            tool_name: input.tool_name.clone(),
            tool_kind: input.tool_kind,
            namespace: input.namespace.clone(),
            input_fingerprint: input.input_fingerprint.clone(),
            original_result_fingerprint: input.original_result_fingerprint.clone(),
        }
    }

    fn same_operation_and_result(&self, other: &Self) -> bool {
        self.tool_name == other.tool_name
            && self.tool_kind == other.tool_kind
            && self.namespace == other.namespace
            && self.input_fingerprint == other.input_fingerprint
            && self.original_result_fingerprint == other.original_result_fingerprint
    }
}

/// Non-blocking repetition observer. Not a second tool ledger: call-id
/// dedup stays in `ToolCallLedger`. This only watches completed original
/// results in dispatch order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ObservationState {
    completed: BTreeMap<u64, ObservationSlot>,
    notice_emitted_for_key: Option<ObservationRecord>,
    deferred_notices: Vec<ObservationDecision>,
}

impl ObservationState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.completed.clear();
        self.notice_emitted_for_key = None;
        self.deferred_notices.clear();
    }

    pub fn take_deferred_notices(&mut self) -> Vec<ObservationDecision> {
        std::mem::take(&mut self.deferred_notices)
    }

    pub fn observe(
        &mut self,
        input: ObservationInput,
        repetition_notice: bool,
    ) -> ObservationDecision {
        if input.native_wait_poll
            || is_native_wait_poll(&input.tool_name, input.tool_kind, &input.namespace)
        {
            self.completed
                .insert(input.dispatch_index, ObservationSlot::Skip);
            return ObservationDecision::skipped(ObservationReason::NativeWaitPollExcluded, &input);
        }
        if input.result_status == ObservationResultStatus::Cancelled {
            self.reset();
            return ObservationDecision::skipped(ObservationReason::TurnReset, &input);
        }
        if input.result_status == ObservationResultStatus::Synthetic {
            return ObservationDecision::skipped(ObservationReason::StatusNotOriginal, &input);
        }

        self.completed.insert(
            input.dispatch_index,
            ObservationSlot::Countable(ObservationRecord::from_input(&input)),
        );
        let (consecutive, trailing, last_index) = trailing_consecutive(&self.completed);
        let input_hash = hash_identifier(&input.input_fingerprint);
        let result_hash = hash_identifier(&input.original_result_fingerprint);
        let belongs_to_current = last_index == Some(input.dispatch_index);

        if consecutive < REPETITION_THRESHOLD {
            if trailing
                .as_ref()
                .is_some_and(|key| self.notice_emitted_for_key.as_ref() != Some(key))
            {
                self.notice_emitted_for_key = None;
            }
            return ObservationDecision {
                consecutive_count: consecutive,
                emit_diagnostic: false,
                emit_model_notice: false,
                standalone_notice: false,
                reason: ObservationReason::BelowThreshold,
                input_fingerprint_hash: input_hash,
                result_fingerprint_hash: result_hash,
                dispatch_index: input.dispatch_index,
            };
        }

        let already = trailing
            .as_ref()
            .is_some_and(|key| self.notice_emitted_for_key.as_ref() == Some(key));
        if already {
            return ObservationDecision {
                consecutive_count: consecutive,
                emit_diagnostic: false,
                emit_model_notice: false,
                standalone_notice: false,
                reason: ObservationReason::NoticeAlreadyEmitted,
                input_fingerprint_hash: input_hash,
                result_fingerprint_hash: result_hash,
                dispatch_index: input.dispatch_index,
            };
        }

        self.notice_emitted_for_key = trailing;
        if !belongs_to_current {
            let late = ObservationDecision {
                consecutive_count: consecutive,
                emit_diagnostic: true,
                emit_model_notice: false,
                standalone_notice: repetition_notice,
                reason: ObservationReason::RepetitionObserved,
                input_fingerprint_hash: input_hash.clone(),
                result_fingerprint_hash: result_hash.clone(),
                dispatch_index: last_index.unwrap_or(input.dispatch_index),
            };
            self.deferred_notices.push(late);
            return ObservationDecision {
                consecutive_count: consecutive,
                emit_diagnostic: false,
                emit_model_notice: false,
                standalone_notice: false,
                reason: ObservationReason::BelowThreshold,
                input_fingerprint_hash: input_hash,
                result_fingerprint_hash: result_hash,
                dispatch_index: input.dispatch_index,
            };
        }

        ObservationDecision {
            consecutive_count: consecutive,
            emit_diagnostic: true,
            emit_model_notice: repetition_notice,
            standalone_notice: false,
            reason: ObservationReason::RepetitionObserved,
            input_fingerprint_hash: input_hash,
            result_fingerprint_hash: result_hash,
            dispatch_index: input.dispatch_index,
        }
    }
}

pub fn is_native_wait_poll(tool_name: &str, tool_kind: ToolKind, namespace: &str) -> bool {
    if !super::tool_reliability::canonical_namespace(namespace).is_empty() {
        return false;
    }
    NATIVE_WAIT_POLL_TOOLS
        .iter()
        .any(|(name, kind)| *name == tool_name && *kind == tool_kind)
}

/// Hash an original tool result after stripping a previously attached notice
/// so later comparisons do not count the reminder as part of the result.
pub fn fingerprint_original_result(result: &str, attached_notice: Option<&str>) -> String {
    let stripped = match attached_notice {
        Some(notice) if !notice.is_empty() => result
            .strip_suffix(notice)
            .map(str::trim_end)
            .unwrap_or(result),
        _ => result,
    };
    let digest = Sha256::digest(stripped.as_bytes());
    format!("sha256:{}", hex::encode(digest))
}

pub fn observation_events(decision: &ObservationDecision, call_id: &str) -> Vec<EnhancedEvent> {
    if !decision.emit_diagnostic {
        return Vec::new();
    }
    let fields = EnhancedEventFields {
        call_id_hash: Some(hash_identifier(call_id)),
        consecutive_count: Some(decision.consecutive_count),
        input_fingerprint_hash: Some(decision.input_fingerprint_hash.clone()),
        result_fingerprint_hash: Some(decision.result_fingerprint_hash.clone()),
        observation_reason: Some(decision.reason.as_str().to_string()),
        outcome: Some(if decision.emit_model_notice {
            "notice".into()
        } else if decision.standalone_notice {
            "standalone".into()
        } else {
            "diagnostic".into()
        }),
        ..EnhancedEventFields::default()
    };
    let mut events = vec![EnhancedEvent::new(
        EnhancedEventKind::ToolRepetitionObserved,
        fields.clone(),
    )];
    if decision.emit_model_notice || decision.standalone_notice {
        events.push(EnhancedEvent::new(
            EnhancedEventKind::ToolRepetitionNoticeAppended,
            fields,
        ));
    }
    events
}

fn trailing_consecutive(
    completed: &BTreeMap<u64, ObservationSlot>,
) -> (u8, Option<ObservationRecord>, Option<u64>) {
    let Some((last_index, last)) = completed.iter().rev().find_map(|(index, slot)| match slot {
        ObservationSlot::Countable(record) => Some((*index, record)),
        ObservationSlot::Skip => None,
    }) else {
        return (0, None, None);
    };
    let mut count = 1u8;
    let mut index = last_index;
    while index > 0 {
        index -= 1;
        match completed.get(&index) {
            Some(ObservationSlot::Skip) => {}
            Some(ObservationSlot::Countable(record)) if record.same_operation_and_result(last) => {
                count = count.saturating_add(1);
            }
            Some(ObservationSlot::Countable(_)) | None => break,
        }
    }
    (count, Some(last.clone()), Some(last_index))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(
        dispatch_index: u64,
        tool: &str,
        args_fp: &str,
        result_fp: &str,
    ) -> ObservationInput {
        ObservationInput {
            tool_name: tool.into(),
            tool_kind: ToolKind::Function,
            namespace: String::new(),
            input_fingerprint: args_fp.into(),
            original_result_fingerprint: result_fp.into(),
            dispatch_index,
            result_status: ObservationResultStatus::Success,
            native_wait_poll: false,
        }
    }

    #[test]
    fn three_consecutive_same_ops_in_dispatch_order_emit_one_diagnostic() {
        let mut state = ObservationState::new();
        let first = state.observe(input(0, "apply_patch", "in-a", "ok"), false);
        let second = state.observe(input(1, "apply_patch", "in-a", "ok"), false);
        let third = state.observe(input(2, "apply_patch", "in-a", "ok"), false);
        assert!(!first.emit_diagnostic);
        assert!(!second.emit_diagnostic);
        assert!(third.emit_diagnostic);
        assert!(!third.emit_model_notice);
        assert_eq!(third.consecutive_count, 3);
        assert_eq!(third.reason, ObservationReason::RepetitionObserved);
        let fourth = state.observe(input(3, "apply_patch", "in-a", "ok"), false);
        assert!(!fourth.emit_diagnostic);
        assert_eq!(fourth.reason, ObservationReason::NoticeAlreadyEmitted);
    }

    #[test]
    fn model_visible_notice_only_when_flag_on() {
        let mut state = ObservationState::new();
        state.observe(input(0, "apply_patch", "in-a", "ok"), true);
        state.observe(input(1, "apply_patch", "in-a", "ok"), true);
        let third = state.observe(input(2, "apply_patch", "in-a", "ok"), true);
        assert!(third.emit_diagnostic);
        assert!(third.emit_model_notice);
    }

    #[test]
    fn fewer_than_three_or_changed_result_does_not_trigger() {
        let mut state = ObservationState::new();
        state.observe(input(0, "apply_patch", "in-a", "ok"), true);
        state.observe(input(1, "apply_patch", "in-a", "ok"), true);
        let changed = state.observe(input(2, "apply_patch", "in-a", "other"), true);
        assert!(!changed.emit_diagnostic);
        state.observe(input(3, "apply_patch", "in-a", "other"), true);
        let still = state.observe(input(4, "apply_patch", "in-b", "other"), true);
        assert!(!still.emit_diagnostic);
    }

    #[test]
    fn parallel_out_of_order_completion_uses_dispatch_order() {
        let mut state = ObservationState::new();
        // Completion order 0,2,3,1 would look like three A's then B.
        // Dispatch order is A, B, A, A — only two trailing A's.
        state.observe(input(0, "read", "file-a", "x"), true);
        state.observe(input(2, "read", "file-a", "x"), true);
        let late_same = state.observe(input(3, "read", "file-a", "x"), true);
        assert!(
            !late_same.emit_diagnostic,
            "must not fire until dispatch holes are filled"
        );
        let hole = state.observe(input(1, "read", "file-b", "x"), true);
        assert!(!hole.emit_diagnostic);
        assert!(!hole.emit_model_notice);
        assert_eq!(hole.consecutive_count, 2);
    }

    #[test]
    fn out_of_order_identical_ops_do_not_attach_notice_to_a_late_middle_result() {
        let mut state = ObservationState::new();
        let first = state.observe(input(0, "apply_patch", "in-a", "ok"), true);
        let late_high = state.observe(input(2, "apply_patch", "in-a", "ok"), true);
        let middle = state.observe(input(1, "apply_patch", "in-a", "ok"), true);
        assert!(!first.emit_model_notice);
        assert!(
            !late_high.emit_model_notice,
            "index 2 was delivered before the streak was known"
        );
        assert!(
            !middle.emit_model_notice,
            "must not attach the notice to the late-completing middle result"
        );
        assert!(!middle.emit_diagnostic);
        let deferred = state.take_deferred_notices();
        assert_eq!(deferred.len(), 1);
        assert_eq!(deferred[0].dispatch_index, 2);
        assert_eq!(deferred[0].reason, ObservationReason::RepetitionObserved);
        assert!(deferred[0].emit_diagnostic);
        assert!(!deferred[0].emit_model_notice);
        assert!(deferred[0].standalone_notice);
    }

    #[test]
    fn native_wait_between_identical_ops_does_not_break_the_streak() {
        let mut state = ObservationState::new();
        let wait = |index: u64| ObservationInput {
            tool_name: "wait".into(),
            tool_kind: ToolKind::Function,
            namespace: String::new(),
            input_fingerprint: "poll".into(),
            original_result_fingerprint: "pending".into(),
            dispatch_index: index,
            result_status: ObservationResultStatus::Success,
            native_wait_poll: false,
        };
        state.observe(input(0, "apply_patch", "in-a", "ok"), true);
        state.observe(wait(1), true);
        state.observe(input(2, "apply_patch", "in-a", "ok"), true);
        state.observe(wait(3), true);
        let third = state.observe(input(4, "apply_patch", "in-a", "ok"), true);
        assert!(third.emit_diagnostic);
        assert_eq!(third.consecutive_count, 3);
    }

    #[test]
    fn native_wait_poll_is_excluded_and_name_suffix_is_not_guessed() {
        let mut state = ObservationState::new();
        for index in 0..3 {
            let decision = state.observe(
                ObservationInput {
                    tool_name: "wait".into(),
                    tool_kind: ToolKind::Function,
                    namespace: String::new(),
                    input_fingerprint: "poll".into(),
                    original_result_fingerprint: "pending".into(),
                    dispatch_index: index,
                    result_status: ObservationResultStatus::Success,
                    native_wait_poll: false,
                },
                true,
            );
            assert_eq!(decision.reason, ObservationReason::NativeWaitPollExcluded);
            assert!(!decision.emit_diagnostic);
        }
        let mut suffix = ObservationState::new();
        for index in 0..3 {
            let decision = suffix.observe(
                ObservationInput {
                    tool_name: "server_wait".into(),
                    tool_kind: ToolKind::Function,
                    namespace: String::new(),
                    input_fingerprint: "poll".into(),
                    original_result_fingerprint: "pending".into(),
                    dispatch_index: index,
                    result_status: ObservationResultStatus::Success,
                    native_wait_poll: false,
                },
                true,
            );
            if index == 2 {
                assert!(
                    decision.emit_diagnostic,
                    "unknown *_wait tools are not excluded by suffix"
                );
            }
        }
        let mut namespaced = ObservationState::new();
        for index in 0..3 {
            let decision = namespaced.observe(
                ObservationInput {
                    tool_name: "wait".into(),
                    tool_kind: ToolKind::Function,
                    namespace: "mcp".into(),
                    input_fingerprint: "poll".into(),
                    original_result_fingerprint: "pending".into(),
                    dispatch_index: index,
                    result_status: ObservationResultStatus::Success,
                    native_wait_poll: false,
                },
                true,
            );
            if index == 2 {
                assert!(decision.emit_diagnostic);
            }
        }
    }

    #[test]
    fn user_turn_cancel_idle_and_ended_resume_reset() {
        let mut state = ObservationState::new();
        state.observe(input(0, "shell", "a", "ok"), true);
        state.observe(input(1, "shell", "a", "ok"), true);
        state.reset();
        let after = state.observe(input(2, "shell", "a", "ok"), true);
        assert!(!after.emit_diagnostic);
        assert_eq!(after.consecutive_count, 1);

        state.observe(input(3, "shell", "a", "ok"), true);
        let cancel = state.observe(
            ObservationInput {
                result_status: ObservationResultStatus::Cancelled,
                ..input(4, "shell", "a", "ok")
            },
            true,
        );
        assert_eq!(cancel.reason, ObservationReason::TurnReset);
        let resumed = ObservationState::new();
        assert!(resumed.completed.is_empty());
    }

    #[test]
    fn attached_notice_is_stripped_before_result_fingerprint() {
        let original = "patched ok";
        let with_notice = format!("{original}\n{REPETITION_NOTICE}");
        assert_eq!(
            fingerprint_original_result(&with_notice, Some(REPETITION_NOTICE)),
            fingerprint_original_result(original, None)
        );
        assert_ne!(
            fingerprint_original_result(&with_notice, None),
            fingerprint_original_result(original, None)
        );
    }
}
