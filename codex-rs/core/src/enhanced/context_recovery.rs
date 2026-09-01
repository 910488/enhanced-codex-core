use super::context_pruner::{
    apply_pressure_prune, ModelVisibleSurface, PruneOutcome, ToolResultPrunePolicy,
};
use super::telemetry::{EnhancedEvent, EnhancedEventFields, EnhancedEventKind};

pub const MAX_CONTEXT_OVERFLOW_RETRIES: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactDecision {
    Skip,
    RunNativeCodexLocalCompact,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PressurePlan {
    pub prune: PruneOutcome,
    pub compact: CompactDecision,
    pub compaction_avoided: bool,
    pub events: Vec<EnhancedEvent>,
}

/// Context pressure path: prune first, re-measure, then only call Codex local
/// compact if the pruned surface is still above the compact threshold.
pub fn plan_context_pressure(
    surface: &ModelVisibleSurface,
    policy: ToolResultPrunePolicy,
    context_window_tokens: u64,
    compact_threshold_tokens: u64,
) -> PressurePlan {
    let prune = apply_pressure_prune(surface, policy, context_window_tokens);
    let mut events = prune.events.clone();
    let still_high = prune.surface.estimated_tokens() >= compact_threshold_tokens;
    let compaction_avoided = prune.rewritten && !still_high;
    if compaction_avoided {
        events.push(EnhancedEvent::new(
            EnhancedEventKind::ContextCompactionAvoided,
            EnhancedEventFields {
                before_token_estimate: Some(prune.before_token_estimate),
                after_token_estimate: Some(prune.after_token_estimate),
                chars_removed: Some(prune.chars_removed),
                ..EnhancedEventFields::default()
            },
        ));
    }
    PressurePlan {
        compact: if still_high {
            CompactDecision::RunNativeCodexLocalCompact
        } else {
            CompactDecision::Skip
        },
        compaction_avoided,
        events,
        prune,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverflowAttempt {
    pub generation_before: u64,
    pub retries_used: u8,
    pub cancelled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowDecision {
    Retry { retry_index: u8 },
    PreserveOriginalError,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OverflowPlan {
    pub decision: OverflowDecision,
    pub events: Vec<EnhancedEvent>,
}

/// Provider-confirmed context overflow recovery. Retry at most once, and only
/// when prune/compact actually advanced the surface generation and reduced
/// the model-visible size. Cancellation always wins.
pub fn plan_overflow_retry(
    attempt: OverflowAttempt,
    before: &ModelVisibleSurface,
    after: &ModelVisibleSurface,
) -> OverflowPlan {
    if attempt.cancelled {
        return OverflowPlan {
            decision: OverflowDecision::Cancelled,
            events: Vec::new(),
        };
    }
    if attempt.retries_used >= MAX_CONTEXT_OVERFLOW_RETRIES {
        return refused(attempt.retries_used);
    }
    let shrunk = after.char_count() < before.char_count();
    let progressed = after.generation > attempt.generation_before;
    if shrunk && progressed {
        let retry_index = attempt.retries_used.saturating_add(1);
        return OverflowPlan {
            decision: OverflowDecision::Retry { retry_index },
            events: vec![EnhancedEvent::new(
                EnhancedEventKind::ContextOverflowRetry,
                EnhancedEventFields {
                    retry_index: Some(retry_index),
                    before_token_estimate: Some(before.estimated_tokens()),
                    after_token_estimate: Some(after.estimated_tokens()),
                    chars_removed: Some(before.char_count().saturating_sub(after.char_count()) as u64),
                    ..EnhancedEventFields::default()
                },
            )],
        };
    }
    refused(attempt.retries_used)
}

fn refused(retry_index: u8) -> OverflowPlan {
    OverflowPlan {
        decision: OverflowDecision::PreserveOriginalError,
        events: vec![EnhancedEvent::new(
            EnhancedEventKind::ContextOverflowRetryRefused,
            EnhancedEventFields {
                retry_index: Some(retry_index),
                ..EnhancedEventFields::default()
            },
        )],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::context_pruner::{ContentBlock, SurfaceItem};

    fn surface(chars: usize, generation: u64) -> ModelVisibleSurface {
        ModelVisibleSurface {
            generation,
            items: vec![
                SurfaceItem::ToolCall {
                    call_id: "c1".into(),
                    tool_name: "read".into(),
                    tool_type: "function".into(),
                    arguments: "{}".into(),
                },
                SurfaceItem::ToolResult {
                    call_id: "c1".into(),
                    tool_name: "read".into(),
                    tool_type: "function".into(),
                    blocks: vec![ContentBlock {
                        kind: "text".into(),
                        text: Some("x".repeat(chars)),
                    }],
                },
            ],
        }
    }

    #[test]
    fn pressure_prune_can_avoid_compaction() {
        let policy = ToolResultPrunePolicy {
            trigger_ratio: 0.01,
            min_text_chars: 20,
            keep_head_chars: 8,
            keep_tail_chars: 8,
        };
        let plan = plan_context_pressure(&surface(8_000, 1), policy, 100, 2_000);
        assert!(plan.prune.rewritten);
        assert_eq!(plan.compact, CompactDecision::Skip);
        assert!(plan.compaction_avoided);
    }

    #[test]
    fn pressure_prune_then_native_compact() {
        let policy = ToolResultPrunePolicy {
            trigger_ratio: 0.01,
            min_text_chars: 20,
            keep_head_chars: 3_000,
            keep_tail_chars: 3_000,
        };
        let plan = plan_context_pressure(&surface(8_000, 1), policy, 100, 10);
        assert_eq!(plan.compact, CompactDecision::RunNativeCodexLocalCompact);
        assert!(!plan.compaction_avoided);
    }

    #[test]
    fn overflow_retry_requires_surface_progress() {
        let before = surface(1_000, 4);
        let mut after = before.clone();
        after.generation = 4;
        after.items[1] = SurfaceItem::ToolResult {
            call_id: "c1".into(),
            tool_name: "read".into(),
            tool_type: "function".into(),
            blocks: vec![ContentBlock {
                kind: "text".into(),
                text: Some("y".repeat(10)),
            }],
        };
        let plan = plan_overflow_retry(
            OverflowAttempt {
                generation_before: 4,
                retries_used: 0,
                cancelled: false,
            },
            &before,
            &after,
        );
        assert_eq!(plan.decision, OverflowDecision::PreserveOriginalError);
    }

    #[test]
    fn overflow_without_progress_preserves_original_error() {
        let surface = surface(100, 2);
        let plan = plan_overflow_retry(
            OverflowAttempt {
                generation_before: 2,
                retries_used: 0,
                cancelled: false,
            },
            &surface,
            &surface,
        );
        assert_eq!(plan.decision, OverflowDecision::PreserveOriginalError);
    }

    #[test]
    fn overflow_retry_is_bounded_to_one() {
        let before = surface(1_000, 1);
        let mut after = before.clone();
        after.generation = 2;
        after.items[1] = SurfaceItem::ToolResult {
            call_id: "c1".into(),
            tool_name: "read".into(),
            tool_type: "function".into(),
            blocks: vec![ContentBlock {
                kind: "text".into(),
                text: Some("z".repeat(20)),
            }],
        };
        let first = plan_overflow_retry(
            OverflowAttempt {
                generation_before: 1,
                retries_used: 0,
                cancelled: false,
            },
            &before,
            &after,
        );
        assert_eq!(first.decision, OverflowDecision::Retry { retry_index: 1 });
        let second = plan_overflow_retry(
            OverflowAttempt {
                generation_before: 2,
                retries_used: 1,
                cancelled: false,
            },
            &after,
            &after,
        );
        assert_eq!(second.decision, OverflowDecision::PreserveOriginalError);
        let cancelled = plan_overflow_retry(
            OverflowAttempt {
                generation_before: 1,
                retries_used: 0,
                cancelled: true,
            },
            &before,
            &after,
        );
        assert_eq!(cancelled.decision, OverflowDecision::Cancelled);
    }
}
