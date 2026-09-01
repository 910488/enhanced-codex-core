//! Agent-loop adapters for the four Enhanced seams.

use std::sync::Arc;

use codex_protocol::items::HookPromptFragment;
use codex_protocol::items::build_hook_prompt_message;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::ResponseItem;
use codex_protocol::plan_tool::StepStatus;
use codex_protocol::plan_tool::UpdatePlanArgs;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::function_tool::FunctionCallError;
use crate::session::session::Session;
use crate::session::turn_context::TurnContext;
use crate::tools::context::ToolPayload;
use crate::tools::router::ToolCall;

use super::bounded_continuation::{
    ContinuationDecision, ReservedContinuation, TurnStopContext, UnfinishedSignal, CONTINUE_NUDGE,
};
use super::context_pruner::{ContentBlock, ModelVisibleSurface, SurfaceItem};
use super::context_recovery::{CompactDecision, OverflowDecision};
use super::hooks::HookDecision;
use super::runtime::EnhancedSessionRuntime;
use super::telemetry::MemoryTelemetry;
use super::tool_reliability::{
    AdmitDecision, ProviderToolCallIdentity, ToolCallLedger, ToolCallResolution,
};

pub(crate) enum PressureSeam {
    DeferToUpstream,
    SkipNativeCompact,
    RunNativeCompact,
}

pub(crate) enum OverflowSeam {
    DeferToUpstream,
    Retry { pruned: Vec<ResponseItem> },
    NeedsNativeCompact { before: ModelVisibleSurface },
    PreserveOriginalError,
    Cancelled,
}

fn split_runtime(
    runtime: &mut EnhancedSessionRuntime,
) -> (
    &mut super::hooks::EnhancedTurnHooks,
    &mut MemoryTelemetry,
) {
    (&mut runtime.hooks, &mut runtime.telemetry)
}

fn log_new_events(telemetry: &MemoryTelemetry, start: usize) {
    for event in telemetry.events.iter().skip(start) {
        tracing::info!(
            target: "codex_enhanced",
            event = event.name.as_str(),
            "enhanced runtime event"
        );
    }
}

fn payload_arguments(payload: &ToolPayload) -> String {
    match payload {
        ToolPayload::Function { arguments } => arguments.clone(),
        ToolPayload::Custom { input } => input.clone(),
        ToolPayload::ToolSearch { arguments } => arguments.query.clone(),
    }
}

fn identity_from_call(call: &ToolCall) -> ProviderToolCallIdentity {
    let arguments = payload_arguments(&call.payload);
    let value = serde_json::from_str::<Value>(&arguments)
        .unwrap_or_else(|_| Value::String(arguments));
    ToolCallLedger::identity(call.call_id.clone(), &call.tool_name.name, &value)
}

pub(crate) fn admit_tool_call(
    runtime: &mut EnhancedSessionRuntime,
    call: &ToolCall,
) -> Result<(), FunctionCallError> {
    let identity = identity_from_call(call);
    let (hooks, telemetry) = split_runtime(runtime);
    let start = telemetry.events.len();
    let decision = hooks.admit_tool_call(identity, telemetry);
    log_new_events(telemetry, start);
    match decision {
        HookDecision::DeferToUpstream | HookDecision::Handled(AdmitDecision::Execute) => Ok(()),
        HookDecision::Handled(AdmitDecision::SuppressDuplicate { message }) => {
            runtime
                .hooks
                .mark_tool_resolution(&call.call_id, ToolCallResolution::SyntheticDuplicate);
            Err(FunctionCallError::RespondToModel(message))
        }
        HookDecision::Handled(AdmitDecision::FailClosed { message }) => {
            Err(FunctionCallError::Fatal(message))
        }
    }
}

pub(crate) fn complete_tool_call(
    runtime: &mut EnhancedSessionRuntime,
    call_id: &str,
    succeeded: bool,
) {
    match runtime.hooks.complete_original_tool_result(call_id) {
        HookDecision::Handled(super::tool_reliability::LateResultDecision::Suppress) => {
            let _ = runtime.hooks.ingest_late_tool_result(call_id);
        }
        HookDecision::Handled(super::tool_reliability::LateResultDecision::Accept) => {
            runtime.hooks.mark_tool_resolution(
                call_id,
                if succeeded {
                    ToolCallResolution::Executed
                } else {
                    ToolCallResolution::Failed
                },
            );
        }
        HookDecision::DeferToUpstream => {}
    }
}

pub(crate) fn on_new_user_input(runtime: &mut EnhancedSessionRuntime) {
    runtime.hooks.on_new_user_input();
}

pub(crate) fn on_assistant_success(runtime: &mut EnhancedSessionRuntime) {
    runtime.hooks.on_assistant_success();
}

pub(crate) fn on_turn_idle(runtime: &mut EnhancedSessionRuntime) {
    runtime.hooks.on_turn_idle();
}

fn content_text(content: &[ContentItem]) -> String {
    content
        .iter()
        .filter_map(|item| match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                Some(text.as_str())
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn response_items_to_surface(items: &[ResponseItem]) -> ModelVisibleSurface {
    let mut surface = ModelVisibleSurface::default();
    for item in items {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let text = content_text(content);
                surface.items.push(match role.as_str() {
                    "assistant" => SurfaceItem::Assistant { text },
                    "system" => SurfaceItem::System { text },
                    "developer" => SurfaceItem::Developer { text },
                    _ => SurfaceItem::User { text },
                });
            }
            ResponseItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => surface.items.push(SurfaceItem::ToolCall {
                call_id: call_id.clone(),
                tool_name: name.clone(),
                tool_type: "function".into(),
                arguments: arguments.clone(),
            }),
            ResponseItem::FunctionCallOutput {
                call_id,
                name,
                output,
                ..
            } => surface.items.push(SurfaceItem::ToolResult {
                call_id: call_id.clone().unwrap_or_default(),
                tool_name: name.clone().unwrap_or_default(),
                tool_type: "function".into(),
                blocks: vec![ContentBlock {
                    kind: "text".into(),
                    text: output.body.to_text(),
                }],
            }),
            ResponseItem::CustomToolCall {
                call_id,
                name,
                input,
                ..
            } => surface.items.push(SurfaceItem::ToolCall {
                call_id: call_id.clone(),
                tool_name: name.clone(),
                tool_type: "custom".into(),
                arguments: input.clone(),
            }),
            ResponseItem::CustomToolCallOutput {
                call_id,
                name,
                output,
                ..
            } => surface.items.push(SurfaceItem::ToolResult {
                call_id: call_id.clone(),
                tool_name: name.clone().unwrap_or_default(),
                tool_type: "custom".into(),
                blocks: vec![ContentBlock {
                    kind: "text".into(),
                    text: output.body.to_text(),
                }],
            }),
            _ => {}
        }
    }
    surface
}

fn apply_surface_to_items(items: &mut [ResponseItem], surface: &ModelVisibleSurface) {
    let mut texts = std::collections::HashMap::new();
    for item in &surface.items {
        if let SurfaceItem::ToolResult { call_id, blocks, .. } = item {
            let text = blocks
                .iter()
                .filter_map(|block| block.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n");
            texts.insert(call_id.clone(), text);
        }
    }
    for item in items {
        match item {
            ResponseItem::FunctionCallOutput {
                call_id, output, ..
            } => {
                if let Some(id) = call_id {
                    if let Some(text) = texts.get(id) {
                        output.body = FunctionCallOutputBody::Text(text.clone());
                    }
                }
            }
            ResponseItem::CustomToolCallOutput {
                call_id, output, ..
            } => {
                if let Some(text) = texts.get(call_id) {
                    output.body = FunctionCallOutputBody::Text(text.clone());
                }
            }
            _ => {}
        }
    }
}

pub(crate) fn apply_pressure_to_prompt(
    runtime: &mut EnhancedSessionRuntime,
    items: &mut Vec<ResponseItem>,
    context_window_tokens: u64,
    compact_threshold_tokens: u64,
) -> PressureSeam {
    if context_window_tokens == 0 {
        return PressureSeam::DeferToUpstream;
    }
    let surface = response_items_to_surface(items);
    let decision = {
        let (hooks, telemetry) = split_runtime(runtime);
        let start = telemetry.events.len();
        let decision = hooks.plan_pressure(
            &surface,
            context_window_tokens,
            compact_threshold_tokens,
            telemetry,
        );
        log_new_events(telemetry, start);
        decision
    };
    match decision {
        HookDecision::DeferToUpstream => PressureSeam::DeferToUpstream,
        HookDecision::Handled(plan) => {
            if plan.prune.rewritten {
                apply_surface_to_items(items, &plan.prune.surface);
            }
            runtime.last_surface = Some(plan.prune.surface);
            match plan.compact {
                CompactDecision::Skip => PressureSeam::SkipNativeCompact,
                CompactDecision::RunNativeCodexLocalCompact => PressureSeam::RunNativeCompact,
            }
        }
    }
}

pub(crate) fn plan_overflow_for_prompt(
    runtime: &mut EnhancedSessionRuntime,
    sent_items: &[ResponseItem],
    context_window_tokens: u64,
    cancelled: bool,
) -> OverflowSeam {
    let before = response_items_to_surface(sent_items);
    let mut candidate = sent_items.to_vec();
    match apply_pressure_to_prompt(
        runtime,
        &mut candidate,
        context_window_tokens,
        context_window_tokens,
    ) {
        PressureSeam::DeferToUpstream => OverflowSeam::DeferToUpstream,
        PressureSeam::SkipNativeCompact => {
            decide_overflow_retry(runtime, &before, &candidate, cancelled)
        }
        PressureSeam::RunNativeCompact => OverflowSeam::NeedsNativeCompact { before },
    }
}

pub(crate) fn decide_overflow_retry(
    runtime: &mut EnhancedSessionRuntime,
    before: &ModelVisibleSurface,
    after_items: &[ResponseItem],
    cancelled: bool,
) -> OverflowSeam {
    let mut after = response_items_to_surface(after_items);
    if after.char_count() < before.char_count() && after.generation <= before.generation {
        after.generation = before.generation.saturating_add(1);
    }
    let decision = {
        let (hooks, telemetry) = split_runtime(runtime);
        let start = telemetry.events.len();
        let decision = hooks.plan_overflow(before, &after, cancelled, telemetry);
        log_new_events(telemetry, start);
        decision
    };
    match decision {
        HookDecision::DeferToUpstream => OverflowSeam::DeferToUpstream,
        HookDecision::Handled(plan) => match plan.decision {
            OverflowDecision::Retry { .. } => OverflowSeam::Retry {
                pruned: after_items.to_vec(),
            },
            OverflowDecision::PreserveOriginalError => OverflowSeam::PreserveOriginalError,
            OverflowDecision::Cancelled => OverflowSeam::Cancelled,
        },
    }
}

fn unfinished_from_items(items: &[ResponseItem]) -> Vec<UnfinishedSignal> {
    let mut unfinished = Vec::new();
    for item in items.iter().rev() {
        let ResponseItem::FunctionCall {
            name, arguments, ..
        } = item
        else {
            continue;
        };
        if name != "update_plan" {
            continue;
        }
        if let Ok(args) = serde_json::from_str::<UpdatePlanArgs>(arguments) {
            if args.plan.iter().any(|step| {
                matches!(step.status, StepStatus::Pending | StepStatus::InProgress)
            }) {
                unfinished.push(UnfinishedSignal::NativePlanIncomplete);
            }
        }
        break;
    }
    unfinished
}

pub(crate) async fn maybe_continue_turn(
    sess: &Arc<Session>,
    turn_context: &TurnContext,
    items: &[ResponseItem],
    cancellation_token: &CancellationToken,
    user_steer_pending: bool,
) -> bool {
    let context = TurnStopContext {
        cancelled: cancellation_token.is_cancelled(),
        user_steer_pending,
        unfinished: unfinished_from_items(items),
    };
    let (plan, reserved) = {
        let mut runtime = sess.enhanced.lock().unwrap_or_else(|error| error.into_inner());
        let (hooks, telemetry) = split_runtime(&mut runtime);
        let start = telemetry.events.len();
        let decision = hooks.on_turn_stop(&context, telemetry);
        log_new_events(telemetry, start);
        match decision {
            HookDecision::DeferToUpstream => return false,
            HookDecision::Handled(plan) => {
                let reserved = plan.reservation;
                (plan, reserved)
            }
        }
    };
    match plan.decision {
        ContinuationDecision::Continue { .. } => {
            let Some(reserved) = reserved else {
                return false;
            };
            let Some(item) = build_hook_prompt_message(&[HookPromptFragment::from_single_hook(
                CONTINUE_NUDGE,
                "enhanced-continuation",
            )]) else {
                sess.enhanced
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .hooks
                    .release_continuation();
                return false;
            };
            sess.record_response_item_and_emit_turn_item(turn_context, item)
                .await;
            sess.enhanced
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .pending_continuation = Some(reserved);
            true
        }
        _ => false,
    }
}

pub(crate) fn commit_pending_continuation(sess: &Session) {
    let mut runtime = sess
        .enhanced
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let Some(reserved) = runtime.pending_continuation.take() else {
        return;
    };
    if runtime.hooks.commit_continuation(reserved).is_err() {
        runtime.hooks.release_continuation();
    }
}

pub(crate) fn release_pending_continuation(sess: &Session) {
    let mut runtime = sess
        .enhanced
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    runtime.pending_continuation = None;
    runtime.hooks.release_continuation();
}

pub(crate) async fn restore_ledger_if_needed(sess: &Session) {
    {
        let runtime = sess
            .enhanced
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if runtime.ledger_restored || !runtime.hooks.features.qwen_tool_reliability {
            return;
        }
    }
    let history = sess.clone_history().await;
    let items = history.raw_items().cloned().collect::<Vec<_>>();
    let mut runtime = sess
        .enhanced
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if runtime.ledger_restored {
        return;
    }
    runtime.hooks.ledger = ledger_from_history(&items);
    runtime.ledger_restored = true;
}

fn ledger_from_history(items: &[ResponseItem]) -> ToolCallLedger {
    let mut ledger = ToolCallLedger::new();
    for item in items {
        match item {
            ResponseItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                let value = serde_json::from_str::<Value>(arguments)
                    .unwrap_or_else(|_| Value::String(arguments.clone()));
                let identity = ToolCallLedger::identity(call_id.clone(), name, &value);
                let _ = ledger.admit(identity);
            }
            ResponseItem::CustomToolCall {
                call_id,
                name,
                input,
                ..
            } => {
                let value = serde_json::from_str::<Value>(input)
                    .unwrap_or_else(|_| Value::String(input.clone()));
                let identity = ToolCallLedger::identity(call_id.clone(), name, &value);
                let _ = ledger.admit(identity);
            }
            ResponseItem::FunctionCallOutput {
                call_id, output, ..
            } => {
                let Some(call_id) = call_id else {
                    continue;
                };
                let _ = ledger.complete_original(call_id);
                ledger.mark_resolution(
                    call_id,
                    if output.success == Some(false) {
                        ToolCallResolution::Failed
                    } else {
                        ToolCallResolution::Executed
                    },
                );
            }
            ResponseItem::CustomToolCallOutput {
                call_id, output, ..
            } => {
                let _ = ledger.complete_original(call_id);
                ledger.mark_resolution(
                    call_id,
                    if output.success == Some(false) {
                        ToolCallResolution::Failed
                    } else {
                        ToolCallResolution::Executed
                    },
                );
            }
            _ => {}
        }
    }
    ledger
}
