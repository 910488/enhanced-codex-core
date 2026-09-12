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

use crate::context_manager::estimate_item_token_count;
use crate::function_tool::FunctionCallError;
use crate::session::session::Session;
use crate::session::turn_context::TurnContext;
use crate::tools::context::ToolPayload;
use crate::tools::router::ToolCall;

use super::bounded_continuation::CONTINUE_NUDGE;
use super::bounded_continuation::ContinuationDecision;
use super::bounded_continuation::ReservedContinuation;
use super::bounded_continuation::TurnStopContext;
use super::bounded_continuation::UnfinishedSignal;
use super::context_pruner::ContentBlock;
use super::context_pruner::ModelVisibleSurface;
use super::context_pruner::SurfaceItem;
use super::context_recovery::CompactDecision;
use super::context_recovery::OverflowDecision;
use super::hooks::HookDecision;
use super::runtime::EnhancedSessionRuntime;
use super::telemetry::EnhancedEvent;
use super::telemetry::EnhancedEventFields;
use super::telemetry::EnhancedEventKind;
use super::telemetry::MemoryTelemetry;
use super::telemetry::hash_identifier;
use super::tool_observation::fingerprint_original_result;
use super::tool_observation::is_native_wait_poll;
use super::tool_observation::ObservationInput;
use super::tool_observation::ObservationResultStatus;
use super::tool_observation::REPETITION_NOTICE;
use super::tool_reliability::AdmitDecision;
use super::tool_reliability::ProviderToolCallIdentity;
use super::tool_reliability::ToolCallLedger;
use super::tool_reliability::ToolCallResolution;
use super::tool_reliability::ToolKind;

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
) -> (&mut super::hooks::EnhancedTurnHooks, &mut MemoryTelemetry) {
    (&mut runtime.hooks, &mut runtime.telemetry)
}

fn log_new_events(telemetry: &MemoryTelemetry, start: usize) {
    for event in telemetry.events.iter().skip(start) {
        super::reporting::publish_event(event);
        tracing::info!(
            target: "codex_enhanced",
            event = event.name.as_str(),
            "enhanced runtime event"
        );
    }
}

fn call_namespace(call: &ToolCall) -> String {
    super::tool_reliability::canonical_namespace(
        call.tool_name.namespace.as_deref().unwrap_or(""),
    )
    .to_string()
}

fn identity_from_call(call: &ToolCall) -> ProviderToolCallIdentity {
    let namespace = call_namespace(call);
    match &call.payload {
        ToolPayload::Function { arguments } => {
            let value = serde_json::from_str::<Value>(arguments)
                .unwrap_or_else(|_| Value::String(arguments.clone()));
            ToolCallLedger::identity_json(
                call.call_id.clone(),
                &call.tool_name.name,
                ToolKind::Function,
                &namespace,
                &value,
            )
        }
        ToolPayload::Custom { input } => ToolCallLedger::identity_raw(
            call.call_id.clone(),
            &call.tool_name.name,
            ToolKind::Custom,
            &namespace,
            input,
        ),
        ToolPayload::ToolSearch { arguments } => {
            let value = serde_json::json!({ "query": arguments.query });
            ToolCallLedger::identity_json(
                call.call_id.clone(),
                &call.tool_name.name,
                ToolKind::ToolSearch,
                &namespace,
                &value,
            )
        }
    }
}

/// Sentinel `RespondToModel` payload. `ToolCallRuntime` must drop it instead
/// of turning it into a model-visible `FunctionCallOutput`.
pub(crate) const DROP_TOOL_OUTPUT_MARKER: &str = "vellum.enhanced.drop-tool-output";

pub(crate) fn drop_tool_output_error() -> FunctionCallError {
    FunctionCallError::RespondToModel(DROP_TOOL_OUTPUT_MARKER.to_string())
}

pub(crate) fn is_drop_tool_output(error: &FunctionCallError) -> bool {
    matches!(
        error,
        FunctionCallError::RespondToModel(message) if message == DROP_TOOL_OUTPUT_MARKER
    )
}

fn original_result_already_delivered(runtime: &EnhancedSessionRuntime, call_id: &str) -> bool {
    runtime
        .hooks
        .ledger
        .get(call_id)
        .is_some_and(|handled| handled.original_result_delivered)
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
            if original_result_already_delivered(runtime, &call.call_id) {
                Err(drop_tool_output_error())
            } else {
                Err(FunctionCallError::RespondToModel(message))
            }
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
) -> Result<(), FunctionCallError> {
    complete_tool_call_with_result(runtime, call_id, succeeded, None).map(|_| ())
}

pub(crate) fn complete_tool_call_with_result(
    runtime: &mut EnhancedSessionRuntime,
    call_id: &str,
    succeeded: bool,
    result_text: Option<&str>,
) -> Result<Option<String>, FunctionCallError> {
    let handled = runtime.hooks.ledger.get(call_id).cloned();
    let synthetic_duplicate = handled
        .as_ref()
        .is_some_and(|handled| handled.synthetic_duplicate_emitted);
    let late = runtime.hooks.ingest_late_tool_result(call_id);
    let original = runtime.hooks.complete_original_tool_result(call_id);
    if synthetic_duplicate
        || matches!(
            late,
            HookDecision::Handled(super::tool_reliability::LateResultDecision::Suppress)
        )
        || matches!(
            original,
            HookDecision::Handled(super::tool_reliability::LateResultDecision::Suppress)
        )
    {
        return Err(drop_tool_output_error());
    }
    if let HookDecision::Handled(super::tool_reliability::LateResultDecision::Accept) = original {
        runtime.hooks.mark_tool_resolution(
            call_id,
            if succeeded {
                ToolCallResolution::Executed
            } else {
                ToolCallResolution::Failed
            },
        );
        let start = runtime.telemetry.events.len();
        runtime.telemetry.emit(EnhancedEvent::new(
            EnhancedEventKind::ToolCallCompleted,
            EnhancedEventFields {
                call_id_hash: Some(hash_identifier(call_id)),
                outcome: Some(if succeeded { "executed" } else { "failed" }.into()),
                succeeded: Some(succeeded),
                ..EnhancedEventFields::default()
            },
        ));
        let notice = observe_completed_result(runtime, call_id, handled.as_ref(), succeeded, result_text);
        log_new_events(&runtime.telemetry, start);
        return Ok(notice);
    }
    Ok(None)
}

fn observe_completed_result(
    runtime: &mut EnhancedSessionRuntime,
    call_id: &str,
    handled: Option<&super::tool_reliability::HandledToolCall>,
    succeeded: bool,
    result_text: Option<&str>,
) -> Option<String> {
    let Some(handled) = handled else {
        return None;
    };
    let original = result_text.unwrap_or(if succeeded { "success" } else { "failed" });
    let result_fingerprint = fingerprint_original_result(original, Some(REPETITION_NOTICE));
    let input = ObservationInput {
        tool_name: handled.identity.tool_name.clone(),
        tool_kind: handled.identity.tool_kind,
        namespace: handled.identity.namespace.clone(),
        input_fingerprint: handled.identity.argument_fingerprint.clone(),
        original_result_fingerprint: result_fingerprint,
        dispatch_index: handled.dispatch_index,
        result_status: if succeeded {
            ObservationResultStatus::Success
        } else {
            ObservationResultStatus::Failed
        },
        native_wait_poll: is_native_wait_poll(
            &handled.identity.tool_name,
            handled.identity.tool_kind,
            &handled.identity.namespace,
        ),
    };
    let decision = {
        let (hooks, telemetry) = split_runtime(runtime);
        hooks.observe_tool_result(call_id, input, telemetry)
    };
    if decision.emit_model_notice {
        runtime
            .pending_repetition_notices
            .insert(call_id.to_string());
        Some(REPETITION_NOTICE.to_string())
    } else if decision.standalone_notice {
        runtime.pending_standalone_repetition_notice = true;
        None
    } else {
        None
    }
}

pub(crate) fn on_new_user_input(runtime: &mut EnhancedSessionRuntime) {
    runtime.pending_continuation = None;
    runtime.pending_repetition_notices.clear();
    runtime.pending_standalone_repetition_notice = false;
    runtime.hooks.on_new_user_input();
}

pub(crate) fn on_assistant_success(runtime: &mut EnhancedSessionRuntime) {
    runtime.hooks.on_assistant_success();
}

pub(crate) fn on_turn_idle(runtime: &mut EnhancedSessionRuntime) {
    runtime.pending_continuation = None;
    runtime.pending_repetition_notices.clear();
    runtime.pending_standalone_repetition_notice = false;
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

/// Build the pruner's view of the conversation, carrying Codex's own token
/// estimate for every item it produces.
///
/// The pruner cannot recompute that number: it sees decoded text, while Codex
/// counts the serialized item and applies per-modality adjustments, so an image
/// or any other base64 payload is worth far less to Codex than its bytes
/// suggest. Passing the estimate down keeps one authority for "how big is this"
/// instead of two that disagree exactly where the disagreement is largest.
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
        // An item maps to one surface item or none; keep the two vectors in
        // step whichever it was, so index N always describes item N.
        let estimate = u64::try_from(estimate_item_token_count(item)).unwrap_or(0);
        while surface.item_token_estimates.len() < surface.items.len() {
            surface.item_token_estimates.push(Some(estimate));
        }
    }
    surface
}

fn apply_surface_to_items(items: &mut [ResponseItem], surface: &ModelVisibleSurface) {
    let mut texts = std::collections::HashMap::new();
    for item in &surface.items {
        if let SurfaceItem::ToolResult {
            call_id, blocks, ..
        } = item
        {
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
                if let Some(id) = call_id
                    && let Some(text) = texts.get(id)
                {
                    output.body = FunctionCallOutputBody::Text(text.clone());
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

fn output_identity_and_text(item: &ResponseItem) -> Option<(&str, String)> {
    match item {
        ResponseItem::FunctionCallOutput {
            call_id: Some(call_id),
            output,
            ..
        }
        | ResponseItem::CustomToolCallOutput {
            call_id, output, ..
        } => output.body.to_text().map(|text| (call_id.as_str(), text)),
        _ => None,
    }
}

fn replace_output_text(item: &mut ResponseItem, text: String) {
    match item {
        ResponseItem::FunctionCallOutput { output, .. }
        | ResponseItem::CustomToolCallOutput { output, .. } => {
            output.body = FunctionCallOutputBody::Text(text);
        }
        _ => {}
    }
}

fn remember_context_projections(
    runtime: &mut EnhancedSessionRuntime,
    original: &[ResponseItem],
    projected: &[ResponseItem],
) {
    if !runtime.hooks.features.deepseek_context_recovery {
        return;
    }
    let projected_by_call = projected
        .iter()
        .filter_map(output_identity_and_text)
        .collect::<std::collections::HashMap<_, _>>();
    for item in original {
        let Some((call_id, original_text)) = output_identity_and_text(item) else {
            continue;
        };
        let Some(projected_text) = projected_by_call.get(call_id) else {
            continue;
        };
        runtime
            .context_projections
            .record(call_id, &original_text, projected_text);
    }
}

pub(crate) fn apply_repetition_notices(
    runtime: &mut EnhancedSessionRuntime,
    items: &mut Vec<ResponseItem>,
) {
    if !runtime.pending_repetition_notices.is_empty() {
        for item in items.iter_mut() {
            let Some((call_id, text)) = output_identity_and_text(item) else {
                continue;
            };
            if runtime.pending_repetition_notices.contains(call_id)
                && !text.contains(REPETITION_NOTICE)
            {
                replace_output_text(item, format!("{text}\n{REPETITION_NOTICE}"));
            }
        }
    }
    if runtime.pending_standalone_repetition_notice {
        runtime.pending_standalone_repetition_notice = false;
        let already = items.iter().any(|item| match item {
            ResponseItem::Message { content, .. } => content_text(content).contains(REPETITION_NOTICE),
            _ => output_identity_and_text(item)
                .is_some_and(|(_, text)| text.contains(REPETITION_NOTICE)),
        });
        if !already {
            items.push(ResponseItem::Message {
                id: None,
                role: "user".into(),
                content: vec![ContentItem::InputText {
                    text: REPETITION_NOTICE.to_string(),
                }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            });
        }
    }
}

pub(crate) fn apply_context_projections(
    runtime: &mut EnhancedSessionRuntime,
    items: &mut [ResponseItem],
) {
    if !runtime.hooks.features.deepseek_context_recovery {
        return;
    }
    let before_token_estimate = response_items_to_surface(items).estimated_tokens();
    let mut applied = 0_u64;
    let mut chars_removed = 0_u64;
    let mut first_call_id = None;
    for item in items.iter_mut() {
        let Some((call_id, original_text)) = output_identity_and_text(item) else {
            continue;
        };
        let replacement = runtime
            .context_projections
            .replacement(call_id, &original_text)
            .map(str::to_string);
        let Some(replacement) = replacement else {
            continue;
        };
        applied = applied.saturating_add(1);
        chars_removed = chars_removed
            .saturating_add(original_text.len().saturating_sub(replacement.len()) as u64);
        first_call_id.get_or_insert_with(|| call_id.to_string());
        replace_output_text(item, replacement);
    }
    let stats = runtime
        .context_projections
        .note_application(applied, chars_removed);
    if stats.applied == 0 {
        return;
    }
    let start = runtime.telemetry.events.len();
    let after_token_estimate = response_items_to_surface(items).estimated_tokens();
    let fields = EnhancedEventFields {
        call_id_hash: first_call_id.as_deref().map(hash_identifier),
        before_token_estimate: Some(before_token_estimate),
        after_token_estimate: Some(after_token_estimate),
        chars_removed: Some(stats.chars_removed),
        ..EnhancedEventFields::default()
    };
    runtime.telemetry.emit(EnhancedEvent::new(
        EnhancedEventKind::ContextProjectionApplied,
        fields.clone(),
    ));
    if stats.restored {
        runtime.telemetry.emit(EnhancedEvent::new(
            EnhancedEventKind::ContextProjectionRestored,
            fields,
        ));
    }
    log_new_events(&runtime.telemetry, start);
}

pub(crate) fn clear_context_projections_after_compaction(runtime: &mut EnhancedSessionRuntime) {
    if !runtime.hooks.features.deepseek_context_recovery
        || !runtime.context_projections.clear_after_compaction()
    {
        return;
    }
    let start = runtime.telemetry.events.len();
    runtime.telemetry.emit(EnhancedEvent::new(
        EnhancedEventKind::ContextProjectionCleared,
        EnhancedEventFields::default(),
    ));
    log_new_events(&runtime.telemetry, start);
}

pub(crate) fn apply_pressure_to_prompt(
    runtime: &mut EnhancedSessionRuntime,
    items: &mut [ResponseItem],
    context_window_tokens: u64,
    compact_threshold_tokens: u64,
) -> PressureSeam {
    if context_window_tokens == 0 {
        return PressureSeam::DeferToUpstream;
    }
    let original = items.to_vec();
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
                remember_context_projections(runtime, &original, items);
            }
            runtime.last_surface = Some(plan.prune.surface);
            match plan.compact {
                CompactDecision::Skip => PressureSeam::SkipNativeCompact,
                CompactDecision::RunNativeCodexLocalCompact => PressureSeam::RunNativeCompact,
            }
        }
    }
}

/// Keep exactly one provider call per call id in the model-visible prompt.
/// The execution seam already suppresses replayed side effects; this closes
/// the second half of the contract so a duplicate cannot poison the next
/// provider request with an invalid call/output pairing.
pub(crate) fn remove_replayed_tool_calls_from_prompt(
    runtime: &EnhancedSessionRuntime,
    items: &mut Vec<ResponseItem>,
) {
    if !runtime.hooks.features.qwen_tool_reliability {
        return;
    }
    let mut seen = std::collections::HashMap::new();
    items.retain(|item| match item {
        ResponseItem::FunctionCall {
            call_id,
            name,
            arguments,
            ..
        } => match seen.get(call_id) {
            Some((seen_name, seen_payload)) => seen_name != name || seen_payload != arguments,
            None => {
                seen.insert(call_id.clone(), (name.clone(), arguments.clone()));
                true
            }
        },
        ResponseItem::CustomToolCall {
            call_id,
            name,
            input,
            ..
        } => match seen.get(call_id) {
            Some((seen_name, seen_payload)) => seen_name != name || seen_payload != input,
            None => {
                seen.insert(call_id.clone(), (name.clone(), input.clone()));
                true
            }
        },
        _ => true,
    });
}

/// Suppress exact duplicate calls inside one provider response before Codex
/// schedules either call. Returning a synthetic failure makes the model retry
/// the side effect under a fresh call id, so the duplicate is removed at the
/// stream boundary and only the original reaches the tool runtime.
pub(crate) fn suppress_replayed_tool_call_in_response(
    runtime: &mut EnhancedSessionRuntime,
    item: &ResponseItem,
    seen: &mut std::collections::HashMap<String, (String, String)>,
) -> bool {
    if !runtime.hooks.features.qwen_tool_reliability {
        return false;
    }
    let (call_id, name, payload) = match item {
        ResponseItem::FunctionCall {
            call_id,
            name,
            arguments,
            ..
        } => (call_id, name, arguments),
        ResponseItem::CustomToolCall {
            call_id,
            name,
            input,
            ..
        } => (call_id, name, input),
        _ => return false,
    };
    match seen.get(call_id) {
        Some((seen_name, seen_payload)) if seen_name == name && seen_payload == payload => {
            let start = runtime.telemetry.events.len();
            let fields = EnhancedEventFields {
                call_id_hash: Some(hash_identifier(call_id)),
                ..EnhancedEventFields::default()
            };
            runtime.telemetry.emit(EnhancedEvent::new(
                EnhancedEventKind::ToolDuplicateDetected,
                fields.clone(),
            ));
            runtime.telemetry.emit(EnhancedEvent::new(
                EnhancedEventKind::ToolDuplicateSuppressed,
                fields,
            ));
            log_new_events(&runtime.telemetry, start);
            true
        }
        Some(_) => false,
        None => {
            seen.insert(call_id.clone(), (name.clone(), payload.clone()));
            false
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
        // A provider-confirmed overflow is stronger evidence than the local
        // estimate. Force the prune trigger while retaining the real window
        // as the threshold that decides whether native compact is necessary.
        1,
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
    if after.byte_count() < before.byte_count() && after.generation <= before.generation {
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
    let mut successful_outputs = std::collections::HashSet::new();
    for item in items.iter().rev() {
        if let ResponseItem::FunctionCallOutput {
            call_id: Some(call_id),
            output,
            ..
        } = item
        {
            if output.success == Some(true) {
                successful_outputs.insert(call_id.as_str());
            }
            continue;
        }
        let ResponseItem::FunctionCall {
            name,
            arguments,
            call_id,
            ..
        } = item
        else {
            continue;
        };
        if name != "update_plan" {
            continue;
        }
        // Arguments are model output, not trusted control state. Only a plan
        // accepted by the native PlanHandler may arm automatic continuation.
        if !successful_outputs.contains(call_id.as_str()) {
            break;
        }
        if let Ok(args) = serde_json::from_str::<UpdatePlanArgs>(arguments)
            && args
                .plan
                .iter()
                .any(|step| matches!(step.status, StepStatus::Pending | StepStatus::InProgress))
        {
            unfinished.push(UnfinishedSignal::NativePlanIncomplete);
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
    last_agent_message: Option<String>,
) -> bool {
    let intent_continuation_enabled = sess
        .enhanced
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .hooks
        .features
        .intent_continuation;
    let context = TurnStopContext {
        cancelled: cancellation_token.is_cancelled(),
        user_steer_pending,
        unfinished: unfinished_from_items(items),
        intent_continuation_enabled,
        natural_stop: true,
        assistant_final_text: last_agent_message,
    };
    let (plan, reserved) = {
        let mut runtime = sess
            .enhanced
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .hooks
                    .release_continuation();
                return false;
            };
            sess.record_response_item_and_emit_turn_item(turn_context, item)
                .await;
            sess.enhanced
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
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
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    runtime.pending_continuation = None;
    runtime.hooks.release_continuation();
}

pub(crate) async fn restore_ledger_if_needed(sess: &Session) {
    {
        let runtime = sess
            .enhanced
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if runtime.ledger_restored || !runtime.hooks.features.qwen_tool_reliability {
            return;
        }
    }
    let history = sess.clone_history().await;
    let items = history.raw_items().cloned().collect::<Vec<_>>();
    let mut runtime = sess
        .enhanced
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
                let identity = ToolCallLedger::identity_raw(
                    call_id.clone(),
                    name,
                    ToolKind::Custom,
                    "",
                    input,
                );
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

#[cfg(test)]
mod tests {
    use super::super::config::AblationProfile;
    use super::super::hooks::EnhancedTurnHooks;
    use super::*;
    use codex_protocol::models::FunctionCallOutputPayload;
    use serde_json::json;

    fn call_item(call_id: &str, arguments: &str) -> ResponseItem {
        ResponseItem::FunctionCall {
            id: None,
            name: "shell".into(),
            namespace: None,
            arguments: arguments.into(),
            encrypted_function_args: None,
            call_id: call_id.into(),
            internal_chat_message_metadata_passthrough: None,
        }
    }

    fn output_item(call_id: &str) -> ResponseItem {
        ResponseItem::FunctionCallOutput {
            id: None,
            call_id: Some(call_id.into()),
            name: Some("shell".into()),
            namespace: None,
            output: FunctionCallOutputPayload::from_text("ok".into()),
            internal_chat_message_metadata_passthrough: None,
        }
    }

    fn plan_call_item(call_id: &str, status: &str) -> ResponseItem {
        ResponseItem::FunctionCall {
            id: None,
            name: "update_plan".into(),
            namespace: None,
            arguments: json!({
                "explanation": null,
                "plan": [{"step": "finish", "status": status}]
            })
            .to_string(),
            encrypted_function_args: None,
            call_id: call_id.into(),
            internal_chat_message_metadata_passthrough: None,
        }
    }

    fn plan_output_item(call_id: &str, success: bool) -> ResponseItem {
        let mut output = FunctionCallOutputPayload::from_text("Plan updated".into());
        output.success = Some(success);
        ResponseItem::FunctionCallOutput {
            id: None,
            call_id: Some(call_id.into()),
            name: Some("update_plan".into()),
            namespace: None,
            output,
            internal_chat_message_metadata_passthrough: None,
        }
    }

    #[test]
    fn only_a_successfully_applied_native_plan_can_arm_continuation() {
        let pending = plan_call_item("plan-1", "pending");
        assert!(unfinished_from_items(std::slice::from_ref(&pending)).is_empty());
        assert!(
            unfinished_from_items(&[pending.clone(), plan_output_item("plan-1", false)]).is_empty()
        );
        assert_eq!(
            unfinished_from_items(&[pending, plan_output_item("plan-1", true)]),
            vec![UnfinishedSignal::NativePlanIncomplete]
        );
        assert!(
            unfinished_from_items(&[
                plan_call_item("plan-2", "completed"),
                plan_output_item("plan-2", true),
            ])
            .is_empty()
        );
    }

    #[test]
    fn empty_history_first_call_executes() {
        let mut ledger = ledger_from_history(&[]);
        let identity = ToolCallLedger::identity("c1", "shell", &json!({"a": 1}));
        let outcome = ledger.admit(identity);
        assert!(matches!(outcome.decision, AdmitDecision::Execute));
    }

    #[test]
    fn first_original_result_is_forwarded() {
        let mut runtime = EnhancedSessionRuntime::all_off();
        runtime.hooks = EnhancedTurnHooks::new(AblationProfile::E1.features());
        let mut telemetry = MemoryTelemetry::default();
        let identity = ToolCallLedger::identity("c1", "shell", &json!({"a": 1}));
        assert!(matches!(
            runtime.hooks.admit_tool_call(identity, &mut telemetry),
            HookDecision::Handled(AdmitDecision::Execute)
        ));
        assert!(complete_tool_call(&mut runtime, "c1", true).is_ok());
    }

    #[test]
    fn replayed_provider_call_is_removed_only_when_e1_is_enabled() {
        let duplicate = call_item("c1", r#"{"path":"result.txt"}"#);
        let original = duplicate.clone();
        let output = output_item("c1");

        let mut disabled_items = vec![original.clone(), duplicate.clone(), output.clone()];
        remove_replayed_tool_calls_from_prompt(
            &EnhancedSessionRuntime::all_off(),
            &mut disabled_items,
        );
        assert_eq!(disabled_items.len(), 3);

        let mut runtime = EnhancedSessionRuntime::all_off();
        runtime.hooks = EnhancedTurnHooks::new(AblationProfile::E1.features());
        let mut enabled_items = vec![original, duplicate, output.clone()];
        remove_replayed_tool_calls_from_prompt(&runtime, &mut enabled_items);
        assert_eq!(
            enabled_items,
            vec![call_item("c1", r#"{"path":"result.txt"}"#), output]
        );
    }

    #[test]
    fn colliding_call_id_is_preserved_for_fail_closed_handling() {
        let mut runtime = EnhancedSessionRuntime::all_off();
        runtime.hooks = EnhancedTurnHooks::new(AblationProfile::E1.features());
        let mut items = vec![
            call_item("c1", r#"{"value":1}"#),
            call_item("c1", r#"{"value":2}"#),
        ];
        remove_replayed_tool_calls_from_prompt(&runtime, &mut items);
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn response_replay_is_suppressed_before_tool_scheduling() {
        let mut runtime = EnhancedSessionRuntime::all_off();
        runtime.hooks = EnhancedTurnHooks::new(AblationProfile::E1.features());
        let call = call_item("c1", r#"{"value":1}"#);
        let mut seen = std::collections::HashMap::new();
        assert!(!suppress_replayed_tool_call_in_response(
            &mut runtime,
            &call,
            &mut seen
        ));
        assert!(suppress_replayed_tool_call_in_response(
            &mut runtime,
            &call,
            &mut seen
        ));
        assert_eq!(
            runtime
                .telemetry
                .count(EnhancedEventKind::ToolDuplicateSuppressed),
            1
        );
        assert!(runtime.hooks.ledger.is_empty());
    }

    #[test]
    fn provider_overflow_forces_prune_below_local_pressure_threshold() {
        let mut runtime = EnhancedSessionRuntime::all_off();
        runtime.hooks = EnhancedTurnHooks::new(AblationProfile::E2.features());
        let call = call_item("c1", "{}");
        let mut output = output_item("c1");
        if let ResponseItem::FunctionCallOutput { output, .. } = &mut output {
            *output = FunctionCallOutputPayload::from_text("x".repeat(8_000));
        }
        assert!(matches!(
            plan_overflow_for_prompt(&mut runtime, &[call, output], 24_000, false),
            OverflowSeam::Retry { .. }
        ));
        assert_eq!(
            runtime
                .telemetry
                .count(EnhancedEventKind::ContextOverflowRetry),
            1
        );
    }

    #[test]
    fn context_projection_persists_across_follow_up_prompts_without_losing_new_input() {
        let mut runtime = EnhancedSessionRuntime::all_off();
        runtime.hooks = EnhancedTurnHooks::new(AblationProfile::E2.features());
        let mut original_output = output_item("c1");
        if let ResponseItem::FunctionCallOutput { output, .. } = &mut original_output {
            *output = FunctionCallOutputPayload::from_text("x".repeat(8_000));
        }
        let original = vec![call_item("c1", "{}"), original_output];
        let OverflowSeam::Retry { pruned } =
            plan_overflow_for_prompt(&mut runtime, &original, 24_000, false)
        else {
            panic!("overflow must produce one pruned retry");
        };
        assert!(
            response_items_to_surface(&pruned).byte_count()
                < response_items_to_surface(&original).byte_count()
        );

        for follow_up in 0..3 {
            let mut prompt = original.clone();
            prompt.push(ResponseItem::Message {
                id: None,
                role: "user".into(),
                content: vec![ContentItem::InputText {
                    text: format!("follow-up-{follow_up}"),
                }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            });
            apply_context_projections(&mut runtime, &mut prompt);
            assert!(
                response_items_to_surface(&prompt).byte_count()
                    < response_items_to_surface(&original).byte_count() + 100
            );
            assert!(matches!(
                prompt.last(),
                Some(ResponseItem::Message { content, .. })
                    if content_text(content) == format!("follow-up-{follow_up}")
            ));
        }
        assert_eq!(
            runtime
                .telemetry
                .count(EnhancedEventKind::ContextProjectionApplied),
            3
        );
    }

    #[test]
    fn context_projection_does_not_match_changed_output_and_clears_after_compaction() {
        let mut runtime = EnhancedSessionRuntime::all_off();
        runtime.hooks = EnhancedTurnHooks::new(AblationProfile::E2.features());
        runtime
            .context_projections
            .record("c1", "large original", "small");
        let mut changed = output_item("c1");
        if let ResponseItem::FunctionCallOutput { output, .. } = &mut changed {
            *output = FunctionCallOutputPayload::from_text("different original".into());
        }
        apply_context_projections(&mut runtime, std::slice::from_mut(&mut changed));
        assert_eq!(
            output_identity_and_text(&changed).unwrap().1,
            "different original"
        );

        clear_context_projections_after_compaction(&mut runtime);
        let mut original = output_item("c1");
        if let ResponseItem::FunctionCallOutput { output, .. } = &mut original {
            *output = FunctionCallOutputPayload::from_text("large original".into());
        }
        apply_context_projections(&mut runtime, std::slice::from_mut(&mut original));
        assert_eq!(
            output_identity_and_text(&original).unwrap().1,
            "large original"
        );
    }

    #[test]
    fn restored_completed_call_is_not_reexecuted() {
        let arguments = json!({"a": 1}).to_string();
        let mut ledger = ledger_from_history(&[call_item("c1", &arguments), output_item("c1")]);
        let identity = ToolCallLedger::identity("c1", "shell", &json!({"a": 1}));
        let outcome = ledger.admit(identity);
        assert!(matches!(
            outcome.decision,
            AdmitDecision::SuppressDuplicate { .. }
        ));
    }

    #[test]
    fn pending_continuation_clears_on_idle_and_new_input() {
        let mut runtime = EnhancedSessionRuntime::all_off();
        runtime.pending_continuation = Some(ReservedContinuation { index: 1 });
        on_turn_idle(&mut runtime);
        assert!(runtime.pending_continuation.is_none());
        runtime.pending_continuation = Some(ReservedContinuation { index: 1 });
        on_new_user_input(&mut runtime);
        assert!(runtime.pending_continuation.is_none());
    }

    #[test]
    fn late_original_after_synthetic_duplicate_is_not_forwarded() {
        let mut runtime = EnhancedSessionRuntime::all_off();
        runtime.hooks = EnhancedTurnHooks::new(AblationProfile::E1.features());
        let mut telemetry = MemoryTelemetry::default();
        let identity = ToolCallLedger::identity("c1", "shell", &json!({"a": 1}));
        assert!(matches!(
            runtime
                .hooks
                .admit_tool_call(identity.clone(), &mut telemetry),
            HookDecision::Handled(AdmitDecision::Execute)
        ));
        assert!(matches!(
            runtime.hooks.admit_tool_call(identity, &mut telemetry),
            HookDecision::Handled(AdmitDecision::SuppressDuplicate { .. })
        ));
        runtime
            .hooks
            .mark_tool_resolution("c1", ToolCallResolution::SyntheticDuplicate);
        let forwarded = complete_tool_call(&mut runtime, "c1", true);
        assert!(is_drop_tool_output(
            forwarded.as_ref().expect_err("late original must drop")
        ));
    }

    fn custom_call(call_id: &str, input: &str) -> crate::tools::router::ToolCall {
        crate::tools::router::ToolCall {
            tool_name: codex_tools::ToolName::plain("apply_patch"),
            call_id: call_id.into(),
            payload: ToolPayload::Custom {
                input: input.into(),
            },
            encrypted_function_args: None,
        }
    }

    fn shell_call(call_id: &str, cmd: &str) -> crate::tools::router::ToolCall {
        crate::tools::router::ToolCall {
            tool_name: codex_tools::ToolName::plain("shell"),
            call_id: call_id.into(),
            payload: ToolPayload::Function {
                arguments: json!({"cmd": cmd}).to_string(),
            },
            encrypted_function_args: None,
        }
    }

    #[test]
    fn custom_input_keeps_whitespace_and_is_not_json_canonicalized() {
        let mut runtime = EnhancedSessionRuntime::all_off();
        runtime.hooks = EnhancedTurnHooks::new(AblationProfile::E1.features());
        let patch = "*** Begin Patch\n*** Update File: a.rs\n@@\n- old\n+ 新\n";
        assert!(admit_tool_call(&mut runtime, &custom_call("c1", patch)).is_ok());
        let stored = runtime.hooks.ledger.get("c1").unwrap();
        assert_eq!(stored.identity.tool_kind, ToolKind::Custom);
        assert_eq!(
            stored.identity.argument_fingerprint,
            ToolCallLedger::fingerprint_raw("apply_patch", ToolKind::Custom, "", patch)
        );
        let collapsed = "*** Begin Patch\n*** Update File: a.rs\n@@\n-old\n+新\n";
        assert_ne!(
            stored.identity.argument_fingerprint,
            ToolCallLedger::fingerprint_raw("apply_patch", ToolKind::Custom, "", collapsed)
        );
    }

    #[test]
    fn distinct_patches_with_the_same_success_are_not_blocked() {
        let mut runtime = EnhancedSessionRuntime::all_off();
        runtime.hooks = EnhancedTurnHooks::new(AblationProfile::E5.features());
        let success = "Success. Updated the following files:\nM src-tauri/src/commands/runtime.rs";
        for (index, patch) in [
            "*** Begin Patch\nA\n",
            "*** Begin Patch\nB\n",
            "*** Begin Patch\nC\n",
        ]
        .into_iter()
        .enumerate()
        {
            let call = custom_call(&format!("p{index}"), patch);
            assert!(admit_tool_call(&mut runtime, &call).is_ok());
            assert!(complete_tool_call_with_result(&mut runtime, &call.call_id, true, Some(success))
                .unwrap()
                .is_none());
        }
        assert!(runtime.pending_repetition_notices.is_empty());
        assert_eq!(
            runtime
                .telemetry
                .count(EnhancedEventKind::ToolRepetitionObserved),
            0
        );
    }

    #[test]
    fn twenty_five_distinct_ops_without_assistant_text_are_not_blocked() {
        let mut runtime = EnhancedSessionRuntime::all_off();
        runtime.hooks = EnhancedTurnHooks::new(AblationProfile::E5.features());
        for index in 0..25 {
            let call = shell_call(&format!("op{index}"), &format!("step-{index}"));
            assert!(admit_tool_call(&mut runtime, &call).is_ok());
            assert!(complete_tool_call_with_result(
                &mut runtime,
                &call.call_id,
                true,
                Some("ok")
            )
            .unwrap()
            .is_none());
        }
        assert!(runtime.pending_repetition_notices.is_empty());
        let plan = {
            let (hooks, telemetry) = split_runtime(&mut runtime);
            hooks.on_turn_stop(
                &TurnStopContext {
                    natural_stop: true,
                    assistant_final_text: None,
                    ..TurnStopContext::default()
                },
                telemetry,
            )
        };
        assert!(matches!(
            plan,
            HookDecision::Handled(ref inner) if inner.decision == ContinuationDecision::AllowStop
        ));
    }

    #[test]
    fn flags_off_scripted_path_does_not_inject_notice_or_text_continuation() {
        let mut runtime = EnhancedSessionRuntime::all_off();
        runtime.hooks = EnhancedTurnHooks::new(AblationProfile::E5.features());
        let patch = "*** Begin Patch\nsame\n";
        let success = "ok";
        for index in 0..3 {
            let call = custom_call(&format!("same{index}"), patch);
            assert!(admit_tool_call(&mut runtime, &call).is_ok());
            let notice = complete_tool_call_with_result(
                &mut runtime,
                &call.call_id,
                true,
                Some(success),
            )
            .unwrap();
            assert!(notice.is_none());
        }
        assert!(runtime.pending_repetition_notices.is_empty());
        assert_eq!(
            runtime
                .telemetry
                .count(EnhancedEventKind::ToolRepetitionNoticeAppended),
            0
        );
        let mut items = vec![output_item("same2")];
        apply_repetition_notices(&mut runtime, &mut items);
        assert_eq!(output_identity_and_text(&items[0]).unwrap().1, "ok");
        let plan = {
            let (hooks, telemetry) = split_runtime(&mut runtime);
            hooks.on_turn_stop(
                &TurnStopContext {
                    intent_continuation_enabled: false,
                    natural_stop: true,
                    assistant_final_text: Some("I will continue immediately.".into()),
                    ..TurnStopContext::default()
                },
                telemetry,
            )
        };
        assert!(matches!(
            plan,
            HookDecision::Handled(ref inner) if inner.decision == ContinuationDecision::AllowStop
        ));
    }

    #[test]
    fn scripted_provider_path_admits_executes_observes_and_stops() {
        let mut runtime = EnhancedSessionRuntime::all_off();
        runtime.hooks = EnhancedTurnHooks::new(AblationProfile::E1.features());
        let call = shell_call("wired", "echo hi");
        assert!(admit_tool_call(&mut runtime, &call).is_ok());
        assert!(complete_tool_call_with_result(&mut runtime, "wired", true, Some("hi")).is_ok());
        assert_eq!(
            runtime.telemetry.count(EnhancedEventKind::ToolCallAdmitted),
            1
        );
        assert_eq!(
            runtime
                .telemetry
                .count(EnhancedEventKind::ToolCallCompleted),
            1
        );
        on_new_user_input(&mut runtime);
        assert!(runtime.pending_repetition_notices.is_empty());
    }

    #[test]
    fn out_of_order_identical_ops_inject_a_standalone_notice_not_the_late_middle_result() {
        let mut runtime = EnhancedSessionRuntime::all_off();
        runtime.hooks = EnhancedTurnHooks::new(AblationProfile::E5.features());
        runtime.hooks.features.repetition_notice = true;
        let patch = "*** Begin Patch\nsame\n";
        let success = "ok";
        let c0 = custom_call("c0", patch);
        let c1 = custom_call("c1", patch);
        let c2 = custom_call("c2", patch);
        assert!(admit_tool_call(&mut runtime, &c0).is_ok());
        assert!(admit_tool_call(&mut runtime, &c1).is_ok());
        assert!(admit_tool_call(&mut runtime, &c2).is_ok());
        assert!(complete_tool_call_with_result(&mut runtime, "c0", true, Some(success))
            .unwrap()
            .is_none());
        assert!(complete_tool_call_with_result(&mut runtime, "c2", true, Some(success))
            .unwrap()
            .is_none());
        let middle = complete_tool_call_with_result(&mut runtime, "c1", true, Some(success)).unwrap();
        assert!(
            middle.is_none(),
            "must not attach the notice to the late-completing middle result"
        );
        assert!(runtime.pending_repetition_notices.is_empty());
        assert!(runtime.pending_standalone_repetition_notice);
        let mut items = vec![output_item("c0"), output_item("c2"), output_item("c1")];
        apply_repetition_notices(&mut runtime, &mut items);
        assert!(!runtime.pending_standalone_repetition_notice);
        assert_eq!(output_identity_and_text(&items[2]).unwrap().1, "ok");
        assert!(
            matches!(
                items.last(),
                Some(ResponseItem::Message { content, role, .. })
                    if role == "user" && content_text(content) == REPETITION_NOTICE
            ),
            "standalone notice must be a separate model-visible message: {:?}",
            items.last()
        );
        assert_eq!(
            runtime
                .telemetry
                .count(EnhancedEventKind::ToolRepetitionNoticeAppended),
            1
        );
    }
}
