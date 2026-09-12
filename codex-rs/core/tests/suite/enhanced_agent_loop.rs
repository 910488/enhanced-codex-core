//! Deterministic agent-loop fixtures for Enhanced Codex ports A/B/C.
//!
//! These tests enter through `run_turn` and the real stream path
//! (`ToolRouter`, native compact, stop-hook continuation). They do not
//! call portable unit functions.

use std::fs;
use std::time::Duration;

use std::sync::Arc;
use std::sync::Mutex;

use anyhow::Result;
use codex_core::TurnInputRequest;
use codex_core::compact::SUMMARIZATION_PROMPT;
use codex_features::Feature;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::config_types::AutoCompactTokenLimitScope;
use codex_protocol::config_types::CollaborationMode;
use codex_protocol::config_types::ModeKind;
use codex_protocol::config_types::Settings;
use codex_protocol::error::CodexErr;
use codex_protocol::models::PermissionProfile;
use codex_protocol::protocol::AskForApproval;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::protocol::ThreadSettingsOverrides;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ResponsesRequest;
use core_test_support::responses::ev_apply_patch_custom_tool_call;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_function_call;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_response_sequence;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::sse_failed;
use core_test_support::responses::sse_response;
use core_test_support::responses::start_mock_server;
use core_test_support::TempDirExt;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodex;
use core_test_support::test_codex::TestCodexBuilder;
use core_test_support::test_codex::local_selections;
use core_test_support::test_codex::test_codex;
use core_test_support::test_codex::turn_permission_fields;
use core_test_support::wait_for_event_with_timeout;
use serde_json::Value;
use serde_json::json;
use wiremock::Match;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::Request;
use wiremock::Respond;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path_regex;

const CALL_ID: &str = "c1";
const CONTINUE_NUDGE: &str =
    "Continue the unfinished deterministic work. Do not wait for a new user message.";
const TURN_TIMEOUT: Duration = Duration::from_secs(30);
const CONTEXT_LIMIT_MESSAGE: &str =
    "Your input exceeds the context window of this model. Please adjust your input and try again.";

fn write_enhanced_runtime(home: &std::path::Path, profile: &str) {
    std::fs::write(
        home.join("enhanced-runtime.json"),
        format!(r#"{{"ablationProfile":"{profile}"}}"#),
    )
    .expect("write enhanced-runtime.json");
}

fn enhanced_builder(profile: &'static str) -> TestCodexBuilder {
    test_codex()
        .with_pre_build_hook(move |home| write_enhanced_runtime(home, profile))
        .with_config(|config| {
            config.update_plan_enabled = true;
        })
}

fn unfinished_plan_args() -> String {
    json!({
        "explanation": "agent-loop fixture",
        "plan": [
            {"step": "Do the work", "status": "in_progress"},
            {"step": "Finish", "status": "pending"},
        ],
    })
    .to_string()
}

fn function_call_output_count(request: &ResponsesRequest, call_id: &str) -> usize {
    request
        .input()
        .iter()
        .filter(|item| {
            item.get("type").and_then(Value::as_str) == Some("function_call_output")
                && item.get("call_id").and_then(Value::as_str) == Some(call_id)
        })
        .count()
}

fn max_function_call_outputs(requests: &[ResponsesRequest], call_id: &str) -> usize {
    requests
        .iter()
        .map(|request| function_call_output_count(request, call_id))
        .max()
        .unwrap_or(0)
}

#[derive(Clone)]
struct CapturingMatch {
    requests: Arc<Mutex<Vec<Request>>>,
}

impl Match for CapturingMatch {
    fn matches(&self, request: &Request) -> bool {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(request.clone());
        true
    }
}

fn json_fragment(text: &str) -> String {
    serde_json::to_string(text)
        .expect("serialize text")
        .trim_matches('"')
        .to_string()
}

fn request_is_compact(request: &Request) -> bool {
    String::from_utf8_lossy(&request.body).contains(&json_fragment(SUMMARIZATION_PROMPT))
}

fn request_input_char_count(request: &Request) -> usize {
    let Ok(body) = serde_json::from_slice::<Value>(&request.body) else {
        return request.body.len();
    };
    body.get("input")
        .and_then(Value::as_array)
        .map(|items| items.iter().map(|item| item.to_string().len()).sum())
        .unwrap_or(request.body.len())
}

struct OverflowOrCompact {
    overflow: String,
    compact_ok: String,
}

impl Respond for OverflowOrCompact {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        if request_is_compact(request) {
            sse_response(self.compact_ok.clone())
        } else {
            sse_response(self.overflow.clone())
        }
    }
}

async fn mount_overflow_and_compact(server: &MockServer) -> Arc<Mutex<Vec<Request>>> {
    let captured = CapturingMatch {
        requests: Arc::new(Mutex::new(Vec::new())),
    };
    Mock::given(method("POST"))
        .and(path_regex(".*/(responses|guardian)$"))
        .and(captured.clone())
        .respond_with(OverflowOrCompact {
            overflow: sse_failed("resp-cwe", "context_length_exceeded", CONTEXT_LIMIT_MESSAGE),
            compact_ok: assistant_sse("resp-compact", "SUMMARY_ONLY_CONTEXT"),
        })
        .mount(server)
        .await;
    captured.requests
}

fn contains_nudge(request: &ResponsesRequest) -> bool {
    request.body_contains_text(CONTINUE_NUDGE)
}

async fn wait_turn_complete(codex: &codex_core::CodexThread) -> usize {
    let mut plan_updates = 0usize;
    wait_for_event_with_timeout(
        codex,
        |event| match event {
            EventMsg::PlanUpdate(_) => {
                plan_updates += 1;
                false
            }
            EventMsg::TurnComplete(_) => true,
            _ => false,
        },
        TURN_TIMEOUT,
    )
    .await;
    plan_updates
}

async fn submit_text(test: &TestCodex, text: &str) -> Result<()> {
    test.codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: text.into(),
            text_elements: Vec::new(),
        }]))
        .await?;
    Ok(())
}

fn plan_call_sse(response_id: &str) -> String {
    sse(vec![
        ev_response_created(response_id),
        ev_function_call(CALL_ID, "update_plan", &unfinished_plan_args()),
        ev_completed(response_id),
    ])
}

fn duplicate_plan_call_sse(response_id: &str) -> String {
    sse(vec![
        ev_response_created(response_id),
        ev_function_call(CALL_ID, "update_plan", &unfinished_plan_args()),
        ev_function_call(CALL_ID, "update_plan", &unfinished_plan_args()),
        ev_completed(response_id),
    ])
}

fn assistant_sse(response_id: &str, text: &str) -> String {
    sse(vec![
        ev_response_created(response_id),
        ev_assistant_message(&format!("{response_id}-msg"), text),
        ev_completed(response_id),
    ])
}

/// Port A: at-most-once physical execution and exactly one model-visible
/// terminal output for logical provider call ID `c1`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tool_reliability_fresh_duplicate_late_resume_single_terminal_output() -> Result<()> {
    skip_if_no_network!(Ok(()));
    Box::pin(tool_reliability_fresh_duplicate_late_resume_single_terminal_output_inner()).await
}

async fn tool_reliability_fresh_duplicate_late_resume_single_terminal_output_inner() -> Result<()> {
    let server = start_mock_server().await;
    let request_log = mount_sse_sequence(
        &server,
        vec![
            duplicate_plan_call_sse("resp-fresh"),
            assistant_sse("resp-fresh-followup", "fresh call acknowledged"),
            plan_call_sse("resp-replay"),
            assistant_sse("resp-replay-followup", "replay suppressed"),
            plan_call_sse("resp-resume"),
            assistant_sse("resp-resume-followup", "resume suppressed"),
        ],
    )
    .await;

    let mut builder = enhanced_builder("E1");
    let initial = Box::pin(builder.build(&server)).await?;

    submit_text(&initial, "fresh c1").await?;
    let mut physical = wait_turn_complete(&initial.codex).await;
    assert_eq!(physical, 1, "fresh c1 must execute exactly once");

    submit_text(&initial, "replay c1").await?;
    physical += wait_turn_complete(&initial.codex).await;
    assert_eq!(physical, 1, "duplicate/replay c1 must not execute again");

    initial.codex.ensure_rollout_materialized().await;
    let mut resume_builder = enhanced_builder("E1");
    let resumed = Box::pin(resume_builder.restart(&server, &initial)).await?;

    submit_text(&resumed, "resume c1").await?;
    physical += wait_turn_complete(&resumed.codex).await;
    assert_eq!(physical, 1, "process/session resume must not re-execute c1");

    let requests = request_log.requests();
    let terminal_outputs = max_function_call_outputs(&requests, CALL_ID);
    assert_eq!(
        terminal_outputs,
        1,
        "model-visible terminal FunctionCallOutput count for call_id=c1 must be exactly 1; got {terminal_outputs}. requests={}",
        requests.len()
    );
    assert_eq!(physical, 1, "physical execution count must remain 1");
    Ok(())
}

/// Port B: real ContextWindowExceeded recovery — prune/compact progress,
/// retry once, second overflow preserves the original error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn context_recovery_overflow_retries_once_then_preserves_original_error() -> Result<()> {
    skip_if_no_network!(Ok(()));
    Box::pin(context_recovery_overflow_retries_once_then_preserves_original_error_inner()).await
}

async fn context_recovery_overflow_retries_once_then_preserves_original_error_inner() -> Result<()>
{
    let server = start_mock_server().await;
    let captured = mount_overflow_and_compact(&server).await;

    let mut model_provider = built_in_model_providers(/*openai_base_url*/ None)["openai"].clone();
    model_provider.name = "OpenAI (test)".into();
    model_provider.base_url = Some(format!("{}/v1", server.uri()));
    model_provider.supports_websockets = false;
    model_provider.stream_max_retries = Some(0);
    model_provider.request_max_retries = Some(0);

    let mut builder = enhanced_builder("E2").with_config(move |config| {
        config.model_provider = model_provider;
        config.compact_prompt = Some(SUMMARIZATION_PROMPT.to_string());
        config.model_context_window = Some(1_000);
        config.model_auto_compact_token_limit = Some(10_000_000);
        config.model_auto_compact_token_limit_scope = AutoCompactTokenLimitScope::BodyAfterPrefix;
        config.include_environment_context = false;
        let _ = config.features.disable(Feature::RemoteCompactionV2);
    });
    let test = Box::pin(builder.build(&server)).await?;

    let huge = format!("overflow-trigger {}", "x".repeat(20_000));
    submit_text(&test, &huge).await?;

    let error_event = wait_for_event_with_timeout(
        &test.codex,
        |event| matches!(event, EventMsg::Error(_)),
        TURN_TIMEOUT,
    )
    .await;
    let EventMsg::Error(error) = error_event else {
        panic!("expected Error event");
    };
    assert_eq!(error.message, CodexErr::ContextWindowExceeded.to_string());
    wait_for_event_with_timeout(
        &test.codex,
        |event| matches!(event, EventMsg::TurnComplete(_)),
        TURN_TIMEOUT,
    )
    .await;

    let requests = captured
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let sampling: Vec<_> = requests
        .iter()
        .filter(|request| !request_is_compact(request))
        .collect();
    let compact: Vec<_> = requests
        .iter()
        .filter(|request| request_is_compact(request))
        .collect();

    assert_eq!(
        sampling.len(),
        2,
        "overflow must retry sampling exactly once (got {} sampling requests, {} compact)",
        sampling.len(),
        compact.len()
    );
    assert!(
        !compact.is_empty(),
        "overflow recovery must run native compact when prune cannot get under the window"
    );

    let s0 = request_input_char_count(sampling[0]);
    let s_retry = request_input_char_count(sampling[1]);
    assert!(
        s_retry < s0,
        "S1/S2 after prune/compact must be smaller than S0 ({s_retry} >= {s0})"
    );
    Ok(())
}

/// Port C: unfinished native plan continues at most twice; a pre-stream
/// failure does not consume the budget.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bounded_continuation_prestream_failure_then_two_successes_no_third() -> Result<()> {
    skip_if_no_network!(Ok(()));
    Box::pin(bounded_continuation_prestream_failure_then_two_successes_no_third_inner()).await
}

async fn bounded_continuation_prestream_failure_then_two_successes_no_third_inner() -> Result<()> {
    let server = start_mock_server().await;
    let fail = ResponseTemplate::new(500)
        .insert_header("content-type", "application/json")
        .set_body_string(
            json!({
                "error": {"type": "server_error", "message": "pre-stream failure"}
            })
            .to_string(),
        );

    let request_log = mount_response_sequence(
        &server,
        vec![
            sse_response(plan_call_sse("resp-plan")),
            sse_response(assistant_sse("resp-after-plan", "plan is still unfinished")),
            fail,
            sse_response(assistant_sse(
                "resp-continue-1",
                "first continuation after pre-stream failure",
            )),
            sse_response(assistant_sse(
                "resp-continue-2",
                "second continuation; budget exhausted after this",
            )),
        ],
    )
    .await;

    let mut builder = enhanced_builder("E3").with_config(|config| {
        config.model_provider.stream_max_retries = Some(1);
        config.model_provider.request_max_retries = Some(1);
    });
    let test = Box::pin(builder.build(&server)).await?;

    submit_text(&test, "start unfinished plan").await?;
    wait_for_event_with_timeout(
        &test.codex,
        |event| matches!(event, EventMsg::TurnComplete(_)),
        TURN_TIMEOUT,
    )
    .await;

    let requests = request_log.requests();
    let nudge_requests: Vec<_> = requests
        .iter()
        .filter(|request| contains_nudge(request))
        .collect();

    // Pre-stream 500 + successful retry is continuation #1 (used stays 0
    // until the successful stream). The next successful stream is #2.
    assert!(
        nudge_requests.len() >= 2,
        "expected reserved continuation requests; got {} of {}",
        nudge_requests.len(),
        requests.len()
    );
    assert!(
        nudge_requests.len() <= 3,
        "no third continuation: nudge requests={} total={}",
        nudge_requests.len(),
        requests.len()
    );
    assert_eq!(
        requests.len(),
        5,
        "plan + follow-up + failed continue + continue#1 + continue#2; extra request would be a third continuation"
    );
    Ok(())
}

const REPETITION_NOTICE: &str = "Notice: the same tool, input, and original result have now occurred three times in a row. Consider a different approach.";

fn completed_plan_args() -> String {
    json!({
        "explanation": "agent-loop fixture",
        "plan": [{"step": "Finish", "status": "completed"}],
    })
    .to_string()
}

fn request_contains_notice(request: &ResponsesRequest) -> bool {
    request.body_contains_text(REPETITION_NOTICE)
}

async fn submit_unrestricted(test: &TestCodex, text: &str) -> Result<()> {
    let cwd_path = test.cwd.abs();
    let (sandbox_policy, permission_profile) =
        turn_permission_fields(PermissionProfile::Disabled, cwd_path.as_path());
    test.codex
        .start_or_steer_turn(
            TurnInputRequest::user_input(vec![UserInput::Text {
                text: text.into(),
                text_elements: Vec::new(),
            }])
            .with_thread_settings(ThreadSettingsOverrides {
                environments: Some(local_selections(cwd_path)),
                approval_policy: Some(AskForApproval::Never),
                sandbox_policy: Some(sandbox_policy),
                permission_profile,
                collaboration_mode: Some(CollaborationMode {
                    mode: ModeKind::Default,
                    settings: Settings {
                        model: test.session_configured.model.clone(),
                        reasoning_effort: None,
                        developer_instructions: None,
                    },
                }),
                ..Default::default()
            }),
        )
        .await?;
    Ok(())
}

/// E5 keeps new experiments off. Three identical successful tools plus a
/// trailing "I will continue immediately." must not inject a model-visible
/// notice or an extra text-continuation request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flags_off_scripted_stream_injects_neither_notice_nor_text_continuation() -> Result<()> {
    skip_if_no_network!(Ok(()));
    Box::pin(flags_off_scripted_stream_injects_neither_notice_nor_text_continuation_inner()).await
}

async fn flags_off_scripted_stream_injects_neither_notice_nor_text_continuation_inner() -> Result<()>
{
    let server = start_mock_server().await;
    let args = completed_plan_args();
    let request_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-tools"),
                ev_function_call("obs-1", "update_plan", &args),
                ev_function_call("obs-2", "update_plan", &args),
                ev_function_call("obs-3", "update_plan", &args),
                ev_completed("resp-tools"),
            ]),
            assistant_sse("resp-final", "I will continue immediately."),
        ],
    )
    .await;

    let mut builder = enhanced_builder("E5");
    let test = Box::pin(builder.build(&server)).await?;
    submit_text(&test, "run three identical completed plans").await?;
    let plan_updates = wait_turn_complete(&test.codex).await;
    assert_eq!(plan_updates, 3, "all three new-id plan calls must execute");

    let requests = request_log.requests();
    assert_eq!(
        requests.len(),
        2,
        "flags-off must not open a third (text-continuation) stream; got {}",
        requests.len()
    );
    assert!(
        requests.iter().all(|request| !contains_nudge(request)),
        "intentContinuation is off; CONTINUE_NUDGE must not appear"
    );
    assert!(
        requests
            .iter()
            .all(|request| !request_contains_notice(request)),
        "repetitionNotice is off; model-visible notice must not appear"
    );
    Ok(())
}

/// Duplicate same-ID apply_patch in one SSE body executes once in the temp
/// workdir. A later replay of that call_id must not rewrite the file.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scripted_stream_runs_apply_patch_once_and_suppresses_same_id_replay() -> Result<()> {
    skip_if_no_network!(Ok(()));
    Box::pin(scripted_stream_runs_apply_patch_once_and_suppresses_same_id_replay_inner()).await
}

async fn scripted_stream_runs_apply_patch_once_and_suppresses_same_id_replay_inner() -> Result<()> {
    let server = start_mock_server().await;
    let patch = "*** Begin Patch\n*** Add File: marker.txt\n+from-scripted-stream\n*** End Patch";
    let request_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-patch"),
                ev_apply_patch_custom_tool_call("patch-1", patch),
                ev_apply_patch_custom_tool_call("patch-1", patch),
                ev_completed("resp-patch"),
            ]),
            assistant_sse("resp-after-patch", "patch applied"),
            sse(vec![
                ev_response_created("resp-replay"),
                ev_apply_patch_custom_tool_call("patch-1", patch),
                ev_completed("resp-replay"),
            ]),
            assistant_sse("resp-after-replay", "replay suppressed"),
        ],
    )
    .await;

    let mut builder = enhanced_builder("E1");
    let test = Box::pin(builder.build(&server)).await?;
    let marker = test.cwd.path().join("marker.txt");

    submit_unrestricted(&test, "apply the marker patch").await?;
    wait_turn_complete(&test.codex).await;
    let first = fs::read_to_string(&marker).expect("apply_patch must create marker.txt");
    assert_eq!(first.trim(), "from-scripted-stream");

    submit_unrestricted(&test, "replay the same call id").await?;
    wait_turn_complete(&test.codex).await;
    let second = fs::read_to_string(&marker).expect("marker.txt must still exist");
    assert_eq!(
        second, first,
        "same-id replay must not re-apply the patch (duplicate side effect)"
    );

    let requests = request_log.requests();
    let outputs: usize = requests
        .iter()
        .map(|request| {
            request
                .input()
                .iter()
                .filter(|item| {
                    item.get("type").and_then(Value::as_str) == Some("custom_tool_call_output")
                        && item.get("call_id").and_then(Value::as_str) == Some("patch-1")
                })
                .count()
        })
        .sum();
    assert!(
        outputs >= 1,
        "the original apply_patch result must reach the next request; got {outputs} across {} requests",
        requests.len()
    );
    Ok(())
}

/// Interrupt while the model stream is outstanding must not start a
/// continuation (or any other) follow-up request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_during_stream_starts_no_new_request() -> Result<()> {
    skip_if_no_network!(Ok(()));
    Box::pin(cancel_during_stream_starts_no_new_request_inner()).await
}

async fn cancel_during_stream_starts_no_new_request_inner() -> Result<()> {
    let server = start_mock_server().await;
    let delayed = sse_response(assistant_sse("resp-slow", "I will continue immediately."))
        .set_delay(Duration::from_secs(60));
    let request_log = mount_response_sequence(&server, vec![delayed]).await;

    let mut builder = enhanced_builder("E5");
    let test = Box::pin(builder.build(&server)).await?;
    submit_text(&test, "start a turn we will cancel").await?;
    let deadline = tokio::time::Instant::now() + TURN_TIMEOUT;
    loop {
        if !request_log.requests().is_empty() {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("timed out waiting for the first sampling request");
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let in_flight = request_log.requests().len();
    test.codex.submit(Op::Interrupt).await?;
    wait_for_event_with_timeout(
        &test.codex,
        |event| matches!(event, EventMsg::TurnAborted(_)),
        TURN_TIMEOUT,
    )
    .await;

    let requests = request_log.requests();
    assert_eq!(
        requests.len(),
        in_flight,
        "cancel must not start a new request after interrupt; before={in_flight} after={}",
        requests.len()
    );
    assert_eq!(
        in_flight, 1,
        "the cancelled turn should have exactly one outstanding sampling request"
    );
    assert!(
        requests.iter().all(|request| !contains_nudge(request)),
        "cancel must not emit a continuation nudge request"
    );
    Ok(())
}
