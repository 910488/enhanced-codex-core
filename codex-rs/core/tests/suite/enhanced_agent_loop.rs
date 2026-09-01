//! Deterministic agent-loop fixtures for Enhanced Codex ports A/B/C.
//!
//! These tests enter through `run_turn` and the real stream path
//! (`ToolRouter`, native compact, stop-hook continuation). They do not
//! call portable unit functions.

use std::time::Duration;

use std::sync::Arc;
use std::sync::Mutex;

use anyhow::Result;
use codex_core::TurnInputRequest;
use codex_core::compact::SUMMARIZATION_PROMPT;
use codex_features::Feature;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::config_types::AutoCompactTokenLimitScope;
use codex_protocol::error::CodexErr;
use codex_protocol::protocol::EventMsg;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ResponsesRequest;
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
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodex;
use core_test_support::test_codex::TestCodexBuilder;
use core_test_support::test_codex::test_codex;
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
