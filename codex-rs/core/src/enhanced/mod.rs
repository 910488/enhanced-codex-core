//! Portable Enhanced Codex MVP modules.
//!
//! These modules are the source that should land in a pinned OpenAI Codex
//! fork at `codex-rs/core/src/enhanced/`. They must not depend on Vellum
//! proxy-runtime, Canonical compaction, Action Conversion, or any other
//! historical Vellum agent loop.
//!
//! Official Codex stays unmodified. Each port is independently gated.

#![allow(unused_imports, dead_code)]

pub mod bounded_continuation;
pub mod config;
pub mod context_pruner;
pub mod context_recovery;
pub mod digest;
pub mod gateway;
pub mod hooks;
pub mod lockfile;
pub mod telemetry;
pub mod tool_reliability;

pub use bounded_continuation::{
    commit_continuation, plan_turn_stop, release_continuation, AutoContinuationBudget,
    ContinuationDecision, ContinuationPlan, ReservedContinuation, TurnStopContext,
    UnfinishedSignal, CONTINUE_NUDGE, MAX_AUTO_CONTINUATIONS,
};
pub use config::{AblationProfile, EnhancedRuntimeFeatures};
pub use context_pruner::{
    apply_pressure_prune, ContentBlock, ModelVisibleSurface, PruneOutcome, SurfaceItem,
    ToolResultPrunePolicy,
};
pub use context_recovery::{
    plan_context_pressure, plan_overflow_retry, CompactDecision, OverflowAttempt, OverflowDecision,
    OverflowPlan, PressurePlan, MAX_CONTEXT_OVERFLOW_RETRIES,
};
pub use digest::{
    compute_runtime_digest, is_pinned_git_sha, is_sha256_digest, DigestError, DigestInputs,
};
pub use gateway::{
    ForbiddenGatewayOperation, GatewayIsolationError, GatewayMutationGuard, GatewayOperation,
};
pub use hooks::{EnhancedTurnHooks, HookDecision};
pub use lockfile::{
    EnhancedRuntimeLockFile, APP_SERVER_PROTOCOL_HASH, BUILD_PROFILE_ENHANCED_MVP_V1,
    CODEX_UPSTREAM_COMMIT, DEEPSEEK_HARNESS_SOURCE_COMMIT, QWEN_CODE_SOURCE_COMMIT,
};
pub use telemetry::{
    field_name_is_forbidden, hash_identifier, EnhancedEvent, EnhancedEventFields, EnhancedEventKind,
    MemoryTelemetry,
};
pub use tool_reliability::{
    AdmitDecision, LateResultDecision, ProviderToolCallIdentity, SyntheticResultKind,
    SyntheticToolResult, ToolCallLedger, ToolCallResolution,
};

pub mod runtime;
pub mod seams;
