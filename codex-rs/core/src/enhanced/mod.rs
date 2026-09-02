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
pub mod notifications;
pub mod reporting;
pub mod telemetry;
pub mod tool_reliability;

pub use bounded_continuation::AutoContinuationBudget;
pub use bounded_continuation::CONTINUE_NUDGE;
pub use bounded_continuation::ContinuationDecision;
pub use bounded_continuation::ContinuationPlan;
pub use bounded_continuation::MAX_AUTO_CONTINUATIONS;
pub use bounded_continuation::ReservedContinuation;
pub use bounded_continuation::TurnStopContext;
pub use bounded_continuation::UnfinishedSignal;
pub use bounded_continuation::commit_continuation;
pub use bounded_continuation::plan_turn_stop;
pub use bounded_continuation::release_continuation;
pub use config::AblationProfile;
pub use config::EnhancedRuntimeFeatures;
pub use context_pruner::ContentBlock;
pub use context_pruner::ModelVisibleSurface;
pub use context_pruner::PruneOutcome;
pub use context_pruner::SurfaceItem;
pub use context_pruner::ToolResultPrunePolicy;
pub use context_pruner::apply_pressure_prune;
pub use context_recovery::CompactDecision;
pub use context_recovery::MAX_CONTEXT_OVERFLOW_RETRIES;
pub use context_recovery::OverflowAttempt;
pub use context_recovery::OverflowDecision;
pub use context_recovery::OverflowPlan;
pub use context_recovery::PressurePlan;
pub use context_recovery::plan_context_pressure;
pub use context_recovery::plan_overflow_retry;
pub use digest::DigestError;
pub use digest::DigestInputs;
pub use digest::compute_runtime_digest;
pub use digest::is_pinned_git_sha;
pub use digest::is_sha256_digest;
pub use gateway::ForbiddenGatewayOperation;
pub use gateway::GatewayIsolationError;
pub use gateway::GatewayMutationGuard;
pub use gateway::GatewayOperation;
pub use hooks::EnhancedTurnHooks;
pub use hooks::HookDecision;
pub use lockfile::APP_SERVER_PROTOCOL_HASH;
pub use lockfile::BUILD_PROFILE_ENHANCED_MVP_V1;
pub use lockfile::CODEX_UPSTREAM_COMMIT;
pub use lockfile::DEEPSEEK_HARNESS_SOURCE_COMMIT;
pub use lockfile::EnhancedRuntimeLockFile;
pub use lockfile::QWEN_CODE_SOURCE_COMMIT;
pub use notifications::event_notification;
pub use notifications::identity_notification;
pub use reporting::identity_from_environment;
pub use reporting::publish_event;
pub use reporting::subscribe;
pub use telemetry::EnhancedEvent;
pub use telemetry::EnhancedEventFields;
pub use telemetry::EnhancedEventKind;
pub use telemetry::MemoryTelemetry;
pub use telemetry::field_name_is_forbidden;
pub use telemetry::hash_identifier;
pub use tool_reliability::AdmitDecision;
pub use tool_reliability::LateResultDecision;
pub use tool_reliability::ProviderToolCallIdentity;
pub use tool_reliability::SyntheticResultKind;
pub use tool_reliability::SyntheticToolResult;
pub use tool_reliability::ToolCallLedger;
pub use tool_reliability::ToolCallResolution;

pub mod runtime;
pub mod seams;
