use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::config::EnhancedRuntimeFeatures;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DigestInputs {
    pub codex_upstream_commit: String,
    pub enhanced_codex_commit: String,
    pub qwen_source_commit: String,
    pub deepseek_source_commit: String,
    pub feature_defaults: EnhancedRuntimeFeatures,
    pub build_profile: String,
    pub target_triple: String,
    pub artifact_sha256: String,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DigestError {
    #[error("runtime identity is incomplete: {0} is missing")]
    Incomplete(&'static str),
    #[error("runtime identity field {field} is not a pinned git SHA: {value}")]
    NotGitSha { field: &'static str, value: String },
    #[error("runtime identity field {field} is not a sha256 digest: {value}")]
    NotSha256 { field: &'static str, value: String },
}

pub fn is_pinned_git_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub fn is_sha256_digest(value: &str) -> bool {
    let value = value.strip_prefix("sha256:").unwrap_or(value);
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

impl DigestInputs {
    pub fn validate(&self) -> Result<(), DigestError> {
        for (field, value) in [
            ("codexUpstreamCommit", self.codex_upstream_commit.as_str()),
            ("enhancedCodexCommit", self.enhanced_codex_commit.as_str()),
            ("qwenCodeSourceCommit", self.qwen_source_commit.as_str()),
            (
                "deepseekHarnessSourceCommit",
                self.deepseek_source_commit.as_str(),
            ),
        ] {
            if value.trim().is_empty() {
                return Err(DigestError::Incomplete(field));
            }
            if !is_pinned_git_sha(value) {
                return Err(DigestError::NotGitSha {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.build_profile.trim().is_empty() {
            return Err(DigestError::Incomplete("buildProfile"));
        }
        if self.target_triple.trim().is_empty() {
            return Err(DigestError::Incomplete("targetTriple"));
        }
        if self.artifact_sha256.trim().is_empty() {
            return Err(DigestError::Incomplete("artifactSha256"));
        }
        if !is_sha256_digest(&self.artifact_sha256) {
            return Err(DigestError::NotSha256 {
                field: "artifactSha256",
                value: self.artifact_sha256.clone(),
            });
        }
        Ok(())
    }
}

/// Runtime digest is the immutable identity of an Enhanced Codex install.
/// A thread binds to this value at creation and must not migrate.
pub fn compute_runtime_digest(inputs: &DigestInputs) -> Result<String, DigestError> {
    inputs.validate()?;
    let mut hasher = Sha256::new();
    feed(
        &mut hasher,
        "codexUpstreamCommit",
        &inputs.codex_upstream_commit,
    );
    feed(
        &mut hasher,
        "enhancedCodexCommit",
        &inputs.enhanced_codex_commit,
    );
    feed(
        &mut hasher,
        "qwenCodeSourceCommit",
        &inputs.qwen_source_commit,
    );
    feed(
        &mut hasher,
        "deepseekHarnessSourceCommit",
        &inputs.deepseek_source_commit,
    );
    feed(
        &mut hasher,
        "qwenToolReliability",
        bool_flag(inputs.feature_defaults.qwen_tool_reliability),
    );
    feed(
        &mut hasher,
        "deepseekContextRecovery",
        bool_flag(inputs.feature_defaults.deepseek_context_recovery),
    );
    feed(
        &mut hasher,
        "qwenBoundedContinuation",
        bool_flag(inputs.feature_defaults.qwen_bounded_continuation),
    );
    feed(
        &mut hasher,
        "repetitionNotice",
        bool_flag(inputs.feature_defaults.repetition_notice),
    );
    feed(
        &mut hasher,
        "intentContinuation",
        bool_flag(inputs.feature_defaults.intent_continuation),
    );
    feed(&mut hasher, "buildProfile", &inputs.build_profile);
    feed(&mut hasher, "targetTriple", &inputs.target_triple);
    feed(&mut hasher, "artifactSha256", &inputs.artifact_sha256);
    Ok(format!("sha256:{}", hex::encode(hasher.finalize())))
}

fn bool_flag(value: bool) -> &'static str {
    if value {
        "1"
    } else {
        "0"
    }
}

fn feed(hasher: &mut Sha256, key: &str, value: &str) {
    hasher.update(key.as_bytes());
    hasher.update([0]);
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
    hasher.update([0]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> DigestInputs {
        DigestInputs {
            codex_upstream_commit: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
            enhanced_codex_commit: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
            qwen_source_commit: "cccccccccccccccccccccccccccccccccccccccc".into(),
            deepseek_source_commit: "dddddddddddddddddddddddddddddddddddddddd".into(),
            feature_defaults: EnhancedRuntimeFeatures::all_on(),
            build_profile: "enhanced-mvp-v1".into(),
            target_triple: "x86_64-pc-windows-msvc".into(),
            artifact_sha256:
                "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".into(),
        }
    }

    #[test]
    fn digest_is_stable_and_source_sensitive() {
        let first = compute_runtime_digest(&sample()).unwrap();
        let second = compute_runtime_digest(&sample()).unwrap();
        assert_eq!(first, second);
        let mut changed = sample();
        changed.qwen_source_commit = "ccccccccccccccccccccccccccccccccccccccce".into();
        assert_ne!(first, compute_runtime_digest(&changed).unwrap());
        let mut artifact = sample();
        artifact.artifact_sha256 =
            "sha256:0000000000000000000000000000000000000000000000000000000000000001".into();
        assert_ne!(first, compute_runtime_digest(&artifact).unwrap());
        let mut notice = sample();
        notice.feature_defaults.repetition_notice = true;
        assert_ne!(first, compute_runtime_digest(&notice).unwrap());
        let mut intent = sample();
        intent.feature_defaults.intent_continuation = true;
        assert_ne!(first, compute_runtime_digest(&intent).unwrap());
    }

    #[test]
    fn rejects_unreleased_placeholder_commits() {
        let mut inputs = sample();
        inputs.enhanced_codex_commit = "unreleased-vellum-enhanced-codex-mvp".into();
        assert!(matches!(
            compute_runtime_digest(&inputs),
            Err(DigestError::NotGitSha { .. })
        ));
    }
}
