use std::collections::BTreeMap;

use serde::Deserialize;
use serde::Serialize;

use super::config::EnhancedRuntimeFeatures;
use super::digest::DigestError;
use super::digest::DigestInputs;
use super::digest::compute_runtime_digest;
use super::digest::is_pinned_git_sha;
use super::digest::is_sha256_digest;

pub const LOCKFILE_SCHEMA_VERSION: u32 = 2;
pub const BUILD_PROFILE_ENHANCED_MVP_V1: &str = "enhanced-mvp-v1";

/// Pinned source revisions for the Enhanced Codex MVP. These must not drift
/// with `main` / `master` / `latest`.
pub const CODEX_UPSTREAM_COMMIT: &str = "f70e26c29ccb731e22d1104de550b1b9594d7070";
pub const QWEN_CODE_SOURCE_COMMIT: &str = "2b8f73c1e9cf8b355ec46c4623398c27b458b076";
pub const DEEPSEEK_HARNESS_SOURCE_COMMIT: &str = "dd6322d604e00eec1ba5e0c8541159906a21094a";
/// Codex app-server schema pin shipped in this repository.
pub const APP_SERVER_PROTOCOL_HASH: &str =
    "sha256:0d00aee4fbb8c9de8634c05234acf535f9998ec7fa0b0ba07f37b748e31ee0b5";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnhancedRuntimeArtifact {
    pub artifact_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnhancedRuntimeLockFile {
    pub schema_version: u32,
    pub codex_upstream_commit: String,
    #[serde(default)]
    pub enhanced_codex_commit: Option<String>,
    pub qwen_code_source_commit: String,
    pub deepseek_harness_source_commit: String,
    pub app_server_protocol_hash: String,
    pub build_profile: String,
    #[serde(default)]
    pub artifact_sha256: Option<String>,
    #[serde(default)]
    pub target_triple: Option<String>,
    #[serde(default)]
    pub artifacts: BTreeMap<String, EnhancedRuntimeArtifact>,
}

impl EnhancedRuntimeLockFile {
    pub fn mvp_pins() -> Self {
        Self {
            schema_version: LOCKFILE_SCHEMA_VERSION,
            codex_upstream_commit: CODEX_UPSTREAM_COMMIT.into(),
            enhanced_codex_commit: None,
            qwen_code_source_commit: QWEN_CODE_SOURCE_COMMIT.into(),
            deepseek_harness_source_commit: DEEPSEEK_HARNESS_SOURCE_COMMIT.into(),
            app_server_protocol_hash: APP_SERVER_PROTOCOL_HASH.into(),
            build_profile: BUILD_PROFILE_ENHANCED_MVP_V1.into(),
            artifact_sha256: None,
            target_triple: None,
            artifacts: BTreeMap::new(),
        }
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, LockFileError> {
        let value = serde_json::from_slice::<Self>(bytes)?;
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), LockFileError> {
        if !matches!(self.schema_version, 1 | LOCKFILE_SCHEMA_VERSION) {
            return Err(LockFileError::UnsupportedSchema(self.schema_version));
        }
        for (name, value) in [
            ("codexUpstreamCommit", self.codex_upstream_commit.as_str()),
            (
                "qwenCodeSourceCommit",
                self.qwen_code_source_commit.as_str(),
            ),
            (
                "deepseekHarnessSourceCommit",
                self.deepseek_harness_source_commit.as_str(),
            ),
        ] {
            require_git_sha(name, value)?;
        }
        if let Some(commit) = self.enhanced_codex_commit.as_deref()
            && !commit.trim().is_empty()
        {
            require_git_sha("enhancedCodexCommit", commit)?;
        }
        if self.app_server_protocol_hash.trim().is_empty() {
            return Err(LockFileError::EmptyField("appServerProtocolHash"));
        }
        if self.build_profile.trim().is_empty()
            || matches!(self.build_profile.as_str(), "main" | "master" | "latest")
        {
            return Err(LockFileError::FloatingRevision {
                field: "buildProfile",
                value: self.build_profile.clone(),
            });
        }
        if let Some(artifact) = self.artifact_sha256.as_deref()
            && !artifact.trim().is_empty()
            && !is_sha256_digest(artifact)
        {
            return Err(LockFileError::InvalidDigest {
                field: "artifactSha256",
                value: artifact.to_string(),
            });
        }
        for artifact in self.artifacts.values() {
            if !is_sha256_digest(&artifact.artifact_sha256) {
                return Err(LockFileError::InvalidDigest {
                    field: "artifacts.*.artifactSha256",
                    value: artifact.artifact_sha256.clone(),
                });
            }
        }
        Ok(())
    }

    pub fn artifact_for_target(&self, target_triple: &str) -> Option<&str> {
        if let Some(artifact) = self.artifacts.get(target_triple) {
            return Some(&artifact.artifact_sha256);
        }
        if self.artifacts.is_empty() && self.target_triple.as_deref() == Some(target_triple) {
            return self.artifact_sha256.as_deref();
        }
        None
    }

    pub fn identity_complete_for_target(&self, target_triple: &str) -> bool {
        matches!(
            self.enhanced_codex_commit.as_deref(),
            Some(value) if is_pinned_git_sha(value)
        ) && matches!(
            self.artifact_for_target(target_triple),
            Some(value) if is_sha256_digest(value)
        )
    }

    pub fn identity_complete(&self) -> bool {
        matches!(
            self.enhanced_codex_commit.as_deref(),
            Some(value) if is_pinned_git_sha(value)
        ) && matches!(
            self.artifact_sha256.as_deref(),
            Some(value) if is_sha256_digest(value)
        ) && matches!(
            self.target_triple.as_deref(),
            Some(value) if !value.trim().is_empty()
        )
    }

    pub fn digest_inputs(
        &self,
        feature_defaults: EnhancedRuntimeFeatures,
        target_triple: impl Into<String>,
    ) -> Result<DigestInputs, DigestError> {
        let enhanced = self
            .enhanced_codex_commit
            .as_deref()
            .filter(|value| !value.is_empty())
            .ok_or(DigestError::Incomplete("enhancedCodexCommit"))?;
        let target_triple = target_triple.into();
        let artifact = self
            .artifact_for_target(&target_triple)
            .filter(|value| !value.is_empty())
            .ok_or(DigestError::Incomplete("artifactSha256"))?;
        let inputs = DigestInputs {
            codex_upstream_commit: self.codex_upstream_commit.clone(),
            enhanced_codex_commit: enhanced.to_string(),
            qwen_source_commit: self.qwen_code_source_commit.clone(),
            deepseek_source_commit: self.deepseek_harness_source_commit.clone(),
            feature_defaults,
            app_server_protocol_hash: self.app_server_protocol_hash.clone(),
            build_profile: self.build_profile.clone(),
            target_triple,
            artifact_sha256: artifact.to_string(),
        };
        inputs.validate()?;
        Ok(inputs)
    }

    pub fn runtime_digest(
        &self,
        feature_defaults: EnhancedRuntimeFeatures,
        target_triple: impl Into<String>,
    ) -> Result<String, DigestError> {
        compute_runtime_digest(&self.digest_inputs(feature_defaults, target_triple)?)
    }
}

fn require_git_sha(field: &'static str, value: &str) -> Result<(), LockFileError> {
    if value.trim().is_empty() {
        return Err(LockFileError::EmptyField(field));
    }
    if matches!(value, "main" | "master" | "latest")
        || value.starts_with("unreleased")
        || !is_pinned_git_sha(value)
    {
        return Err(LockFileError::FloatingRevision {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LockFileError {
    #[error("lock file JSON is invalid: {0}")]
    Json(String),
    #[error("unsupported lock file schema version: {0}")]
    UnsupportedSchema(u32),
    #[error("lock file field {0} is empty")]
    EmptyField(&'static str),
    #[error("lock file field {field} must be a pinned git SHA, not {value}")]
    FloatingRevision { field: &'static str, value: String },
    #[error("lock file field {field} must be a sha256 digest, not {value}")]
    InvalidDigest { field: &'static str, value: String },
}

impl From<serde_json::Error> for LockFileError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_floating_and_unreleased_revisions() {
        let mut lock = EnhancedRuntimeLockFile::mvp_pins();
        lock.codex_upstream_commit = "main".into();
        assert!(matches!(
            lock.validate(),
            Err(LockFileError::FloatingRevision { .. })
        ));
        let mut lock = EnhancedRuntimeLockFile::mvp_pins();
        lock.enhanced_codex_commit = Some("unreleased-vellum-enhanced-codex-mvp".into());
        assert!(matches!(
            lock.validate(),
            Err(LockFileError::FloatingRevision { .. })
        ));
    }

    #[test]
    fn incomplete_identity_cannot_form_a_digest() {
        let lock = EnhancedRuntimeLockFile::mvp_pins();
        assert!(!lock.identity_complete());
        assert!(
            lock.runtime_digest(EnhancedRuntimeFeatures::all_on(), "x86_64-pc-windows-msvc")
                .is_err()
        );
    }

    #[test]
    fn resolves_artifact_by_target() {
        let mut lock = EnhancedRuntimeLockFile::mvp_pins();
        lock.enhanced_codex_commit = Some("b".repeat(40));
        lock.artifacts.insert(
            "aarch64-apple-darwin".into(),
            EnhancedRuntimeArtifact {
                artifact_sha256:
                    "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
                        .into(),
            },
        );

        assert!(lock.identity_complete_for_target("aarch64-apple-darwin"));
        assert!(!lock.identity_complete_for_target("x86_64-pc-windows-msvc"));
        assert!(
            lock.runtime_digest(EnhancedRuntimeFeatures::all_on(), "aarch64-apple-darwin")
                .is_ok()
        );
    }
}
