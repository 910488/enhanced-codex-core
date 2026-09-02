use std::sync::OnceLock;

use serde_json::Value;
use tokio::sync::broadcast;

use super::config::AblationProfile;
use super::config::EnhancedRuntimeFeatures;
use super::notifications::event_notification;
use super::notifications::identity_notification;
use super::runtime::load_features;
use super::telemetry::EnhancedEvent;

const PLANE_ENV: &str = "VELLUM_EXECUTION_PLANE";
const ENHANCED_PLANE: &str = "enhanced-codex";
const DIGEST_ENV: &str = "VELLUM_RUNTIME_DIGEST";
const FEATURE_PROFILE_ENV: &str = "VELLUM_ENHANCED_FEATURE_PROFILE";
const ENHANCED_COMMIT_ENV: &str = "VELLUM_ENHANCED_COMMIT";
const CHANNEL_CAPACITY: usize = 256;

static REPORTS: OnceLock<broadcast::Sender<Value>> = OnceLock::new();

fn reports() -> &'static broadcast::Sender<Value> {
    REPORTS.get_or_init(|| broadcast::channel(CHANNEL_CAPACITY).0)
}

pub fn is_enhanced_process() -> bool {
    std::env::var(PLANE_ENV).as_deref() == Ok(ENHANCED_PLANE)
}

pub fn subscribe() -> Option<broadcast::Receiver<Value>> {
    is_enhanced_process().then(|| reports().subscribe())
}

pub fn publish_event(event: &EnhancedEvent) {
    if is_enhanced_process() {
        let _ = reports().send(event_notification(event));
    }
}

pub fn identity_from_environment() -> Option<Value> {
    if !is_enhanced_process() {
        return None;
    }
    let digest = std::env::var(DIGEST_ENV).ok()?.trim().to_string();
    let commit = std::env::var(ENHANCED_COMMIT_ENV).ok()?.trim().to_string();
    if digest.is_empty() || commit.is_empty() {
        return None;
    }
    let features = configured_features();
    Some(identity_notification(
        commit,
        digest,
        profile_label(features),
        features,
    ))
}

fn configured_features() -> EnhancedRuntimeFeatures {
    std::env::var(FEATURE_PROFILE_ENV)
        .ok()
        .and_then(|value| serde_json::from_str(&value).ok())
        .unwrap_or_else(|| {
            std::env::var_os("CODEX_HOME")
                .map(|codex_home| load_features(std::path::Path::new(&codex_home)))
                .unwrap_or_else(EnhancedRuntimeFeatures::all_off)
        })
}

fn profile_label(features: EnhancedRuntimeFeatures) -> &'static str {
    [
        AblationProfile::E0,
        AblationProfile::E1,
        AblationProfile::E2,
        AblationProfile::E3,
        AblationProfile::E4,
        AblationProfile::E5,
    ]
    .into_iter()
    .find(|profile| profile.features() == features)
    .map(AblationProfile::as_str)
    .unwrap_or("custom")
}
