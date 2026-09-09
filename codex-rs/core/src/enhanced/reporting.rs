use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
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
const ENHANCED_COMMIT_ENV: &str = "VELLUM_ENHANCED_COMMIT";
const EVENT_LOG_ENV: &str = "VELLUM_ENHANCED_EVENT_LOG";
const CHANNEL_CAPACITY: usize = 256;

static REPORTS: OnceLock<broadcast::Sender<Value>> = OnceLock::new();
static EVENT_LOG: OnceLock<Option<Mutex<PathBuf>>> = OnceLock::new();
static IDENTITY_LOGGED: OnceLock<()> = OnceLock::new();

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
        if IDENTITY_LOGGED.set(()).is_ok() {
            if let Some(identity) = identity_from_environment() {
                append_jsonl(&identity);
            }
        }
        let notification = event_notification(event);
        append_jsonl(&notification);
        let _ = reports().send(notification);
    }
}

/// The evaluator's one-shot `codex exec` client cannot receive App Server
/// notifications. When it supplies an explicit path, persist the exact same
/// already-redacted notification values so a case can prove which profile and
/// module branch actually ran. Production launches do not set this variable.
fn append_jsonl(value: &Value) {
    let log = EVENT_LOG.get_or_init(|| {
        std::env::var_os(EVENT_LOG_ENV)
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .map(Mutex::new)
    });
    let Some(path) = log else { return };
    let Ok(path) = path.lock() else { return };
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&*path) else {
        return;
    };
    if serde_json::to_writer(&mut file, value).is_ok() {
        let _ = file.write_all(b"\n");
        let _ = file.flush();
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
    std::env::var_os("CODEX_HOME")
        .map(|codex_home| load_features(std::path::Path::new(&codex_home)))
        .unwrap_or_else(EnhancedRuntimeFeatures::all_off)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_log_path_must_be_absolute() {
        assert!(!PathBuf::from("relative.jsonl").is_absolute());
        assert!(std::env::temp_dir().join("events.jsonl").is_absolute());
    }
}
