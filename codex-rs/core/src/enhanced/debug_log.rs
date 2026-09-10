//! Buffered, bounded diagnostic logging for Enhanced runtime events.
//!
//! The evaluator evidence log remains synchronous and exact. This sink is for
//! longer real-machine observations, where logging must not block the agent
//! loop or grow without bound.

use std::fs::File;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::SyncSender;
use std::sync::mpsc::TrySendError;
use std::sync::mpsc::sync_channel;
use std::time::Duration;
use std::time::Instant;

use serde::Deserialize;
use serde_json::Value;

const DEBUG_LOG_ENV: &str = "VELLUM_ENHANCED_DEBUG_LOG";
const CONFIG_FILE: &str = "enhanced-runtime.json";
const DEFAULT_MAX_BYTES: u64 = 8 * 1024 * 1024;
const DEFAULT_RETAINED_FILES: usize = 4;
const DEFAULT_QUEUE_CAPACITY: usize = 2_048;
const DEFAULT_FLUSH_INTERVAL: Duration = Duration::from_millis(250);
const MAX_BATCH: usize = 256;

static DEBUG_LOG: OnceLock<Option<DebugLogSink>> = OnceLock::new();

#[derive(Debug, Clone)]
struct DebugLogConfig {
    path: PathBuf,
    max_bytes: u64,
    retained_files: usize,
    queue_capacity: usize,
    flush_interval: Duration,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeFile {
    debug_log: Option<RuntimeDebugLogConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeDebugLogConfig {
    #[serde(default)]
    enabled: bool,
    path: Option<PathBuf>,
    max_bytes: Option<u64>,
    retained_files: Option<usize>,
    queue_capacity: Option<usize>,
    flush_interval_ms: Option<u64>,
}

impl DebugLogConfig {
    fn load() -> Option<Self> {
        let codex_home = std::env::var_os("CODEX_HOME").map(PathBuf::from);
        let file = codex_home
            .as_deref()
            .and_then(|home| std::fs::read(home.join(CONFIG_FILE)).ok())
            .and_then(|bytes| serde_json::from_slice::<RuntimeFile>(&bytes).ok())
            .and_then(|file| file.debug_log)
            .unwrap_or_default();
        let env_path = std::env::var_os(DEBUG_LOG_ENV).map(PathBuf::from);
        if env_path.is_none() && !file.enabled {
            return None;
        }
        let path = env_path.or(file.path).or_else(|| {
            codex_home
                .as_ref()
                .map(|home| home.join("log").join("enhanced-events.jsonl"))
        })?;
        let path = if path.is_absolute() {
            path
        } else {
            codex_home?.join(path)
        };
        Some(Self {
            path,
            max_bytes: file
                .max_bytes
                .unwrap_or(DEFAULT_MAX_BYTES)
                .clamp(64 * 1024, 1024 * 1024 * 1024),
            retained_files: file
                .retained_files
                .unwrap_or(DEFAULT_RETAINED_FILES)
                .clamp(1, 16),
            queue_capacity: file
                .queue_capacity
                .unwrap_or(DEFAULT_QUEUE_CAPACITY)
                .clamp(64, 65_536),
            flush_interval: Duration::from_millis(
                file.flush_interval_ms
                    .unwrap_or(DEFAULT_FLUSH_INTERVAL.as_millis() as u64)
                    .clamp(10, 5_000),
            ),
        })
    }
}

struct DebugLogSink {
    sender: SyncSender<Value>,
    dropped: Arc<AtomicU64>,
}

impl DebugLogSink {
    fn start(config: DebugLogConfig) -> Option<Self> {
        let (sender, receiver) = sync_channel(config.queue_capacity);
        let dropped = Arc::new(AtomicU64::new(0));
        let worker_dropped = Arc::clone(&dropped);
        std::thread::Builder::new()
            .name("enhanced-debug-log".into())
            .spawn(move || run_worker(config, receiver, worker_dropped))
            .ok()?;
        Some(Self { sender, dropped })
    }

    fn publish(&self, value: Value) {
        if let Err(error) = self.sender.try_send(value)
            && matches!(error, TrySendError::Full(_)) {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
    }
}

pub(super) fn publish(value: Value) {
    if let Some(sink) =
        DEBUG_LOG.get_or_init(|| DebugLogConfig::load().and_then(DebugLogSink::start))
    {
        sink.publish(value);
    }
}

fn run_worker(config: DebugLogConfig, receiver: Receiver<Value>, dropped: Arc<AtomicU64>) {
    let Some(parent) = config.path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let Ok(mut writer) = RotatingJsonlWriter::open(&config) else {
        return;
    };
    let mut sequence = 0_u64;
    while let Ok(first) = receiver.recv() {
        let started = Instant::now();
        let mut batch = vec![first];
        while batch.len() < MAX_BATCH {
            let Some(remaining) = config.flush_interval.checked_sub(started.elapsed()) else {
                break;
            };
            match receiver.recv_timeout(remaining) {
                Ok(value) => batch.push(value),
                Err(_) => break,
            }
        }
        for notification in batch {
            sequence = sequence.saturating_add(1);
            let record = serde_json::json!({
                "timestamp": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                "sequence": sequence,
                "processId": std::process::id(),
                "notification": notification,
            });
            let _ = writer.write(&record);
        }
        let dropped_count = dropped.swap(0, Ordering::Relaxed);
        if dropped_count > 0 {
            sequence = sequence.saturating_add(1);
            let record = serde_json::json!({
                "timestamp": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                "sequence": sequence,
                "processId": std::process::id(),
                "sinkStatus": {"droppedEvents": dropped_count},
            });
            let _ = writer.write(&record);
        }
        let _ = writer.flush();
    }
    let _ = writer.flush();
}

struct RotatingJsonlWriter {
    config: DebugLogConfig,
    writer: Option<BufWriter<File>>,
    bytes_written: u64,
}

impl RotatingJsonlWriter {
    fn open(config: &DebugLogConfig) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.path)?;
        let bytes_written = file.metadata()?.len();
        Ok(Self {
            config: config.clone(),
            writer: Some(BufWriter::with_capacity(64 * 1024, file)),
            bytes_written,
        })
    }

    fn write(&mut self, value: &Value) -> std::io::Result<()> {
        let mut encoded = serde_json::to_vec(value)?;
        encoded.push(b'\n');
        if self.bytes_written > 0
            && self.bytes_written.saturating_add(encoded.len() as u64) > self.config.max_bytes
        {
            self.rotate()?;
        }
        let Some(writer) = self.writer.as_mut() else {
            return Err(std::io::Error::other(
                "enhanced debug log writer is closed",
            ));
        };
        writer.write_all(&encoded)?;
        self.bytes_written = self.bytes_written.saturating_add(encoded.len() as u64);
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let Some(writer) = self.writer.as_mut() else {
            return Err(std::io::Error::other(
                "enhanced debug log writer is closed",
            ));
        };
        writer.flush()
    }

    fn rotate(&mut self) -> std::io::Result<()> {
        if let Some(mut writer) = self.writer.take() {
            writer.flush()?;
        }
        if self.config.retained_files == 1 {
            self.writer = Some(BufWriter::with_capacity(
                64 * 1024,
                File::create(&self.config.path)?,
            ));
            self.bytes_written = 0;
            return Ok(());
        }
        let oldest = rotated_path(&self.config.path, self.config.retained_files - 1);
        let _ = std::fs::remove_file(oldest);
        for index in (1..self.config.retained_files - 1).rev() {
            let source = rotated_path(&self.config.path, index);
            if !source.exists() {
                continue;
            }
            let destination = rotated_path(&self.config.path, index + 1);
            let _ = std::fs::remove_file(&destination);
            std::fs::rename(source, destination)?;
        }
        let first = rotated_path(&self.config.path, 1);
        let _ = std::fs::remove_file(&first);
        std::fs::rename(&self.config.path, first)?;
        self.writer = Some(BufWriter::with_capacity(
            64 * 1024,
            File::create(&self.config.path)?,
        ));
        self.bytes_written = 0;
        Ok(())
    }
}

fn rotated_path(path: &Path, index: usize) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(format!(".{index}"));
    PathBuf::from(value)
}

#[cfg(test)]
#[path = "debug_log_tests.rs"]
mod tests;
