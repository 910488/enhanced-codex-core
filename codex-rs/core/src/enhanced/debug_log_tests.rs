use std::fs::read;
use std::io::Write as _;
use std::time::Instant;

use pretty_assertions::assert_eq;
use tempfile::TempDir;

use super::*;

fn config(dir: &TempDir, max_bytes: u64, retained_files: usize) -> DebugLogConfig {
    DebugLogConfig {
        path: dir.path().join("enhanced.jsonl"),
        max_bytes,
        retained_files,
        queue_capacity: 64,
        flush_interval: Duration::from_millis(10),
    }
}

#[test]
fn batches_valid_jsonl_and_rotates_with_a_hard_file_count_bound() {
    let dir = TempDir::new().unwrap();
    let config = config(&dir, 128, 3);
    let mut writer = RotatingJsonlWriter::open(&config).unwrap();
    for index in 0..20 {
        writer
            .write(&serde_json::json!({"sequence": index, "value": "x".repeat(32)}))
            .unwrap();
    }
    writer.flush().unwrap();

    let files = [
        config.path.clone(),
        rotated_path(&config.path, 1),
        rotated_path(&config.path, 2),
    ];
    assert_eq!(files.iter().filter(|path| path.exists()).count(), 3);
    assert!(!rotated_path(&config.path, 3).exists());
    for path in files {
        for line in String::from_utf8(read(path).unwrap()).unwrap().lines() {
            serde_json::from_str::<Value>(line).unwrap();
        }
    }
}

#[test]
fn publisher_never_waits_when_the_bounded_queue_is_full() {
    let (sender, receiver) = sync_channel(1);
    let sink = DebugLogSink {
        sender,
        dropped: Arc::new(AtomicU64::new(0)),
    };
    sink.publish(serde_json::json!({"first": true}));
    sink.publish(serde_json::json!({"second": true}));

    assert_eq!(sink.dropped.load(Ordering::Relaxed), 1);
    assert_eq!(receiver.try_iter().count(), 1);
}

#[test]
#[ignore = "storage performance probe"]
fn storage_performance_probe() {
    let dir = TempDir::new().unwrap();
    let config = config(&dir, 128 * 1024 * 1024, 2);
    let mut writer = RotatingJsonlWriter::open(&config).unwrap();
    let record = serde_json::json!({
        "timestamp": "2026-09-10T00:00:00.000Z",
        "sequence": 1,
        "processId": 1,
        "notification": {
            "method": "vellum/enhancedEvent",
            "params": {
                "name": "enhanced.context.pressure_checked",
                "fields": {"beforeTokenEstimate": 10000, "afterTokenEstimate": 1200}
            }
        }
    });
    let records = 100_000_u64;
    let started = Instant::now();
    for _ in 0..records {
        writer.write(&record).unwrap();
    }
    writer.flush().unwrap();
    let elapsed = started.elapsed();
    let bytes = std::fs::metadata(&config.path).unwrap().len();
    writeln!(
        std::io::stdout().lock(),
        "{{\"records\":{records},\"bytes\":{bytes},\"elapsedMs\":{},\"recordsPerSecond\":{:.0},\"mibPerSecond\":{:.2}}}",
        elapsed.as_millis(),
        records as f64 / elapsed.as_secs_f64(),
        bytes as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0,
    )
    .unwrap();
}
