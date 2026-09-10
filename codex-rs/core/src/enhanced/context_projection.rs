use std::fs;
use std::path::Path;
use std::path::PathBuf;

use serde::Deserialize;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;

const STORE_DIR: &str = "enhanced-context-projections";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextProjectionRecord {
    pub call_id: String,
    pub original_sha256: String,
    pub projected_text: String,
    pub generation: u64,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionApplyStats {
    pub applied: u64,
    pub chars_removed: u64,
    pub restored: bool,
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DurableProjectionStore {
    #[serde(default = "schema_version")]
    schema_version: u32,
    #[serde(default)]
    generation: u64,
    #[serde(default)]
    records: Vec<ContextProjectionRecord>,
}

fn schema_version() -> u32 {
    1
}

#[derive(Debug, Default)]
pub struct ContextProjectionStore {
    path: Option<PathBuf>,
    generation: u64,
    records: Vec<ContextProjectionRecord>,
    restored_from_disk: bool,
    restoration_reported: bool,
}

impl ContextProjectionStore {
    pub fn load(codex_home: &Path, thread_id: &str) -> Self {
        let path = codex_home.join(STORE_DIR).join(format!("{thread_id}.json"));
        let durable = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<DurableProjectionStore>(&bytes).ok())
            .filter(|store| store.schema_version == schema_version());
        match durable {
            Some(store) => Self {
                path: Some(path),
                generation: store.generation,
                restored_from_disk: !store.records.is_empty(),
                records: store.records,
                restoration_reported: false,
            },
            None => Self {
                path: Some(path),
                ..Self::default()
            },
        }
    }

    pub fn in_memory() -> Self {
        Self::default()
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn record(&mut self, call_id: &str, original_text: &str, projected_text: &str) -> bool {
        if original_text == projected_text {
            return false;
        }
        let original_sha256 = text_sha256(original_text);
        if let Some(existing) = self.records.iter_mut().find(|record| {
            record.call_id == call_id
                && record.original_sha256 == original_sha256
                && record.generation == self.generation
        }) {
            existing.projected_text = projected_text.to_string();
        } else if let Some(existing) = self.records.iter_mut().find(|record| {
            record.call_id == call_id
                && text_sha256(&record.projected_text) == original_sha256
                && record.generation == self.generation
        }) {
            // Collapse a second pruning pass into the original mapping. Prompt
            // assembly starts from canonical history, so retaining a chain
            // here would otherwise restore only the first, larger projection.
            existing.projected_text = projected_text.to_string();
        } else {
            self.records.push(ContextProjectionRecord {
                call_id: call_id.to_string(),
                original_sha256,
                projected_text: projected_text.to_string(),
                generation: self.generation,
            });
        }
        self.persist();
        true
    }

    pub fn replacement(&self, call_id: &str, current_text: &str) -> Option<&str> {
        let digest = text_sha256(current_text);
        self.records
            .iter()
            .rev()
            .find(|record| {
                record.generation == self.generation
                    && record.call_id == call_id
                    && record.original_sha256 == digest
            })
            .map(|record| record.projected_text.as_str())
    }

    pub fn note_application(&mut self, applied: u64, chars_removed: u64) -> ProjectionApplyStats {
        let restored = applied > 0 && self.restored_from_disk && !self.restoration_reported;
        if restored {
            self.restoration_reported = true;
        }
        ProjectionApplyStats {
            applied,
            chars_removed,
            restored,
        }
    }

    pub fn clear_after_compaction(&mut self) -> bool {
        let had_records = !self.records.is_empty();
        self.records.clear();
        self.generation = self.generation.saturating_add(1);
        self.restored_from_disk = false;
        self.restoration_reported = false;
        self.persist();
        had_records
    }

    fn persist(&self) {
        let Some(path) = &self.path else {
            return;
        };
        let Some(parent) = path.parent() else {
            return;
        };
        if fs::create_dir_all(parent).is_err() {
            return;
        }
        let durable = DurableProjectionStore {
            schema_version: schema_version(),
            generation: self.generation,
            records: self.records.clone(),
        };
        let Ok(bytes) = serde_json::to_vec(&durable) else {
            return;
        };
        let temporary = path.with_extension("json.tmp");
        if fs::write(&temporary, bytes).is_err() {
            return;
        }
        if fs::rename(&temporary, path).is_err() {
            let _ = fs::remove_file(path);
            let _ = fs::rename(temporary, path);
        }
    }
}

fn text_sha256(text: &str) -> String {
    format!("sha256:{}", hex::encode(Sha256::digest(text.as_bytes())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_requires_call_id_original_digest_and_generation() {
        let mut store = ContextProjectionStore::in_memory();
        assert!(store.record("c1", "large original", "small"));
        assert_eq!(store.replacement("c1", "large original"), Some("small"));
        assert_eq!(store.replacement("c2", "large original"), None);
        assert_eq!(store.replacement("c1", "different original"), None);
        assert!(store.record("c1", "small", "smaller"));
        assert_eq!(store.replacement("c1", "large original"), Some("smaller"));
        assert!(store.clear_after_compaction());
        assert_eq!(store.replacement("c1", "large original"), None);
    }

    #[test]
    fn records_restore_from_thread_sidecar() {
        let directory = tempfile::tempdir().unwrap();
        let mut first = ContextProjectionStore::load(directory.path(), "thread-1");
        first.record("c1", "large original", "small");
        let mut restored = ContextProjectionStore::load(directory.path(), "thread-1");
        assert_eq!(restored.replacement("c1", "large original"), Some("small"));
        let stats = restored.note_application(1, 10);
        assert!(stats.restored);
        assert!(!restored.note_application(1, 10).restored);
    }
}
