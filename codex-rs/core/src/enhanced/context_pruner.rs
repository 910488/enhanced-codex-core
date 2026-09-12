use serde::{Deserialize, Serialize};

use super::telemetry::{EnhancedEvent, EnhancedEventFields, EnhancedEventKind};

/// Shared experimental prune profile. V1 does not branch on model name.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultPrunePolicy {
    pub trigger_ratio: f64,
    pub min_text_bytes: usize,
    pub keep_head_bytes: usize,
    pub keep_tail_bytes: usize,
}

impl Default for ToolResultPrunePolicy {
    fn default() -> Self {
        Self {
            trigger_ratio: 0.85,
            min_text_bytes: 2_000,
            keep_head_bytes: 400,
            keep_tail_bytes: 800,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentBlock {
    pub kind: String,
    pub text: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SurfaceItem {
    User {
        text: String,
    },
    System {
        text: String,
    },
    Developer {
        text: String,
    },
    Assistant {
        text: String,
    },
    ToolCall {
        call_id: String,
        tool_name: String,
        tool_type: String,
        arguments: String,
    },
    ToolResult {
        call_id: String,
        tool_name: String,
        tool_type: String,
        blocks: Vec<ContentBlock>,
    },
    Control {
        key: String,
        value: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ModelVisibleSurface {
    pub items: Vec<SurfaceItem>,
    pub generation: u64,
    /// Token counts supplied by the host runtime, one per entry in `items`.
    ///
    /// The seam flattens rich items to plain text before this crate sees them,
    /// which loses the two things Codex measures and bytes cannot recover: the
    /// serialized JSON wrapper, and the per-modality substitutions for image,
    /// audio and encrypted payloads. Base64 is where that hurts — it costs
    /// about 1.6 bytes per token, not 4, so a base64 tool result measured by
    /// bytes reads 60% under.
    ///
    /// So the host passes its own number in and this crate uses it verbatim.
    /// A shorter vector, or `None` at an index, means "measure that one
    /// yourself" — which is what every entry falls back to once pruning has
    /// rewritten it, since the host's number describes the text that was there
    /// before.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub item_token_estimates: Vec<Option<u64>>,
}

impl ModelVisibleSurface {
    /// Codex's own rule, deliberately: `APPROX_BYTES_PER_TOKEN = 4` applied to
    /// UTF-8 bytes, rounded up (`codex-rs/utils/string/src/truncate.rs`).
    ///
    /// This used to count `chars`, which agrees with Codex only for ASCII. A
    /// CJK character is three UTF-8 bytes, so on Chinese or Japanese text the
    /// estimate came out three times low and the pruner believed it had room
    /// it did not have. Two estimators disagreeing about the same history is
    /// worse than either being imprecise, so this one follows Codex.
    ///
    /// It still under-counts relative to Codex: Codex measures the serialized
    /// JSON — field names, quoting, escaping — and substitutes per-modality
    /// estimates for image, audio and encrypted payloads. The seam flattens
    /// items to plain text before this crate sees them, so that part cannot be
    /// recovered here; it has to come from the seam.
    pub fn estimated_tokens(&self) -> u64 {
        // Per item, like Codex: it sums `estimate_item_token_count` over the
        // history rather than dividing one total, so the rounding matches too.
        self.items
            .iter()
            .enumerate()
            .map(|(index, item)| self.item_tokens(index, item))
            .fold(0u64, u64::saturating_add)
    }

    fn item_tokens(&self, index: usize, item: &SurfaceItem) -> u64 {
        self.item_token_estimates
            .get(index)
            .copied()
            .flatten()
            .unwrap_or_else(|| (item_bytes(item) as u64).div_ceil(4))
    }

    pub fn byte_count(&self) -> usize {
        self.items.iter().map(item_bytes).sum()
    }

    pub fn tool_pairs_intact(&self) -> bool {
        let calls = self
            .items
            .iter()
            .filter_map(|item| match item {
                SurfaceItem::ToolCall { call_id, .. } => Some(call_id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let results = self
            .items
            .iter()
            .filter_map(|item| match item {
                SurfaceItem::ToolResult { call_id, .. } => Some(call_id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        calls == results
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PruneOutcome {
    pub surface: ModelVisibleSurface,
    pub rewritten: bool,
    /// UTF-8 bytes, matching the estimator. The telemetry field it feeds is
    /// still named `charsRemoved` on the wire; renaming that is a separate
    /// change because the bridge allowlist and the fork both encode it.
    pub bytes_removed: u64,
    pub before_token_estimate: u64,
    pub after_token_estimate: u64,
    pub events: Vec<EnhancedEvent>,
}

pub fn apply_pressure_prune(
    surface: &ModelVisibleSurface,
    policy: ToolResultPrunePolicy,
    context_window_tokens: u64,
) -> PruneOutcome {
    let before = surface.estimated_tokens();
    let trigger = ((context_window_tokens as f64) * policy.trigger_ratio).floor() as u64;
    if before < trigger {
        return PruneOutcome {
            surface: surface.clone(),
            rewritten: false,
            bytes_removed: 0,
            before_token_estimate: before,
            after_token_estimate: before,
            events: Vec::new(),
        };
    }

    let mut events = vec![EnhancedEvent::new(
        EnhancedEventKind::ContextPruneStarted,
        EnhancedEventFields {
            before_token_estimate: Some(before),
            ..EnhancedEventFields::default()
        },
    )];
    let mut next = surface.clone();
    // A host estimate describes the text as it arrived. Once this rewrites an
    // item, that number is about bytes which are no longer there, so it is
    // dropped and the item is measured directly from what is left.
    let mut rewritten_items = Vec::new();
    for (index, item) in next.items.iter_mut().enumerate() {
        if let SurfaceItem::ToolResult { blocks, .. } = item {
            if prune_blocks(blocks, policy) > 0 {
                rewritten_items.push(index);
            }
        }
    }
    for index in rewritten_items {
        if let Some(slot) = next.item_token_estimates.get_mut(index) {
            *slot = None;
        }
    }
    let bytes_removed = surface.byte_count().saturating_sub(next.byte_count()) as u64;
    next.generation = surface
        .generation
        .saturating_add(if bytes_removed > 0 { 1 } else { 0 });
    let after = next.estimated_tokens();
    events.push(EnhancedEvent::new(
        EnhancedEventKind::ContextPruneCompleted,
        EnhancedEventFields {
            before_token_estimate: Some(before),
            after_token_estimate: Some(after),
            chars_removed: Some(bytes_removed),
            ..EnhancedEventFields::default()
        },
    ));
    PruneOutcome {
        rewritten: bytes_removed > 0,
        bytes_removed,
        before_token_estimate: before,
        after_token_estimate: after,
        surface: next,
        events,
    }
}

fn prune_blocks(blocks: &mut [ContentBlock], policy: ToolResultPrunePolicy) -> u64 {
    // DeepSeek walks one global cursor across every text block in the tool
    // result, emitting a single omitted marker; non-text blocks stay in order.
    // The cursor counts UTF-8 bytes rather than code points so the budget is
    // the same unit Codex estimates tokens in — but whole characters are still
    // emitted, so a multi-byte character straddling a boundary is kept or
    // dropped entire and the text stays valid UTF-8.
    let total_bytes: usize = blocks
        .iter()
        .map(|block| block.text.as_deref().map(str::len).unwrap_or(0))
        .sum();
    if total_bytes < policy.min_text_bytes {
        return 0;
    }
    let keep = policy
        .keep_head_bytes
        .saturating_add(policy.keep_tail_bytes);
    if total_bytes <= keep {
        return 0;
    }
    let head_end = policy.keep_head_bytes;
    let tail_start = total_bytes.saturating_sub(policy.keep_tail_bytes);
    let omitted = tail_start.saturating_sub(head_end);
    let marker = format!(
        "
[... omitted {omitted} bytes ...]
"
    );
    let mut cursor = 0usize;
    let mut marker_inserted = false;
    let mut kept_bytes = 0usize;
    for block in blocks.iter_mut() {
        let Some(text) = block.text.take() else {
            continue;
        };
        let mut kept = String::new();
        for character in text.chars() {
            let width = character.len_utf8();
            if cursor < head_end || cursor >= tail_start {
                kept.push(character);
                kept_bytes += width;
            } else if !marker_inserted {
                kept.push_str(&marker);
                marker_inserted = true;
            }
            cursor += width;
        }
        block.text = Some(kept);
    }
    total_bytes.saturating_sub(kept_bytes) as u64
}

fn item_bytes(item: &SurfaceItem) -> usize {
    // Every term is UTF-8 bytes. The previous version mixed `.len()` for
    // identifiers with `chars().count()` for payload text, which agreed with
    // itself only on ASCII.
    match item {
        SurfaceItem::User { text }
        | SurfaceItem::System { text }
        | SurfaceItem::Developer { text }
        | SurfaceItem::Assistant { text } => text.len(),
        SurfaceItem::ToolCall {
            call_id,
            tool_name,
            tool_type,
            arguments,
        } => call_id.len() + tool_name.len() + tool_type.len() + arguments.len(),
        SurfaceItem::ToolResult {
            call_id,
            tool_name,
            tool_type,
            blocks,
        } => {
            call_id.len()
                + tool_name.len()
                + tool_type.len()
                + blocks
                    .iter()
                    .map(|block| block.text.as_deref().map(str::len).unwrap_or(0))
                    .sum::<usize>()
        }
        SurfaceItem::Control { key, value } => key.len() + value.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Measured against the real endpoint on 2026-09-03: 6,400 bytes of base64
    /// cost 3,999 tokens, where bytes/4 predicts 1,600 — 60% under. The host
    /// knows the real number because it still has the unflattened item, so it
    /// passes it in and this crate uses it rather than its own guess.
    #[test]
    fn a_host_supplied_estimate_is_used_instead_of_the_byte_guess() {
        let base64ish = "A".repeat(6_400);
        let surface = ModelVisibleSurface {
            items: vec![SurfaceItem::ToolResult {
                call_id: "c1".into(),
                tool_name: "read".into(),
                tool_type: "function".into(),
                blocks: vec![ContentBlock {
                    kind: "text".into(),
                    text: Some(base64ish),
                }],
            }],
            generation: 0,
            item_token_estimates: vec![Some(3_999)],
        };
        assert_eq!(surface.estimated_tokens(), 3_999);

        let guessed = ModelVisibleSurface {
            item_token_estimates: Vec::new(),
            ..surface.clone()
        };
        assert!(
            guessed.estimated_tokens() < 1_700,
            "without the host estimate this reads about 1,600"
        );
    }

    /// The host's number describes the text as it arrived. Pruning rewrites it,
    /// so keeping the old number would report bytes that are no longer there —
    /// and the overflow retry decides on exactly that comparison.
    #[test]
    fn pruning_drops_the_host_estimate_for_the_item_it_rewrote() {
        let surface = ModelVisibleSurface {
            items: vec![SurfaceItem::ToolResult {
                call_id: "c1".into(),
                tool_name: "read".into(),
                tool_type: "function".into(),
                blocks: vec![ContentBlock {
                    kind: "text".into(),
                    text: Some("x".repeat(40_000)),
                }],
            }],
            generation: 0,
            item_token_estimates: vec![Some(99_999)],
        };
        let outcome = apply_pressure_prune(&surface, ToolResultPrunePolicy::default(), 1_000);
        assert!(outcome.rewritten);
        assert_eq!(
            outcome.surface.item_token_estimates,
            vec![None],
            "the stale estimate is cleared"
        );
        assert!(
            outcome.surface.estimated_tokens() < 1_000,
            "the pruned item is now measured from what is left"
        );
    }

    /// An untouched item keeps its host estimate even when a sibling is pruned.
    #[test]
    fn only_the_rewritten_item_loses_its_estimate() {
        let surface = ModelVisibleSurface {
            items: vec![
                SurfaceItem::User {
                    text: "keep me".into(),
                },
                SurfaceItem::ToolResult {
                    call_id: "c1".into(),
                    tool_name: "read".into(),
                    tool_type: "function".into(),
                    blocks: vec![ContentBlock {
                        kind: "text".into(),
                        text: Some("x".repeat(40_000)),
                    }],
                },
            ],
            generation: 0,
            item_token_estimates: vec![Some(7), Some(99_999)],
        };
        let outcome = apply_pressure_prune(&surface, ToolResultPrunePolicy::default(), 1_000);
        assert_eq!(outcome.surface.item_token_estimates, vec![Some(7), None]);
    }

    /// Codex estimates tokens as UTF-8 bytes over `APPROX_BYTES_PER_TOKEN = 4`
    /// (`codex-rs/utils/string/src/truncate.rs`). This crate has to agree, or
    /// the two of them make different decisions about the same history.
    ///
    /// The counter-example is any non-ASCII text: a CJK character is three
    /// bytes, so the old `chars().count()` reported a third of Codex's number
    /// and the pruner believed it had room it did not have.
    #[test]
    fn the_token_estimate_follows_codex_bytes_per_token_on_cjk() {
        let text = "重寫本上的字要先刮掉才寫得下".repeat(40);
        let surface = ModelVisibleSurface {
            item_token_estimates: Vec::new(),
            generation: 0,
            items: vec![SurfaceItem::User { text: text.clone() }],
        };
        assert_eq!(surface.byte_count(), text.len());
        assert_eq!(
            surface.estimated_tokens(),
            (text.len() as u64).div_ceil(4),
            "the estimate must be Codex's bytes/4"
        );
        let as_chars = (text.chars().count() as u64).div_ceil(4);
        assert_eq!(
            surface.estimated_tokens(),
            as_chars * 3,
            "three bytes per CJK character is exactly the error this fixes"
        );
    }

    /// The budget is counted in bytes but the text is still cut on character
    /// boundaries, so a pruned multi-byte result stays valid UTF-8 and no
    /// character is half-kept.
    #[test]
    fn pruning_multibyte_text_keeps_whole_characters() {
        let text = "犢皮紙".repeat(400); // 3 chars * 3 bytes * 400 = 3600 bytes
        let mut blocks = vec![ContentBlock {
            kind: "text".into(),
            text: Some(text.clone()),
        }];
        let removed = prune_blocks(&mut blocks, ToolResultPrunePolicy::default());
        assert!(removed > 0, "3600 bytes is past the 2000-byte trigger");
        let kept = blocks[0].text.clone().expect("text survives");
        assert!(
            kept.contains("[... omitted"),
            "one marker replaces the middle"
        );
        assert!(
            !kept.contains(char::REPLACEMENT_CHARACTER),
            "a byte-sliced multi-byte character would decode as U+FFFD"
        );
        assert!(kept.starts_with('犢'), "the head is kept whole");
    }

    fn huge_result(chars: usize) -> ModelVisibleSurface {
        ModelVisibleSurface {
            item_token_estimates: Vec::new(),
            generation: 3,
            items: vec![
                SurfaceItem::User {
                    text: "please inspect the file".into(),
                },
                SurfaceItem::ToolCall {
                    call_id: "c1".into(),
                    tool_name: "read".into(),
                    tool_type: "function".into(),
                    arguments: "{\"path\":\"a\"}".into(),
                },
                SurfaceItem::ToolResult {
                    call_id: "c1".into(),
                    tool_name: "read".into(),
                    tool_type: "function".into(),
                    blocks: vec![ContentBlock {
                        kind: "text".into(),
                        text: Some("😀".repeat(chars / 2) + &"b".repeat(chars / 2)),
                    }],
                },
                SurfaceItem::Control {
                    key: "turn".into(),
                    value: "1".into(),
                },
            ],
        }
    }

    #[test]
    fn pressure_below_threshold_does_not_rewrite() {
        let surface = huge_result(40);
        let outcome = apply_pressure_prune(&surface, ToolResultPrunePolicy::default(), 200_000);
        assert!(!outcome.rewritten);
        assert_eq!(outcome.surface, surface);
    }

    #[test]
    fn tool_pairing_survives_pruning() {
        let policy = ToolResultPrunePolicy {
            trigger_ratio: 0.01,
            min_text_bytes: 20,
            keep_head_bytes: 4,
            keep_tail_bytes: 4,
        };
        let surface = huge_result(80);
        let outcome = apply_pressure_prune(&surface, policy, 10);
        assert!(outcome.rewritten);
        assert!(outcome.surface.tool_pairs_intact());
        match &outcome.surface.items[2] {
            SurfaceItem::ToolResult {
                call_id,
                tool_name,
                tool_type,
                blocks,
            } => {
                assert_eq!(call_id, "c1");
                assert_eq!(tool_name, "read");
                assert_eq!(tool_type, "function");
                assert_eq!(blocks[0].kind, "text");
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(outcome.surface.items[0], SurfaceItem::User { .. }));
        assert!(matches!(
            outcome.surface.items[3],
            SurfaceItem::Control { .. }
        ));
    }

    #[test]
    fn multi_block_tool_result_is_pruned_as_one_surface() {
        let policy = ToolResultPrunePolicy {
            trigger_ratio: 0.01,
            min_text_bytes: 20,
            keep_head_bytes: 4,
            keep_tail_bytes: 4,
        };
        let surface = ModelVisibleSurface {
            item_token_estimates: Vec::new(),
            generation: 1,
            items: vec![SurfaceItem::ToolResult {
                call_id: "c1".into(),
                tool_name: "read".into(),
                tool_type: "function".into(),
                blocks: vec![
                    ContentBlock {
                        kind: "text".into(),
                        text: Some("AAAA".repeat(10)),
                    },
                    ContentBlock {
                        kind: "image".into(),
                        text: None,
                    },
                    ContentBlock {
                        kind: "text".into(),
                        text: Some("BBBB".repeat(10)),
                    },
                ],
            }],
        };
        let outcome = apply_pressure_prune(&surface, policy, 10);
        let SurfaceItem::ToolResult { blocks, .. } = &outcome.surface.items[0] else {
            panic!("expected tool result");
        };
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[1].kind, "image");
        assert!(blocks[1].text.is_none());
        let text: String = blocks
            .iter()
            .filter_map(|block| block.text.as_deref())
            .collect();
        assert_eq!(text.matches("omitted").count(), 1);
        assert!(text.starts_with("AAAA"));
        assert!(text.ends_with("BBBB"));
    }

    #[test]
    fn utf8_pruning_is_safe() {
        let policy = ToolResultPrunePolicy {
            trigger_ratio: 0.01,
            min_text_bytes: 8,
            keep_head_bytes: 3,
            keep_tail_bytes: 3,
        };
        let surface = huge_result(40);
        let outcome = apply_pressure_prune(&surface, policy, 10);
        let SurfaceItem::ToolResult { blocks, .. } = &outcome.surface.items[2] else {
            panic!("expected tool result");
        };
        let text = blocks[0].text.as_deref().unwrap();
        assert!(text.is_char_boundary(text.len()));
        assert!(text.contains("omitted"));
        assert!(std::str::from_utf8(text.as_bytes()).is_ok());
    }
}
