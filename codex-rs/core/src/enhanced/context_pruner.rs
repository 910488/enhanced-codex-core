use serde::Deserialize;
use serde::Serialize;

use super::telemetry::EnhancedEvent;
use super::telemetry::EnhancedEventFields;
use super::telemetry::EnhancedEventKind;

/// Shared experimental prune profile. V1 does not branch on model name.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultPrunePolicy {
    pub trigger_ratio: f64,
    pub min_text_chars: usize,
    pub keep_head_chars: usize,
    pub keep_tail_chars: usize,
}

impl Default for ToolResultPrunePolicy {
    fn default() -> Self {
        Self {
            trigger_ratio: 0.85,
            min_text_chars: 2_000,
            keep_head_chars: 400,
            keep_tail_chars: 800,
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
}

impl ModelVisibleSurface {
    pub fn estimated_tokens(&self) -> u64 {
        (self.char_count() as u64).div_ceil(4)
    }

    pub fn char_count(&self) -> usize {
        self.items.iter().map(item_chars).sum()
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
    pub chars_removed: u64,
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
            chars_removed: 0,
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
    for item in &mut next.items {
        if let SurfaceItem::ToolResult { blocks, .. } = item {
            prune_blocks(blocks, policy);
        }
    }
    let chars_removed = surface.char_count().saturating_sub(next.char_count()) as u64;
    next.generation = surface
        .generation
        .saturating_add(if chars_removed > 0 { 1 } else { 0 });
    let after = next.estimated_tokens();
    events.push(EnhancedEvent::new(
        EnhancedEventKind::ContextPruneCompleted,
        EnhancedEventFields {
            before_token_estimate: Some(before),
            after_token_estimate: Some(after),
            chars_removed: Some(chars_removed),
            ..EnhancedEventFields::default()
        },
    ));
    PruneOutcome {
        rewritten: chars_removed > 0,
        chars_removed,
        before_token_estimate: before,
        after_token_estimate: after,
        surface: next,
        events,
    }
}

fn prune_blocks(blocks: &mut [ContentBlock], policy: ToolResultPrunePolicy) -> u64 {
    // DeepSeek walks a global code-point cursor across every text block in the
    // tool result, emitting one omitted marker. Non-text blocks stay in order.
    let total_chars: usize = blocks
        .iter()
        .map(|block| {
            block
                .text
                .as_deref()
                .map(str::chars)
                .map(Iterator::count)
                .unwrap_or(0)
        })
        .sum();
    if total_chars < policy.min_text_chars {
        return 0;
    }
    let keep = policy
        .keep_head_chars
        .saturating_add(policy.keep_tail_chars);
    if total_chars <= keep {
        return 0;
    }
    let head_end = policy.keep_head_chars;
    let tail_start = total_chars.saturating_sub(policy.keep_tail_chars);
    let omitted = tail_start.saturating_sub(head_end);
    let marker = format!("\n[... omitted {omitted} characters ...]\n");
    let mut cursor = 0usize;
    let mut marker_inserted = false;
    let mut kept_chars = 0usize;
    for block in blocks.iter_mut() {
        let Some(text) = block.text.take() else {
            continue;
        };
        let mut kept = String::new();
        for character in text.chars() {
            if cursor < head_end || cursor >= tail_start {
                kept.push(character);
                kept_chars += 1;
            } else if !marker_inserted {
                kept.push_str(&marker);
                marker_inserted = true;
            }
            cursor += 1;
        }
        block.text = Some(kept);
    }
    total_chars.saturating_sub(kept_chars) as u64
}

fn item_chars(item: &SurfaceItem) -> usize {
    match item {
        SurfaceItem::User { text }
        | SurfaceItem::System { text }
        | SurfaceItem::Developer { text }
        | SurfaceItem::Assistant { text } => text.chars().count(),
        SurfaceItem::ToolCall {
            call_id,
            tool_name,
            tool_type,
            arguments,
        } => call_id.len() + tool_name.len() + tool_type.len() + arguments.chars().count(),
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
                    .map(|block| {
                        block
                            .text
                            .as_deref()
                            .map(str::chars)
                            .map(Iterator::count)
                            .unwrap_or(0)
                    })
                    .sum::<usize>()
        }
        SurfaceItem::Control { key, value } => key.len() + value.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn huge_result(chars: usize) -> ModelVisibleSurface {
        ModelVisibleSurface {
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
            min_text_chars: 20,
            keep_head_chars: 4,
            keep_tail_chars: 4,
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
            min_text_chars: 20,
            keep_head_chars: 4,
            keep_tail_chars: 4,
        };
        let surface = ModelVisibleSurface {
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
            min_text_chars: 8,
            keep_head_chars: 3,
            keep_tail_chars: 3,
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
