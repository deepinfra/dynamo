// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek V4.1 reasoning markers decided on token ids, the way vLLM's deepseek_v41
//! parser does (`token_id_terminals`).
//!
//! The model writes `<think>` / `</think>` as the special tokens 128821 / 128822, but the
//! same text can also come out as ordinary tokens. vLLM switches channels only on the
//! special tokens while streaming, so the frontend rewrites just their text to the
//! parser's stand-ins (see `deepseek_v41::THINK_START_TOKEN_MARKER`) before the unified
//! parser sees it. `usage.completion_tokens_details.reasoning_tokens` is counted from
//! the output token ids with vLLM's rules, independent of how the stream was chunked.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use dynamo_parsers_v2::unified::deepseek_v41::{
    THINK_END, THINK_END_TOKEN_MARKER, THINK_START, THINK_START_TOKEN_MARKER,
};

use crate::protocols::common::llm_backend::BackendOutput;
use crate::tokenizers::traits::Tokenizer;

pub(crate) const THINK_START_ID: u32 = 128821;
pub(crate) const THINK_END_ID: u32 = 128822;
const TOOL_START: &str = "<｜DSML｜ calls>";
const INVOKE_PREFIX: &str = "<｜DSML｜ invoke name=\"";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marker {
    Start,
    End,
}

impl Marker {
    fn of(id: u32) -> Option<Self> {
        match id {
            THINK_START_ID => Some(Self::Start),
            THINK_END_ID => Some(Self::End),
            _ => None,
        }
    }

    fn text(self) -> &'static str {
        match self {
            Self::Start => THINK_START,
            Self::End => THINK_END,
        }
    }

    fn stand_in(self) -> &'static str {
        match self {
            Self::Start => THINK_START_TOKEN_MARKER,
            Self::End => THINK_END_TOKEN_MARKER,
        }
    }
}

/// Which of vLLM's two `reasoning_tokens` counts a request reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReasoningCount {
    /// Streaming: the reasoning parser's count up to the first end of reasoning, as
    /// seen one token per chunk; zero when the prompt already closed reasoning.
    Stream,
    /// Non-streaming: every reasoning span of one token-id parse of the whole output.
    Full,
}

/// Per-request DeepSeek V4.1 token view shared by the backend-output stage and the
/// usage annotator.
pub(crate) struct DeepSeekV41TokenView {
    tokenizer: Arc<dyn Tokenizer>,
    rewrite_markers: bool,
    count: ReasoningCount,
    thinking: bool,
    /// Output ids per choice, kept only for the non-streaming `Full` count.
    output_ids: Mutex<HashMap<u32, Vec<u32>>>,
    /// Running streaming counts per choice. Streamed chunks carry usage on every
    /// chunk (`continuous_usage_stats`), so the count must advance with each new token
    /// instead of re-detokenizing the whole output per chunk (O(n^2) per request).
    stream_counters: Mutex<HashMap<u32, StreamReasoningCounter>>,
    aligners: Mutex<HashMap<u32, MarkerAligner>>,
}

impl DeepSeekV41TokenView {
    /// `None` when the tokenizer does not spell 128821/128822 as the reasoning markers.
    pub(crate) fn new(
        tokenizer: Arc<dyn Tokenizer>,
        rewrite_markers: bool,
        count: ReasoningCount,
        thinking: bool,
    ) -> Option<Arc<Self>> {
        let spelled = |id: u32, text: &str| {
            tokenizer
                .decode(&[id], false)
                .is_ok_and(|decoded| decoded.as_str() == text)
        };
        if !spelled(THINK_START_ID, THINK_START) || !spelled(THINK_END_ID, THINK_END) {
            tracing::warn!("deepseek_v41 tokenizer does not map 128821/128822 to <think>/</think>");
            return None;
        }
        Some(Arc::new(Self {
            tokenizer,
            rewrite_markers,
            count,
            thinking,
            output_ids: Mutex::new(HashMap::new()),
            stream_counters: Mutex::new(HashMap::new()),
            aligners: Mutex::new(HashMap::new()),
        }))
    }

    pub(crate) fn rewrites_markers(&self) -> bool {
        self.rewrite_markers
    }

    /// Record a backend chunk's token ids and, when streaming, replace the special
    /// tokens' marker text with the parser stand-ins.
    pub(crate) fn observe(&self, output: &mut BackendOutput) {
        let index = output.index.unwrap_or(0);
        match self.count {
            ReasoningCount::Full => {
                if let Ok(mut ids) = self.output_ids.lock() {
                    ids.entry(index)
                        .or_default()
                        .extend_from_slice(&output.token_ids);
                }
            }
            ReasoningCount::Stream if !self.thinking => {}
            ReasoningCount::Stream => {
                if let Ok(mut counters) = self.stream_counters.lock() {
                    let counter = counters
                        .entry(index)
                        .or_insert_with(|| StreamReasoningCounter::new(self.tokenizer.clone()));
                    counter.feed_all(&output.token_ids);
                }
            }
        }
        if !self.rewrite_markers {
            return;
        }
        let Ok(mut aligners) = self.aligners.lock() else {
            return;
        };
        let aligner = aligners.entry(index).or_default();
        aligner.push_ids(&output.token_ids);
        if let Some(text) = output.text.as_deref()
            && let Some(rewritten) = aligner.rewrite(text, self.tokenizer.as_ref())
        {
            output.text = Some(rewritten);
        }
    }

    /// `usage.completion_tokens_details.reasoning_tokens` for everything observed.
    pub(crate) fn reasoning_tokens(&self) -> u32 {
        if self.count == ReasoningCount::Stream {
            if !self.thinking {
                return 0;
            }
            let Ok(counters) = self.stream_counters.lock() else {
                return 0;
            };
            return counters
                .values()
                .map(|counter| counter.count)
                .fold(0u32, u32::saturating_add);
        }
        let Ok(ids) = self.output_ids.lock() else {
            return 0;
        };
        ids.values()
            .map(|ids| full_reasoning_tokens(self.tokenizer.as_ref(), ids, self.thinking))
            .fold(0u32, u32::saturating_add)
    }
}

/// Binds marker text in the stream to the special tokens that produced it.
///
/// Marker text can trail its token (the detokenizer holds an incomplete character back)
/// and the same text can also be spelled with ordinary tokens. Occurrences arrive in
/// token order, so the next occurrence is either a plain-text marker of the run of
/// ordinary tokens before the next unbound special token, or that special token.
#[derive(Default)]
struct MarkerAligner {
    /// Arrived special tokens whose text has not been seen, each with the ordinary
    /// tokens before it and how many plain-text markers that run spells.
    unbound: VecDeque<(Marker, Vec<u32>, Option<usize>)>,
    /// Ordinary tokens after the last arrived special token.
    open_run: Vec<u32>,
    /// Plain-text markers already seen from the run in front of the next special token.
    plain_seen: usize,
    /// The previous text's last bytes, to see a marker spelled across chunks.
    tail: String,
}

impl MarkerAligner {
    fn push_ids(&mut self, ids: &[u32]) {
        for id in ids {
            match Marker::of(*id) {
                Some(marker) => {
                    let run = std::mem::take(&mut self.open_run);
                    self.unbound.push_back((marker, run, None));
                }
                None => self.open_run.push(*id),
            }
        }
    }

    /// Whether the next marker occurrence in the text is the special token.
    fn next_is_special(&mut self, found: Marker, tokenizer: &dyn Tokenizer) -> bool {
        let plain_seen = self.plain_seen;
        let Some((marker, run, plain)) = self.unbound.front_mut() else {
            self.plain_seen += 1;
            return false;
        };
        let plain = *plain.get_or_insert_with(|| plain_marker_count(run, tokenizer));
        if plain_seen < plain || *marker != found {
            self.plain_seen += 1;
            return false;
        }
        self.unbound.pop_front();
        self.plain_seen = 0;
        true
    }

    /// `text` with the special tokens' marker text replaced by the stand-ins, or
    /// `None` when nothing changes. A marker spelled across two chunks is plain text:
    /// a special token's text always arrives whole.
    fn rewrite(&mut self, text: &str, tokenizer: &dyn Tokenizer) -> Option<String> {
        let carried = self.tail.len();
        let mut combined = std::mem::take(&mut self.tail);
        combined.push_str(text);
        let keep = floor_boundary(
            &combined,
            combined.len().saturating_sub(THINK_END.len() - 1),
        );
        self.tail = combined[keep..].to_string();
        if self.unbound.is_empty() && !combined.contains('<') {
            return None;
        }
        let mut out = String::with_capacity(text.len());
        let mut cursor = carried;
        for (at, marker) in marker_occurrences(&combined) {
            let end = at + marker.text().len();
            if end <= carried {
                continue;
            }
            if at < carried {
                self.plain_seen += 1;
                continue;
            }
            if !self.next_is_special(marker, tokenizer) {
                continue;
            }
            out.push_str(&combined[cursor..at]);
            out.push_str(marker.stand_in());
            cursor = end;
        }
        if cursor == carried {
            return None;
        }
        out.push_str(&combined[cursor..]);
        Some(out)
    }
}

fn marker_occurrences(text: &str) -> Vec<(usize, Marker)> {
    let mut found = Vec::new();
    let mut cursor = 0;
    while let Some(relative) = text[cursor..].find('<') {
        let at = cursor + relative;
        let rest = &text[at..];
        if rest.starts_with(THINK_START) {
            found.push((at, Marker::Start));
            cursor = at + THINK_START.len();
        } else if rest.starts_with(THINK_END) {
            found.push((at, Marker::End));
            cursor = at + THINK_END.len();
        } else {
            cursor = at + 1;
        }
    }
    found
}

fn plain_marker_count(run: &[u32], tokenizer: &dyn Tokenizer) -> usize {
    if run.is_empty() {
        return 0;
    }
    tokenizer
        .decode(run, false)
        .map(|decoded| marker_occurrences(decoded.as_str()).len())
        .unwrap_or(0)
}

/// vLLM streaming count, one token per chunk: tokens of the opening reasoning span,
/// excluding a token whose chunk carried no text yet (an incomplete character) and
/// stopping at `</think>` or a DSML calls block.
fn stream_reasoning_tokens(tokenizer: Arc<dyn Tokenizer>, ids: &[u32]) -> u32 {
    let mut counter = StreamReasoningCounter::new(tokenizer);
    counter.feed_all(ids);
    counter.count
}

/// Incremental form of vLLM's streaming `reasoning_tokens` count: fed one output token at
/// a time, it reaches the same count as a one-shot pass over the same ids, at O(1)
/// amortized work per token. Once reasoning ends (`</think>` or a DSML calls block) the
/// count is final and further tokens are ignored.
struct StreamReasoningCounter {
    decoder: crate::tokenizers::DecodeStream,
    count: u32,
    emitted: usize,
    /// The last `TOOL_START.len()` bytes of emitted text, to see a calls block that
    /// spans chunks.
    window: String,
    window_start: usize,
    /// Byte offsets (in emitted text) at which counted tokens start, within `window`.
    starts: VecDeque<usize>,
    done: bool,
}

impl StreamReasoningCounter {
    fn new(tokenizer: Arc<dyn Tokenizer>) -> Self {
        Self {
            decoder: crate::tokenizers::DecodeStream::new(tokenizer, &[], false),
            count: 0,
            emitted: 0,
            window: String::new(),
            window_start: 0,
            starts: VecDeque::new(),
            done: false,
        }
    }

    fn feed_all(&mut self, ids: &[u32]) {
        for &id in ids {
            if self.done {
                return;
            }
            self.feed(id);
        }
    }

    fn feed(&mut self, id: u32) {
        match Marker::of(id) {
            Some(Marker::End) => {
                self.done = true;
                return;
            }
            Some(Marker::Start) => return,
            None => {}
        }
        let Ok(Some(text)) = self.decoder.step(id) else {
            return;
        };
        if text.is_empty() {
            return;
        }
        self.starts.push_back(self.emitted);
        self.count += 1;
        self.emitted += text.len();
        self.window.push_str(&text);
        if let Some(at) = self.window.find(TOOL_START) {
            let tool_at = self.window_start + at;
            let inside = self.starts.iter().filter(|start| **start >= tool_at).count() as u32;
            self.count -= inside;
            self.done = true;
            return;
        }
        let keep_from = floor_boundary(
            &self.window,
            self.window.len().saturating_sub(TOOL_START.len()),
        );
        self.window_start += keep_from;
        self.window.drain(..keep_from);
        while self
            .starts
            .front()
            .is_some_and(|start| *start < self.window_start)
        {
            self.starts.pop_front();
        }
    }
}

fn floor_boundary(text: &str, mut at: usize) -> usize {
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Channel {
    Reasoning,
    Content,
    Tool,
}

/// vLLM non-streaming count: the reasoning-channel tokens of one token-id parse of the
/// whole output, with each token credited at the first character of its own decoded
/// text (or at the start of its run when that text is not found, as vLLM's lexer does).
fn full_reasoning_tokens(tokenizer: &dyn Tokenizer, ids: &[u32], thinking: bool) -> u32 {
    let mut channel = if thinking {
        Channel::Reasoning
    } else {
        Channel::Content
    };
    let mut count = 0u32;
    let mut run: Vec<u32> = Vec::new();
    let mut flush = |run: &mut Vec<u32>, channel: &mut Channel, count: &mut u32| {
        if run.is_empty() || *channel == Channel::Tool {
            run.clear();
            return;
        }
        let text: String = tokenizer
            .decode(run, false)
            .map(String::from)
            .unwrap_or_default();
        match *channel {
            Channel::Reasoning => match text.find(TOOL_START) {
                None => *count = count.saturating_add(run.len() as u32),
                Some(tool_at) => {
                    let before = token_starts(tokenizer, run, &text)
                        .into_iter()
                        .filter(|start| *start < tool_at)
                        .count();
                    *count = count.saturating_add(before as u32);
                    *channel = Channel::Tool;
                }
            },
            Channel::Content => {
                if text.contains(TOOL_START) || text.contains(INVOKE_PREFIX) {
                    *channel = Channel::Tool;
                }
            }
            Channel::Tool => {}
        }
        run.clear();
    };
    for &id in ids {
        let Some(marker) = Marker::of(id) else {
            run.push(id);
            continue;
        };
        flush(&mut run, &mut channel, &mut count);
        channel = match (channel, marker) {
            (Channel::Content, Marker::Start) => Channel::Reasoning,
            (Channel::Reasoning, Marker::End) => Channel::Content,
            (unchanged, _) => unchanged,
        };
    }
    flush(&mut run, &mut channel, &mut count);
    count
}

/// Byte offset each token is credited at in `text`, per vLLM's `_char_token_counts`.
fn token_starts(tokenizer: &dyn Tokenizer, run: &[u32], text: &str) -> Vec<usize> {
    let mut starts = Vec::with_capacity(run.len());
    let mut position = 0;
    for id in run {
        let piece: String = tokenizer
            .decode(&[*id], false)
            .map(String::from)
            .unwrap_or_default();
        if piece.is_empty() {
            starts.push(0);
            continue;
        }
        match text
            .get(position..)
            .and_then(|rest| rest.find(piece.as_str()))
        {
            Some(relative) => {
                starts.push(position + relative);
                position += relative + piece.len();
            }
            None => starts.push(0),
        }
    }
    starts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizers::traits::{Decoder, Encoder};
    use crate::tokenizers::{DecodeResult, Encoding};

    /// A byte-level toy vocabulary: ids map to fixed strings.
    struct Vocab(HashMap<u32, &'static str>);

    impl Vocab {
        fn new() -> Arc<dyn Tokenizer> {
            let mut map = HashMap::new();
            map.insert(THINK_START_ID, THINK_START);
            map.insert(THINK_END_ID, THINK_END);
            for (id, text) in [
                (1, "A"),
                (2, "B"),
                (3, " "),
                (4, "<"),
                (5, "/"),
                (6, "think"),
                (7, ">"),
                (8, "<｜DSML｜ calls>"),
                (9, "x"),
                (10, "<th"),
                (11, "ink>"),
            ] {
                map.insert(id, text);
            }
            Arc::new(Self(map))
        }
    }

    impl Encoder for Vocab {
        fn encode(&self, _input: &str) -> anyhow::Result<Encoding> {
            unimplemented!()
        }
        fn encode_batch(&self, _inputs: &[&str]) -> anyhow::Result<Vec<Encoding>> {
            unimplemented!()
        }
    }

    impl Decoder for Vocab {
        fn decode(&self, ids: &[u32], _skip: bool) -> anyhow::Result<DecodeResult> {
            Ok(DecodeResult::from_decoded(
                ids.iter().map(|id| self.0[id]).collect::<String>(),
            ))
        }
    }

    impl Tokenizer for Vocab {}

    fn text_of(tokenizer: &dyn Tokenizer, ids: &[u32]) -> String {
        tokenizer.decode(ids, false).unwrap().into()
    }

    fn rewrite_chunks(chunks: &[(&str, &[u32])]) -> String {
        let tokenizer = Vocab::new();
        let mut aligner = MarkerAligner::default();
        let mut out = String::new();
        for (text, ids) in chunks {
            aligner.push_ids(ids);
            out.push_str(
                &aligner
                    .rewrite(text, tokenizer.as_ref())
                    .unwrap_or_else(|| text.to_string()),
            );
        }
        out
    }

    #[test]
    fn only_special_marker_tokens_become_stand_ins() {
        // "A</think>B" with the closer spelled as ordinary tokens, then the special
        // opener and closer.
        assert_eq!(
            rewrite_chunks(&[(
                "A</think>B<think>x</think>",
                &[1, 4, 5, 6, 7, 2, THINK_START_ID, 9, THINK_END_ID]
            )]),
            format!("A</think>B{THINK_START_TOKEN_MARKER}x{THINK_END_TOKEN_MARKER}")
        );
        // A marker split across chunks is ordinary text, also when a special closer
        // follows it.
        assert_eq!(
            rewrite_chunks(&[
                ("A<th", &[1, 10]),
                ("ink>", &[11]),
                ("B</think>", &[2, THINK_END_ID])
            ]),
            format!("A<think>B{THINK_END_TOKEN_MARKER}")
        );
    }

    #[test]
    fn marker_text_trailing_its_token_is_still_bound() {
        // The detokenizer held the special closer's text back to the next chunk, where a
        // plain-text closer follows it.
        assert_eq!(
            rewrite_chunks(&[
                ("A", &[1]),
                ("", &[THINK_END_ID]),
                ("</think>B</think>", &[2, 4, 5, 6, 7]),
            ]),
            format!("A{THINK_END_TOKEN_MARKER}B</think>")
        );
        // A plain-text closer before the special one in the same chunk.
        assert_eq!(
            rewrite_chunks(&[("A</think>B</think>x", &[1, 4, 5, 6, 7, 2, THINK_END_ID, 9])]),
            format!("A</think>B{THINK_END_TOKEN_MARKER}x")
        );
    }

    #[test]
    fn reasoning_tokens_follow_vllm_stream_and_full_counts() {
        let tokenizer = Vocab::new();
        // thinking: "AB</think>A<think>B</think>x" -> stream counts the first span only.
        let ids = [1, 2, THINK_END_ID, 1, THINK_START_ID, 2, THINK_END_ID, 9];
        assert_eq!(stream_reasoning_tokens(tokenizer.clone(), &ids), 2);
        assert_eq!(full_reasoning_tokens(tokenizer.as_ref(), &ids, true), 3);
        // non-thinking: only the opened span counts in the full parse.
        assert_eq!(full_reasoning_tokens(tokenizer.as_ref(), &ids, false), 1);
        // A plain-text closer is not a marker: everything stays reasoning.
        let plain = [1, 4, 5, 6, 7, 2];
        assert_eq!(stream_reasoning_tokens(tokenizer.clone(), &plain), 6);
        assert_eq!(full_reasoning_tokens(tokenizer.as_ref(), &plain, true), 6);
        // A DSML calls block ends reasoning; later markers inside the tool region
        // do not reopen it in the full parse.
        let tool = [1, 3, 8, 9, THINK_START_ID, 2, THINK_END_ID];
        assert_eq!(stream_reasoning_tokens(tokenizer.clone(), &tool), 2);
        assert_eq!(full_reasoning_tokens(tokenizer.as_ref(), &tool, true), 2);
    }

    #[test]
    fn reasoning_tokens_do_not_depend_on_the_stream_interval() {
        // reasoning_tokens_chunk_estimate: the count is the tokens before `</think>`,
        // however the backend batched them into chunks.
        let tokenizer = Vocab::new();
        let ids = [1, 2, 3, 1, 2, THINK_END_ID, 1, 2, 3];
        for interval in 1..=ids.len() {
            for count in [ReasoningCount::Stream, ReasoningCount::Full] {
                let view = DeepSeekV41TokenView::new(tokenizer.clone(), true, count, true).unwrap();
                for chunk in ids.chunks(interval) {
                    let mut output: BackendOutput = serde_json::from_value(serde_json::json!({
                        "token_ids": chunk,
                        "tokens": [],
                        "text": text_of(tokenizer.as_ref(), chunk),
                        "index": 0,
                    }))
                    .unwrap();
                    view.observe(&mut output);
                }
                assert_eq!(view.reasoning_tokens(), 5, "interval {interval} {count:?}");
            }
        }
    }

    /// The pre-incremental one-shot streaming count, verbatim, as the parity reference.
    fn reference_stream_count(tokenizer: Arc<dyn Tokenizer>, ids: &[u32]) -> u32 {
        let mut decoder = crate::tokenizers::DecodeStream::new(tokenizer, &[], false);
        let mut count = 0u32;
        let mut emitted = 0usize;
        let mut window = String::new();
        let mut window_start = 0usize;
        let mut starts: VecDeque<usize> = VecDeque::new();
        for &id in ids {
            match Marker::of(id) {
                Some(Marker::End) => break,
                Some(Marker::Start) => continue,
                None => {}
            }
            let Ok(Some(text)) = decoder.step(id) else {
                continue;
            };
            if text.is_empty() {
                continue;
            }
            starts.push_back(emitted);
            count += 1;
            emitted += text.len();
            window.push_str(&text);
            if let Some(at) = window.find(TOOL_START) {
                let tool_at = window_start + at;
                let inside = starts.iter().filter(|start| **start >= tool_at).count() as u32;
                return count - inside;
            }
            let keep_from = floor_boundary(&window, window.len().saturating_sub(TOOL_START.len()));
            window_start += keep_from;
            window.drain(..keep_from);
            while starts.front().is_some_and(|start| *start < window_start) {
                starts.pop_front();
            }
        }
        count
    }

    /// With `continuous_usage_stats` every chunk carries usage, so the streaming count is
    /// read after every chunk. Each read must equal a one-shot count over the ids seen so
    /// far (the running counter must not drift from the reference), for every chunking.
    #[test]
    fn streaming_count_read_after_every_chunk_matches_one_shot_prefix_count() {
        let tokenizer = Vocab::new();
        let sequences: [&[u32]; 4] = [
            &[1, 2, 3, 1, 2, THINK_END_ID, 1, 2, 3],
            &[1, 2, THINK_END_ID, 1, THINK_START_ID, 2, THINK_END_ID, 9],
            &[1, 4, 5, 6, 7, 2],
            &[1, 3, 8, 9, THINK_START_ID, 2, THINK_END_ID],
        ];
        for ids in sequences {
            for interval in 1..=ids.len() {
                let view = DeepSeekV41TokenView::new(
                    tokenizer.clone(),
                    true,
                    ReasoningCount::Stream,
                    true,
                )
                .unwrap();
                let mut seen = 0;
                for chunk in ids.chunks(interval) {
                    let mut output: BackendOutput = serde_json::from_value(serde_json::json!({
                        "token_ids": chunk,
                        "tokens": [],
                        "text": text_of(tokenizer.as_ref(), chunk),
                        "index": 0,
                    }))
                    .unwrap();
                    view.observe(&mut output);
                    seen += chunk.len();
                    assert_eq!(
                        view.reasoning_tokens(),
                        reference_stream_count(tokenizer.clone(), &ids[..seen]),
                        "ids {ids:?} interval {interval} after {seen} tokens"
                    );
                }
            }
        }
    }

    #[test]
    fn token_view_counts_zero_when_the_prompt_closed_reasoning_while_streaming() {
        let tokenizer = Vocab::new();
        let view =
            DeepSeekV41TokenView::new(tokenizer, true, ReasoningCount::Stream, false).unwrap();
        let mut output: BackendOutput = serde_json::from_value(serde_json::json!({
            "token_ids": [1, THINK_START_ID, 2, THINK_END_ID],
            "tokens": [],
            "text": "A<think>B</think>",
            "index": 0,
        }))
        .unwrap();
        view.observe(&mut output);
        assert_eq!(
            output.text.as_deref(),
            Some(format!("A{THINK_START_TOKEN_MARKER}B{THINK_END_TOKEN_MARKER}").as_str())
        );
        assert_eq!(view.reasoning_tokens(), 0);
    }
}
