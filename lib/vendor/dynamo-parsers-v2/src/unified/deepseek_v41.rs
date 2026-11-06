// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use serde_json::Value;

use crate::tool_calling::scan::{
    BareRecoveryLatch, InvokeBoundary, InvokeBoundaryFactory, InvokeEmitter, ReasoningSpec,
    WrappedBlockScanner, WrappedBlockSpec,
};
use crate::tool_calling::traits::{Tool, ToolCallDelta};
use crate::unified::{
    GuidedRouted, ScannerUnified, UnifiedParser, UnifiedParserEvent, UnifiedParserOutput,
};

const BLOCK_START: &str = "<｜DSML｜ calls>";
const BLOCK_END: &str = "</｜DSML｜ calls>";
const INVOKE_START: &str = "<｜DSML｜ invoke name=\"";
const INVOKE_END: &str = "</｜DSML｜ invoke>";
const PARAMETER_START: &str = "<｜DSML｜ parameter name=\"";
const PARAMETER_END: &str = "</｜DSML｜ parameter>";

mod args;

/// Stand-ins for the `<think>` / `</think>` SPECIAL tokens (ids 128821/128822) in text
/// fed to [`deepseek_v41_unified_token_markers`]. vLLM matches the reasoning markers by
/// token id, so a marker spelled with ordinary tokens is plain text; the frontend
/// rewrites only the special tokens' text to these noncharacters before parsing and
/// [`restore_token_markers`] maps them back in the parsed output.
pub const THINK_START_TOKEN_MARKER: &str = "\u{FDD0}";
pub const THINK_END_TOKEN_MARKER: &str = "\u{FDD1}";
pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";

pub(crate) fn deepseek_v41_unified(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    build(tools, THINK_START, THINK_END)
}

/// DeepSeek V4.1 with the reasoning markers matched only as the special-token stand-ins.
pub fn deepseek_v41_unified_token_markers(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    build(tools, THINK_START_TOKEN_MARKER, THINK_END_TOKEN_MARKER)
}

/// DeepSeek V4.1 as vLLM parses a complete (non-streaming) output: first a reasoning
/// pass over the text, which ends reasoning at `</think>` or a DSML calls block and
/// treats every `<think>`/`</think>` as a marker, even inside a calls block; then the
/// tool parser over the content that pass left.
pub fn deepseek_v41_unified_two_pass(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    Box::new(TwoPass {
        split: ReasoningSplit::default(),
        content: build(tools, THINK_START, THINK_END),
        held_reasoning_ws: String::new(),
    })
}

#[derive(Default)]
struct ReasoningSplit {
    in_reasoning: bool,
    buffer: String,
}

impl ReasoningSplit {
    /// Split buffered text into `(is_reasoning, text)` runs, holding back a suffix that
    /// may still grow into a marker unless `flush`.
    fn drain(&mut self, flush: bool, out: &mut Vec<(bool, String)>) {
        loop {
            let mut next: Option<(usize, usize, Option<bool>)> = None;
            for (marker, to) in [(THINK_START, Some(true)), (THINK_END, Some(false))] {
                if let Some(at) = self.buffer.find(marker)
                    && next.is_none_or(|(best, _, _)| at < best)
                {
                    next = Some((at, marker.len(), to));
                }
            }
            if self.in_reasoning
                && let Some(at) = self.buffer.find(BLOCK_START)
                && next.is_none_or(|(best, _, _)| at < best)
            {
                // The block opener ends reasoning and stays in the content.
                next = Some((at, 0, Some(false)));
            }
            let Some((at, len, to)) = next else { break };
            self.emit(at, out);
            self.buffer.drain(..len);
            if let Some(to) = to {
                self.in_reasoning = to;
            }
        }
        let keep = if flush {
            0
        } else {
            let mut markers = vec![THINK_START, THINK_END];
            if self.in_reasoning {
                markers.push(BLOCK_START);
            }
            crate::tool_calling::scan::marker_prefix_suffix_len(&self.buffer, markers)
        };
        self.emit(self.buffer.len() - keep, out);
    }

    fn emit(&mut self, len: usize, out: &mut Vec<(bool, String)>) {
        if len == 0 {
            return;
        }
        let text: String = self.buffer.drain(..len).collect();
        match out.last_mut() {
            Some((reasoning, last)) if *reasoning == self.in_reasoning => last.push_str(&text),
            _ => out.push((self.in_reasoning, text)),
        }
    }
}

struct TwoPass {
    split: ReasoningSplit,
    content: Box<dyn UnifiedParser>,
    /// vLLM strips trailing whitespace off the joined reasoning only.
    held_reasoning_ws: String,
}

impl TwoPass {
    fn route(
        &mut self,
        runs: Vec<(bool, String)>,
        output: &mut UnifiedParserOutput,
    ) -> anyhow::Result<()> {
        for (reasoning, text) in runs {
            if reasoning {
                let combined = std::mem::take(&mut self.held_reasoning_ws) + &text;
                let trimmed = combined.trim_end();
                self.held_reasoning_ws = combined[trimmed.len()..].to_string();
                output.push_reasoning(trimmed);
            } else {
                self.content.parse_into(&text, output)?;
            }
        }
        Ok(())
    }
}

impl UnifiedParser for TwoPass {
    fn initialize_request(&mut self, init: crate::unified::UnifiedParserInit) -> anyhow::Result<()> {
        self.split = ReasoningSplit {
            in_reasoning: init.starting_state == crate::unified::UnifiedParserStartingState::Reasoning,
            buffer: String::new(),
        };
        self.held_reasoning_ws.clear();
        self.content.initialize_request(crate::unified::UnifiedParserInit {
            starting_state: crate::unified::UnifiedParserStartingState::Response,
            ..init
        })
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
        self.split.buffer.push_str(delta);
        let mut runs = Vec::new();
        self.split.drain(false, &mut runs);
        self.route(runs, output)
    }

    fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();
        let mut runs = Vec::new();
        self.split.drain(true, &mut runs);
        self.route(runs, &mut output)?;
        let mut tail = self.content.finish()?;
        output.append(&mut tail);
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.held_reasoning_ws.clear();
        let mut recovered = std::mem::take(&mut self.split.buffer);
        recovered.insert_str(0, &self.content.reset());
        recovered
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.content.tool_call_id(tool_index)
    }

    fn suppress_tool_calls(&mut self) {
        self.content.suppress_tool_calls()
    }
}

/// Map the special-token stand-ins back to the marker text.
pub fn restore_token_markers(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.contains(['\u{FDD0}', '\u{FDD1}']) {
        return std::borrow::Cow::Borrowed(text);
    }
    std::borrow::Cow::Owned(
        text.replace(THINK_START_TOKEN_MARKER, THINK_START)
            .replace(THINK_END_TOKEN_MARKER, THINK_END),
    )
}

fn build(
    tools: &[Tool],
    think_start: &'static str,
    think_end: &'static str,
) -> Box<dyn UnifiedParser> {
    let spec = WrappedBlockSpec {
        family: "deepseek_v41",
        block_starts: vec![BLOCK_START.into()],
        block_ends: vec![BLOCK_END.into()],
        invoke_start: INVOKE_START.into(),
        invoke_end: INVOKE_END.into(),
        orphan_markers: vec![BLOCK_END.into()],
        holdback_markers: vec![BLOCK_START.into(), BLOCK_END.into(), INVOKE_START.into()],
        bare_recovery_latch: BareRecoveryLatch::Set,
        invoke_boundary_factory: Some(InvokeBoundaryFactory::custom(invocation_boundary)),
        preserve_special_tokens: true,
        block_ends_turn: true,
        bare_invoke_in_reasoning_is_text: true,
        ..Default::default()
    };
    let emitter = DeepSeekV41 {
        tools: tools.to_vec(),
    };
    let scanner = WrappedBlockScanner::new(spec, emitter).with_reasoning(ReasoningSpec {
        start: think_start,
        end: think_end,
        preserve_special_tokens: true,
        absorb_end_without_reasoning: true,
        ..Default::default()
    });
    Box::new(VllmWhitespace::new(Box::new(GuidedRouted::new(
        ScannerUnified::new(scanner),
    ))))
}

/// vLLM's DeepSeek V4/V4.1 parser-engine whitespace rules
/// (`strip_trailing_reasoning_whitespace`, `drop_whitespace_only_content_before_tools`):
/// reasoning never ends in whitespace, and content that is only whitespace before
/// any visible text is held back, then dropped if a tool call follows, even one
/// truncated at EOF (kept at EOF otherwise). Under `tool_choice: "none"` vLLM's
/// engine suppresses call events, so calls are dropped here and keep that whitespace.
struct VllmWhitespace {
    inner: Box<dyn UnifiedParser>,
    deferred_reasoning: String,
    deferred_text: String,
    content_has_text: bool,
    tool_seen: bool,
    suppress_tool_calls: bool,
}

impl VllmWhitespace {
    fn new(inner: Box<dyn UnifiedParser>) -> Self {
        Self {
            inner,
            deferred_reasoning: String::new(),
            deferred_text: String::new(),
            content_has_text: false,
            tool_seen: false,
            suppress_tool_calls: false,
        }
    }

    fn clear(&mut self) {
        self.deferred_reasoning.clear();
        self.deferred_text.clear();
        self.content_has_text = false;
        self.tool_seen = false;
    }

    fn shape(&mut self, events: Vec<UnifiedParserEvent>, out: &mut UnifiedParserOutput) {
        for event in events {
            match event {
                UnifiedParserEvent::Reasoning(text) => {
                    let combined = std::mem::take(&mut self.deferred_reasoning) + &text;
                    let trimmed = combined.trim_end();
                    self.deferred_reasoning = combined[trimmed.len()..].to_string();
                    if !trimmed.is_empty() {
                        out.push_reasoning(trimmed);
                    }
                }
                UnifiedParserEvent::Text(text) => {
                    self.deferred_reasoning.clear();
                    if self.content_has_text {
                        out.push_text(text);
                        continue;
                    }
                    let combined = std::mem::take(&mut self.deferred_text) + &text;
                    if !combined.trim().is_empty() {
                        self.content_has_text = true;
                        out.push_text(combined);
                    } else if !self.tool_seen {
                        self.deferred_text = combined;
                    }
                }
                UnifiedParserEvent::ToolCall(_) if self.suppress_tool_calls => {}
                UnifiedParserEvent::ToolCall(call) => {
                    self.deferred_reasoning.clear();
                    self.deferred_text.clear();
                    self.tool_seen = true;
                    out.push_call(call);
                }
            }
        }
    }
}

impl UnifiedParser for VllmWhitespace {
    fn initialize_request(&mut self, init: crate::unified::UnifiedParserInit) -> anyhow::Result<()> {
        self.clear();
        self.inner.initialize_request(init)
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
        let mut raw = UnifiedParserOutput::default();
        let result = self.inner.parse_into(delta, &mut raw);
        self.shape(raw.events, output);
        result
    }

    fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
        let raw = self.inner.finish()?;
        let mut out = UnifiedParserOutput::default();
        self.shape(raw.events, &mut out);
        if !self.deferred_text.is_empty() && !self.tool_seen {
            out.push_text(std::mem::take(&mut self.deferred_text));
        }
        Ok(out)
    }

    fn suppress_tool_calls(&mut self) {
        self.suppress_tool_calls = true;
    }

    fn reset(&mut self) -> String {
        self.clear();
        self.inner.reset()
    }

    fn preserve_special_tokens(&self) -> bool {
        self.inner.preserve_special_tokens()
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.inner.tool_call_id(tool_index)
    }
}

/// A parameter header. As in vLLM's `_dsml_arg_converter`, only `string="true"`
/// marks a literal string; a missing or other `string` attribute means "JSON if it
/// parses, else the literal text".
fn parameter_header(text: &str) -> Option<(&str, bool, &str)> {
    let (name, rest) = text.strip_prefix(PARAMETER_START)?.split_once('"')?;
    if let Some(value) = rest.strip_prefix('>') {
        return Some((name, false, value));
    }
    let (string, value) = rest.strip_prefix(" string=\"")?.split_once("\">")?;
    Some((name, string == "true", value))
}

/// `json.dumps(value, ensure_ascii=False)`: `", "`/`": "` separators and Python
/// float repr, so arguments are byte-identical to vLLM's deepseek_v41 parser.
pub fn python_json(value: &Value) -> String {
    use serde::Serialize;
    use serde_json::ser::Formatter;
    use std::io;

    struct PythonFormatter;

    impl Formatter for PythonFormatter {
        fn begin_array_value<W: ?Sized + io::Write>(&mut self, w: &mut W, first: bool) -> io::Result<()> {
            if first { Ok(()) } else { w.write_all(b", ") }
        }

        fn begin_object_key<W: ?Sized + io::Write>(&mut self, w: &mut W, first: bool) -> io::Result<()> {
            if first { Ok(()) } else { w.write_all(b", ") }
        }

        fn begin_object_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
            w.write_all(b": ")
        }

        fn write_f64<W: ?Sized + io::Write>(&mut self, w: &mut W, value: f64) -> io::Result<()> {
            w.write_all(python_float_repr(value).as_bytes())
        }
    }

    let mut buf = Vec::with_capacity(64);
    value
        .serialize(&mut serde_json::Serializer::with_formatter(&mut buf, PythonFormatter))
        .expect("JSON value serialization into a Vec is infallible");
    String::from_utf8(buf).expect("serde_json writes UTF-8")
}

/// Python `repr(float)`: shortest round-trip digits, exponent form below 1e-4 or
/// from 1e16 (`1e-05`, `1e+16`), `.0` on integral values.
fn python_float_repr(value: f64) -> String {
    if !value.is_finite() {
        return if value.is_nan() { "NaN".into() } else if value > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let sci = format!("{value:e}");
    let (sign, sci) = sci.strip_prefix('-').map_or(("", sci.as_str()), |rest| ("-", rest));
    let (mantissa, exponent) = sci.split_once('e').expect("LowerExp has an exponent");
    let exponent: i32 = exponent.parse().expect("LowerExp exponent is an integer");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let point = exponent + 1;
    let body = if !(-4..16).contains(&exponent) {
        let (head, tail) = digits.split_at(1);
        let fraction = if tail.is_empty() { String::new() } else { format!(".{tail}") };
        let exp_sign = if exponent < 0 { '-' } else { '+' };
        format!("{head}{fraction}e{exp_sign}{:02}", exponent.abs())
    } else if point <= 0 {
        format!("0.{}{digits}", "0".repeat((-point) as usize))
    } else if point as usize >= digits.len() {
        format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
    } else {
        let (int, frac) = digits.split_at(point as usize);
        format!("{int}.{frac}")
    };
    format!("{sign}{body}")
}

#[cfg(test)]
std::thread_local! {
    static BOUNDARY_EXAMINED_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn count_boundary_bytes(bytes: usize) {
    #[cfg(test)]
    BOUNDARY_EXAMINED_BYTES.with(|examined| examined.set(examined.get() + bytes));
    #[cfg(not(test))]
    let _ = bytes;
}

fn find_from(text: &str, start: usize, marker: &str) -> Option<usize> {
    let suffix = &text[start..];
    count_boundary_bytes(suffix.len());
    suffix.find(marker).map(|at| start + at)
}

fn next_scan_start(text: &str, marker_len: usize) -> usize {
    let mut start = text.len().saturating_sub(marker_len.saturating_sub(1));
    while !text.is_char_boundary(start) {
        start -= 1;
    }
    start
}

#[derive(Default)]
enum InvocationPosition {
    #[default]
    BetweenParameters,
    ParameterHeader {
        start: usize,
        scan_from: usize,
    },
    ParameterValue {
        start: usize,
        scan_from: usize,
    },
    InvalidParameter {
        start: usize,
    },
}

#[derive(Default)]
struct DeepSeekV41InvocationBoundary {
    position: InvocationPosition,
    scan_from: usize,
}

impl DeepSeekV41InvocationBoundary {
    fn malformed_end(text: &str, start: usize, flush: bool) -> Option<usize> {
        flush
            .then(|| find_from(text, start, INVOKE_END))
            .flatten()
            .map(|at| at + INVOKE_END.len())
    }
}

impl InvokeBoundary for DeepSeekV41InvocationBoundary {
    fn end_append(
        &mut self,
        candidate: &str,
        _append: &str,
        flush: bool,
        _tool_index: usize,
    ) -> Option<usize> {
        loop {
            match self.position {
                InvocationPosition::BetweenParameters => {
                    let close = find_from(candidate, self.scan_from, INVOKE_END);
                    let parameter = find_from(candidate, self.scan_from, PARAMETER_START);
                    match (close, parameter) {
                        (Some(close), Some(parameter)) if parameter <= close => {
                            self.position = InvocationPosition::ParameterHeader {
                                start: parameter,
                                scan_from: parameter + PARAMETER_START.len(),
                            };
                        }
                        (Some(close), _) => return Some(close + INVOKE_END.len()),
                        (None, Some(parameter)) => {
                            self.position = InvocationPosition::ParameterHeader {
                                start: parameter,
                                scan_from: parameter + PARAMETER_START.len(),
                            };
                        }
                        (None, None) => {
                            self.scan_from = next_scan_start(
                                candidate,
                                INVOKE_END.len().max(PARAMETER_START.len()),
                            );
                            return None;
                        }
                    }
                }
                InvocationPosition::ParameterHeader { start, scan_from } => {
                    let Some(header_end) = find_from(candidate, scan_from, "\">") else {
                        if flush {
                            return Self::malformed_end(candidate, start, true);
                        }
                        self.position = InvocationPosition::ParameterHeader {
                            start,
                            scan_from: next_scan_start(candidate, 2),
                        };
                        return None;
                    };
                    let Some((_, _, value)) = parameter_header(&candidate[start..]) else {
                        self.position = InvocationPosition::InvalidParameter { start };
                        continue;
                    };
                    let value_start = candidate.len() - value.len();
                    debug_assert!(value_start >= header_end + 2);
                    self.position = InvocationPosition::ParameterValue {
                        start,
                        scan_from: value_start,
                    };
                }
                InvocationPosition::ParameterValue { start, scan_from } => {
                    let Some(value_end) = find_from(candidate, scan_from, PARAMETER_END) else {
                        if flush {
                            return Self::malformed_end(candidate, start, true);
                        }
                        self.position = InvocationPosition::ParameterValue {
                            start,
                            scan_from: next_scan_start(candidate, PARAMETER_END.len()),
                        };
                        return None;
                    };
                    self.scan_from = value_end + PARAMETER_END.len();
                    self.position = InvocationPosition::BetweenParameters;
                }
                InvocationPosition::InvalidParameter { start } => {
                    return Self::malformed_end(candidate, start, flush);
                }
            }
        }
    }

    fn opens(&self, _text: &str, _at: usize) -> bool {
        true
    }

    fn holdback(&self, _text: &str) -> usize {
        0
    }

    fn resync(&mut self, _text: &str, _flush: bool, _tool_index: usize) -> Option<usize> {
        None
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

fn invocation_boundary() -> Box<dyn InvokeBoundary> {
    Box::new(DeepSeekV41InvocationBoundary::default())
}

/// The request's tools give vLLM's schema coercion and wrapper unwrapping their schemas.
struct DeepSeekV41 {
    tools: Vec<Tool>,
}

impl DeepSeekV41 {
    /// One call as vLLM's `_build_extracted_result` builds it: the stripped name (an empty
    /// one is dropped) and `args::invoke_arguments` of the invoke body.
    fn call(&self, name: &str, body: &str, tool_index: usize) -> Option<ToolCallDelta> {
        let name = name.trim();
        (!name.is_empty()).then(|| ToolCallDelta {
            tool_index,
            name: Some(name.to_string()),
            arguments: args::invoke_arguments(body, name, &self.tools),
            complete: true,
        })
    }
}

impl InvokeEmitter for DeepSeekV41 {
    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let (name, body) = invoke
            .strip_prefix(INVOKE_START)
            .and_then(|text| text.split_once("\">"))
            .context("invalid DeepSeek V4.1 invocation header")?;
        let body = body
            .strip_suffix(INVOKE_END)
            .context("incomplete DeepSeek V4.1 invocation")?;
        Ok(self.call(name, body, tool_index))
    }

    /// vLLM closes a tool call still open at finish: the name streamed so far and the
    /// parameters closed so far (`_dsml_arg_converter(partial=False)`), even mid-name.
    fn parse_truncated_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let Some(rest) = invoke.strip_prefix(INVOKE_START) else {
            return Ok(None);
        };
        let (name, body) = rest.split_once("\">").unwrap_or((rest, ""));
        let body = body.find(INVOKE_END).map_or(body, |end| &body[..end]);
        Ok(self.call(name, body, tool_index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{
        UnifiedEvent, UnifiedParserExt, UnifiedParserInit, UnifiedParserOutput,
        UnifiedParserStartingState, UnifiedToolOutputMode, create_unified_parser_for_family,
    };

    fn parse_chunks(
        input: &str,
        split: usize,
        state: UnifiedParserStartingState,
    ) -> UnifiedParserOutput {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        assert!(parser.preserve_special_tokens());
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: state,
                ..Default::default()
            })
            .unwrap();
        let mut output = UnifiedParserOutput::default();
        parser.parse_into(&input[..split], &mut output).unwrap();
        parser.parse_into(&input[split..], &mut output).unwrap();
        output.append(&mut parser.finish().unwrap());
        output
    }

    fn assert_every_split(
        input: &str,
        state: UnifiedParserStartingState,
        expected: Vec<UnifiedEvent>,
    ) {
        for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
            assert_eq!(
                parse_chunks(input, split, state).assembled(),
                expected,
                "split {split}"
            );
        }
        let mut parser = deepseek_v41_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: state,
                ..Default::default()
            })
            .unwrap();
        let mut output = UnifiedParserOutput::default();
        for ch in input.chars() {
            parser
                .parse_into(ch.encode_utf8(&mut [0; 4]), &mut output)
                .unwrap();
        }
        output.append(&mut parser.finish().unwrap());
        assert_eq!(output.assembled(), expected, "one character at a time");
    }

    #[test]
    fn deepseek_v41_registration() {
        assert_every_split(
            "hello 世界",
            UnifiedParserStartingState::None,
            vec![UnifiedEvent::Text {
                text: "hello 世界".into(),
            }],
        );
    }

    #[test]
    fn reasoning_transition_and_multiple_calls() {
        let input = concat!(
            "Check the tools.\n</think>\n\n<｜DSML｜ calls>\n",
            "<｜DSML｜ invoke name=\"weather\">\n",
            "<｜DSML｜ parameter name=\"city\" string=\"true\">東京 &amp; \"Paris\"\\\n</｜DSML｜ parameter>\n",
            "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n",
            "<｜DSML｜ parameter name=\"options\" string=\"false\">{\"x\":[true,null]}</｜DSML｜ parameter>\n",
            "</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"done\">\n</｜DSML｜ invoke>\n",
            "</｜DSML｜ calls>",
        );
        assert_every_split(
            input,
            UnifiedParserStartingState::Reasoning,
            vec![
                UnifiedEvent::Reasoning {
                    text: "Check the tools.".into(),
                },
                UnifiedEvent::ToolCall {
                    name: "weather".into(),
                    arguments: serde_json::json!({
                        "city": "東京 &amp; \"Paris\"\\\n", "days": 3, "options": {"x": [true, null]}
                    }),
                },
                UnifiedEvent::ToolCall {
                    name: "done".into(),
                    arguments: serde_json::json!({}),
                },
            ],
        );
    }

    #[test]
    fn whitespace_follows_vllm_parser_engine() {
        let call = "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"></｜DSML｜ invoke></｜DSML｜ calls>";
        let run = || UnifiedEvent::ToolCall {
            name: "run".into(),
            arguments: serde_json::json!({}),
        };
        let reasoning = |text: &str| UnifiedEvent::Reasoning { text: text.into() };
        let text = |text: &str| UnifiedEvent::Text { text: text.into() };
        for (input, expected) in [
            (format!("plan \n</think>\n\n{call}"), vec![reasoning("plan"), run()]),
            (format!("a \n b\t</think>Sure.\n\n{call}"), vec![reasoning("a \n b"), text("Sure.\n\n"), run()]),
            ("plan</think>\n\n".to_string(), vec![reasoning("plan"), text("\n\n")]),
            (
                "plan</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"run\">\n<｜DSML｜ parameter name=\"a\" string=\"true\">x".to_string(),
                vec![reasoning("plan"), run()],
            ),
            ("plan</think>\n\n<｜DSML｜ calls>\n</｜DSML｜ calls>".to_string(), vec![reasoning("plan"), text("\n\n")]),
            ("plan</think> \n answer".to_string(), vec![reasoning("plan"), text(" \n answer")]),
            ("r</think>Use <think> tags </think> ok".to_string(), vec![reasoning("r"), text("Use "), reasoning(" tags"), text(" ok")]),
        ] {
            assert_every_split(&input, UnifiedParserStartingState::Reasoning, expected);
        }
    }

    fn assert_every_split_with(
        make: fn(&[Tool]) -> Box<dyn UnifiedParser>,
        input: &str,
        state: UnifiedParserStartingState,
        expected: Vec<UnifiedEvent>,
    ) {
        assert_every_split_filtered(make, input, state, expected, false);
    }

    fn assert_every_split_filtered(
        make: fn(&[Tool]) -> Box<dyn UnifiedParser>,
        input: &str,
        state: UnifiedParserStartingState,
        expected: Vec<UnifiedEvent>,
        reasoning_only: bool,
    ) {
        for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
            let mut parser = make(&[]);
            parser
                .initialize_request(UnifiedParserInit {
                    starting_state: state,
                    ..Default::default()
                })
                .unwrap();
            let mut output = UnifiedParserOutput::default();
            parser.parse_into(&input[..split], &mut output).unwrap();
            parser.parse_into(&input[split..], &mut output).unwrap();
            output.append(&mut parser.finish().unwrap());
            let restored: Vec<UnifiedEvent> = output
                .assembled()
                .into_iter()
                .map(|event| match event {
                    UnifiedEvent::Text { text } => UnifiedEvent::Text {
                        text: restore_token_markers(&text).into_owned(),
                    },
                    UnifiedEvent::Reasoning { text } => UnifiedEvent::Reasoning {
                        text: restore_token_markers(&text).into_owned(),
                    },
                    call => call,
                })
                .collect();
            let restored: Vec<UnifiedEvent> = if reasoning_only {
                restored
                    .into_iter()
                    .filter(|event| matches!(event, UnifiedEvent::Reasoning { .. }))
                    .collect()
            } else {
                restored
            };
            assert_eq!(restored, expected, "split {split} of {input:?}");
        }
    }

    #[test]
    fn token_markers_open_reasoning_from_content_like_vllm() {
        // think_opener_in_content_not_honored: CONTENT + THINK_START -> REASONING, and a
        // bare THINK_END is absorbed, in a turn whose prompt closed reasoning.
        let (s, e) = (THINK_START_TOKEN_MARKER, THINK_END_TOKEN_MARKER);
        let reasoning = |t: &str| UnifiedEvent::Reasoning { text: t.into() };
        let text = |t: &str| UnifiedEvent::Text { text: t.into() };
        for (input, expected) in [
            (format!("A.{s}R{e}B"), vec![text("A."), reasoning("R"), text("B")]),
            (format!("{s}R{e}B"), vec![reasoning("R"), text("B")]),
            (format!("A{e}B"), vec![text("AB")]),
            (format!("A{s}R"), vec![text("A"), reasoning("R")]),
        ] {
            assert_every_split_with(
                deepseek_v41_unified_token_markers,
                &input,
                UnifiedParserStartingState::None,
                expected,
            );
        }
    }

    #[test]
    fn plain_text_markers_are_text_when_markers_are_token_ids() {
        // plaintext_marker_vs_special_token: marker text spelled with ordinary tokens
        // stays where it was written; only the special tokens switch channels.
        let (s, e) = (THINK_START_TOKEN_MARKER, THINK_END_TOKEN_MARKER);
        let reasoning = |t: &str| UnifiedEvent::Reasoning { text: t.into() };
        let text = |t: &str| UnifiedEvent::Text { text: t.into() };
        assert_every_split_with(
            deepseek_v41_unified_token_markers,
            "R</think>still R",
            UnifiedParserStartingState::Reasoning,
            vec![reasoning("R</think>still R")],
        );
        assert_every_split_with(
            deepseek_v41_unified_token_markers,
            &format!("<think>R{e}A</think>B"),
            UnifiedParserStartingState::Reasoning,
            vec![reasoning("<think>R"), text("A</think>B")],
        );
        assert_every_split_with(
            deepseek_v41_unified_token_markers,
            &format!("A<think>B{s}R"),
            UnifiedParserStartingState::None,
            vec![text("A<think>B"), reasoning("R")],
        );
    }

    #[test]
    fn two_pass_reads_markers_inside_a_calls_block_like_vllm_batch() {
        // prod_quirk_think_token_inside_dsml_arg (non-streaming): vLLM's reasoning pass
        // runs before tool parsing, so `</think>` inside an argument is absorbed and
        // `<think>` opens reasoning that takes the rest of the block.
        let input = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"search\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">x </think> y <think> z</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        for state in [UnifiedParserStartingState::None, UnifiedParserStartingState::Reasoning] {
            assert_every_split_filtered(
                deepseek_v41_unified_two_pass,
                input,
                state,
                vec![UnifiedEvent::Reasoning {
                    text: " z</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>".into(),
                }],
                true,
            );
        }
    }

    #[test]
    fn two_pass_splits_reasoning_like_the_single_pass_for_plain_turns() {
        let reasoning = |t: &str| UnifiedEvent::Reasoning { text: t.into() };
        let text = |t: &str| UnifiedEvent::Text { text: t.into() };
        for (input, state, expected) in [
            ("R </think>A", UnifiedParserStartingState::Reasoning, vec![reasoning("R"), text("A")]),
            ("A<think>R</think>B", UnifiedParserStartingState::None, vec![text("A"), reasoning("R"), text("B")]),
            ("A</think>B", UnifiedParserStartingState::None, vec![text("AB")]),
            ("R<think>S", UnifiedParserStartingState::Reasoning, vec![reasoning("RS")]),
        ] {
            assert_every_split_with(deepseek_v41_unified_two_pass, input, state, expected);
        }
    }

    #[test]
    fn bare_think_end_is_absorbed_in_response_state_like_vllm() {
        let text = |text: &str| vec![UnifiedEvent::Text { text: text.into() }];
        for (input, expected) in [
            ("A.</think>B", text("A.B")),
            ("Answer: I.</think>To determine", text("Answer: I.To determine")),
            ("say \"x</think>y\" ok", text("say \"xy\" ok")),
            ("it's `a</think>b` and 'c'", text("it's `ab` and 'c'")),
            ("</think>lead", text("lead")),
            ("tail</think>", text("tail")),
            ("Use <think> literally", text("Use <think> literally")),
        ] {
            assert_every_split(input, UnifiedParserStartingState::Response, expected);
        }
    }

    #[test]
    fn think_end_inside_tool_arguments_stays_data_in_response_state() {
        let input = "x</think><｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\">a</think>b</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
        assert_every_split(
            input,
            UnifiedParserStartingState::Response,
            vec![
                UnifiedEvent::Text { text: "x".into() },
                UnifiedEvent::ToolCall {
                    name: "run".into(),
                    arguments: serde_json::json!({"text": "a</think>b"}),
                },
            ],
        );
    }

    #[test]
    fn tool_markup_inside_string_is_data() {
        let input = "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\"><think>quoted</think> <｜DSML｜ calls></｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
        assert_every_split(
            input,
            UnifiedParserStartingState::None,
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"text": "<think>quoted</think> <｜DSML｜ calls>"}),
            }],
        );
    }

    #[test]
    fn closing_markers_and_whitespace_inside_strings_are_data() {
        let value = " X</｜DSML｜ calls>Y</｜DSML｜ invoke>Z\n ";
        let input = format!(
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"user name\" string=\"true\">{value}</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"
        );
        assert_every_split(
            &input,
            UnifiedParserStartingState::None,
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"user name":value}),
            }],
        );
    }

    /// vLLM closes a call still open at finish with the parameters closed so far
    /// (`_dsml_arg_converter(partial=False)`); outputs recorded from the prod image.
    #[test]
    fn truncated_invoke_is_emitted_with_its_closed_parameters_like_vllm() {
        let call = |name: &str, arguments: serde_json::Value| UnifiedEvent::ToolCall {
            name: name.into(),
            arguments,
        };
        let text = |text: &str| UnifiedEvent::Text { text: text.into() };
        let invoke = "<｜DSML｜ invoke name=\"get_weather\">\n";
        let open = format!("<｜DSML｜ calls>\n{invoke}");
        let city = |v: &str| format!("<｜DSML｜ parameter name=\"city\" string=\"true\">{v}</｜DSML｜ parameter>");
        for (input, expected) in [
            (format!("Sure.\n\n{open}{}", city("Par")), vec![text("Sure.\n\n"), call("get_weather", serde_json::json!({"city": "Par"}))]),
            (format!("{open}<｜DSML｜ parameter name=\"city\" string=\"true\">Par"), vec![call("get_weather", serde_json::json!({}))]),
            (format!("{open}{}\n<｜DSML｜ parameter name=\"days\" string=\"false\">3", city("Paris")), vec![call("get_weather", serde_json::json!({"city": "Paris"}))]),
            (format!("{open}<｜DSML｜ parameter name=\"ci"), vec![call("get_weather", serde_json::json!({}))]),
            ("<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_wea".to_string(), vec![call("get_wea", serde_json::json!({}))]),
            (format!("{open}{}\n</｜DSML｜ inv", city("Lima")), vec![call("get_weather", serde_json::json!({"city": "Lima"}))]),
            (format!("Ok {invoke}{}", city("Bern")), vec![text("Ok "), call("get_weather", serde_json::json!({"city": "Bern"}))]),
            ("Text.\n\n<｜DSML｜ calls>\n".to_string(), vec![text("Text.\n\n")]),
        ] {
            assert_every_split(&input, UnifiedParserStartingState::Response, expected);
        }
    }

    /// vLLM's `TOOL_BETWEEN`: once a calls block opens, later text, reasoning and
    /// reasoning markers are dropped, while later blocks and invokes still parse.
    #[test]
    fn text_after_a_calls_block_is_dropped_like_vllm() {
        let paris = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        let rome = "<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">Rome</｜DSML｜ parameter>\n</｜DSML｜ invoke>";
        let call = |city: &str| UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": city}),
        };
        let states = [UnifiedParserStartingState::Response, UnifiedParserStartingState::Reasoning];
        for state in states {
            for (input, expected) in [
                (format!("{paris}\n\nDone. The answer is 42."), vec![call("Paris")]),
                (format!("{paris}</think>Answer"), vec![call("Paris")]),
                (format!("{paris}\nbetween\n<｜DSML｜ calls>\n{rome}\n</｜DSML｜ calls>\ntail"), vec![call("Paris"), call("Rome")]),
                (format!("{paris}\n{rome}\ntail"), vec![call("Paris"), call("Rome")]),
            ] {
                assert_every_split(&input, state, expected);
            }
        }
        // vLLM's non-streaming reasoning pass still honors a reasoning span opened after
        // the block (its 1-token stream drops it; see the tk3 group-U residuals).
        assert_every_split(
            &format!("{paris}<think>R</think>Answer"),
            UnifiedParserStartingState::Reasoning,
            vec![call("Paris"), UnifiedEvent::Reasoning { text: "R".into() }],
        );
        let lead = |state: UnifiedParserStartingState, text: &str| match state {
            UnifiedParserStartingState::Reasoning => UnifiedEvent::Reasoning { text: text.trim_end().into() },
            _ => UnifiedEvent::Text { text: text.into() },
        };
        for state in states {
            for (input, expected) in [
                ("Lead.\n\n<｜DSML｜ calls>\n</｜DSML｜ calls>\n\nTail text.".to_string(), vec![lead(state, "Lead.\n\n")]),
                ("Lead. <｜DSML｜ calls>junk</｜DSML｜ calls> tail".to_string(), vec![lead(state, "Lead. ")]),
                ("Write \"<｜DSML｜ calls>\" literally. Then more.".to_string(), vec![lead(state, "Write \"")]),
                (format!("Think.\n\n{paris}\nmore thinking</think>Answer"), vec![lead(state, "Think.\n\n"), call("Paris")]),
            ] {
                assert_every_split(&input, state, expected);
            }
        }
    }

    /// A bare invoke inside reasoning is reasoning text in vLLM (no REASONING +
    /// INVOKE_PREFIX transition); outside reasoning it is a call.
    #[test]
    fn bare_invoke_in_reasoning_stays_reasoning_like_vllm() {
        let invoke = "<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">Bern</｜DSML｜ parameter>\n</｜DSML｜ invoke>";
        let input = format!("Ok {invoke} after");
        assert_every_split(
            &input,
            UnifiedParserStartingState::Reasoning,
            vec![UnifiedEvent::Reasoning { text: input.clone() }],
        );
        assert_every_split(
            &input,
            UnifiedParserStartingState::Response,
            vec![
                UnifiedEvent::Text { text: "Ok ".into() },
                UnifiedEvent::ToolCall {
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city": "Bern"}),
                },
            ],
        );
    }

    /// `tool_choice: "none"`: vLLM suppresses call events, so the whitespace before a
    /// block is kept and nothing after it surfaces.
    #[test]
    fn suppressed_calls_keep_leading_whitespace_like_vllm() {
        let block = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        let text = |text: &str| UnifiedEvent::Text { text: text.into() };
        for (input, state, expected) in [
            (format!("\n\n{block}\ntail"), UnifiedParserStartingState::Response, vec![text("\n\n")]),
            (format!("R</think>\n\n{block}"), UnifiedParserStartingState::Reasoning, vec![UnifiedEvent::Reasoning { text: "R".into() }, text("\n\n")]),
            (format!("Sure.\n\n{block}"), UnifiedParserStartingState::Response, vec![text("Sure.\n\n")]),
        ] {
            for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
                let mut parser = deepseek_v41_unified(&[]);
                parser
                    .initialize_request(UnifiedParserInit {
                        starting_state: state,
                        ..Default::default()
                    })
                    .unwrap();
                parser.suppress_tool_calls();
                let mut output = UnifiedParserOutput::default();
                parser.parse_into(&input[..split], &mut output).unwrap();
                parser.parse_into(&input[split..], &mut output).unwrap();
                output.append(&mut parser.finish().unwrap());
                assert_eq!(output.assembled(), expected, "{input:?} split {split}");
            }
        }
    }

    /// Arguments reach the client as vLLM builds them: schema coercion, wrapper
    /// unwrapping, Python `json.dumps` text.
    #[test]
    fn arguments_follow_the_request_tool_schema() {
        let tools = vec![Tool {
            name: "send_money".into(),
            description: None,
            parameters: serde_json::json!({"type": "object", "properties": {
                "recipient": {"type": "string"}, "amount": {"type": "number"}}}),
            strict: None,
        }];
        let first_call = |input: &str| {
            let mut parser = deepseek_v41_unified(&tools);
            let events = parser.push(input).unwrap();
            let Some(crate::unified::UnifiedParserEvent::ToolCall(call)) = events.into_iter().next() else {
                panic!("expected a tool call for {input:?}");
            };
            call.arguments
        };
        assert_eq!(
            first_call("<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"send_money\">\n<｜DSML｜ parameter name=\"recipient\" string=\"true\">GB29</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"amount\" string=\"false\">4.0</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"),
            r#"{"recipient": "GB29", "amount": 4}"#
        );
        assert_eq!(
            first_call("<｜DSML｜ calls><｜DSML｜ invoke name=\"send_money\"><｜DSML｜ parameter name=\"arguments\" string=\"false\">{\"amount\": 10.0}</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"),
            r#"{"amount": 10}"#
        );
    }

    #[test]
    fn partial_plain_markers_and_open_reasoning_survive_eof() {
        for input in ["hello <", "hello <｜DS", "ordinary text\n", "你好"] {
            assert_every_split(
                input,
                UnifiedParserStartingState::None,
                vec![UnifiedEvent::Text { text: input.into() }],
            );
            assert_every_split(
                input,
                UnifiedParserStartingState::Reasoning,
                vec![UnifiedEvent::Reasoning {
                    text: input.trim_end().into(),
                }],
            );
        }
    }

    #[test]
    fn calls_stream_at_each_invocation_close() {
        let mut parser = deepseek_v41_unified(&[]);
        assert!(parser.push("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\">hello").unwrap().is_empty());
        let events = parser
            .push("</｜DSML｜ parameter></｜DSML｜ invoke>")
            .unwrap();
        let output: UnifiedParserOutput = events.into_iter().collect();
        assert_eq!(
            output.assembled(),
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"text":"hello"}),
            }]
        );
        assert!(parser.push("</｜DSML｜ calls>").unwrap().is_empty());
    }

    #[test]
    fn invocation_waits_for_its_complete_closing_tag() {
        for suffix in ["", " invoke", " banana>"] {
            let input = format!("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"></｜DSML｜{suffix}");
            let mut parser = deepseek_v41_unified(&[]);
            assert!(parser.push(&input).unwrap().is_empty());
            // An invoke still open at EOF is closed as vLLM does.
            assert_eq!(
                parser.finish().unwrap().assembled(),
                vec![UnifiedEvent::ToolCall {
                    name: "run".into(),
                    arguments: serde_json::json!({}),
                }]
            );
        }
    }

    #[test]
    fn invocation_boundary_scans_large_streamed_parameters_linearly() {
        let mut parser = deepseek_v41_unified(&[]);
        parser
            .push("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\">")
            .unwrap();
        BOUNDARY_EXAMINED_BYTES.with(|examined| examined.set(0));

        let value = "x".repeat(16 * 1024);
        for byte in value.as_bytes() {
            parser
                .push(std::str::from_utf8(std::slice::from_ref(byte)).unwrap())
                .unwrap();
        }
        let events = parser
            .push("</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>")
            .unwrap();
        let examined = BOUNDARY_EXAMINED_BYTES.with(std::cell::Cell::get);

        let output: UnifiedParserOutput = events.into_iter().collect();
        assert_eq!(
            output.assembled(),
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"text": value}),
            }]
        );
        assert!(
            examined < value.len() * PARAMETER_END.len() * 2,
            "boundary examined {examined} bytes for a {}-byte value",
            value.len()
        );
    }

    #[test]
    fn lenient_parameters_decode_like_vllm() {
        let call = |params: &str| {
            format!(
                "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\">{params}</｜DSML｜ invoke></｜DSML｜ calls>"
            )
        };
        for (params, expected) in [
            (
                "<｜DSML｜ parameter name=\"x\" string=\"true\">first</｜DSML｜ parameter><｜DSML｜ parameter name=\"y\" string=\"false\">2</｜DSML｜ parameter><｜DSML｜ parameter name=\"x\" string=\"true\">second</｜DSML｜ parameter>",
                serde_json::json!({"x": "second", "y": 2}),
            ),
            (
                "<｜DSML｜ parameter name=\"value\" string=\"false\">invalid</｜DSML｜ parameter>",
                serde_json::json!({"value": "invalid"}),
            ),
            (
                "<｜DSML｜ parameter name=\"value\" string=\"maybe\">1</｜DSML｜ parameter>",
                serde_json::json!({"value": 1}),
            ),
            (
                "<｜DSML｜ parameter name=\"value\">[1, 2.5e-05]</｜DSML｜ parameter>",
                serde_json::json!({"value": [1, 2.5e-05]}),
            ),
        ] {
            assert_every_split(
                &call(params),
                UnifiedParserStartingState::None,
                vec![UnifiedEvent::ToolCall {
                    name: "run".into(),
                    arguments: expected,
                }],
            );
        }
        let mut parser = deepseek_v41_unified(&[]);
        let events = parser
            .push(&call("<｜DSML｜ parameter name=\"a\" string=\"false\">{\"k\":[1,0.00001,true]}</｜DSML｜ parameter><｜DSML｜ parameter name=\"b\" string=\"true\">東京</｜DSML｜ parameter>"))
            .unwrap();
        let Some(crate::unified::UnifiedParserEvent::ToolCall(call)) = events.first() else {
            panic!("expected a tool call, got {events:?}");
        };
        assert_eq!(call.arguments, r#"{"a": {"k": [1, 1e-05, true]}, "b": "東京"}"#);
    }

    /// vLLM's parameter regex needs a parameter close (or the next parameter's
    /// header), so an unclosed parameter before the invoke close is dropped.
    #[test]
    fn unclosed_parameter_before_the_invoke_close_is_dropped_like_vllm() {
        let input = "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"value\" string=\"false\">1</｜DSML｜ invoke></｜DSML｜ calls>";
        assert_every_split(
            input,
            UnifiedParserStartingState::None,
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({}),
            }],
        );
    }

    #[test]
    fn guided_output_uses_shared_decoder() {
        let mut parser = deepseek_v41_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Reasoning,
                tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("weather".into()),
                },
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            parser
                .parse_complete("check</think>{\"city\":\"Paris\"}")
                .unwrap(),
            vec![
                UnifiedEvent::Reasoning {
                    text: "check".into()
                },
                UnifiedEvent::ToolCall {
                    name: "weather".into(),
                    arguments: serde_json::json!({"city":"Paris"})
                },
            ]
        );
    }

    #[test]
    fn reset_restarts_tool_indices() {
        let input =
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"done\"></｜DSML｜ invoke></｜DSML｜ calls>";
        let mut parser = deepseek_v41_unified(&[]);
        let first = parser.push(input).unwrap();
        parser.finish().unwrap();
        assert!(parser.reset().is_empty());
        assert_eq!(parser.push(input).unwrap(), first);
    }
}
