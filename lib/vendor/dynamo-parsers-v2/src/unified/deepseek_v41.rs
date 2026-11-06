// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use serde_json::{Map, Value};

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

pub(crate) fn deepseek_v41_unified(_tools: &[Tool]) -> Box<dyn UnifiedParser> {
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
        ..Default::default()
    };
    let scanner = WrappedBlockScanner::new(spec, DeepSeekV41).with_reasoning(ReasoningSpec {
        start: "<think>",
        end: "</think>",
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
/// any visible text is held back, then dropped if a tool-call block follows, even
/// one truncated before its first call completes (kept at EOF otherwise).
struct VllmWhitespace {
    inner: Box<dyn UnifiedParser>,
    deferred_reasoning: String,
    deferred_text: String,
    content_has_text: bool,
    tool_seen: bool,
    block_started: bool,
    raw_tail: String,
}

impl VllmWhitespace {
    fn new(inner: Box<dyn UnifiedParser>) -> Self {
        Self {
            inner,
            deferred_reasoning: String::new(),
            deferred_text: String::new(),
            content_has_text: false,
            tool_seen: false,
            block_started: false,
            raw_tail: String::new(),
        }
    }

    fn watch_block_start(&mut self, delta: &str) {
        if self.block_started {
            return;
        }
        self.raw_tail.push_str(delta);
        if self.raw_tail.contains(BLOCK_START) {
            self.block_started = true;
            self.raw_tail.clear();
            return;
        }
        let mut keep = self.raw_tail.len().saturating_sub(BLOCK_START.len() - 1);
        while !self.raw_tail.is_char_boundary(keep) {
            keep -= 1;
        }
        self.raw_tail.drain(..keep);
    }

    fn clear(&mut self) {
        self.deferred_reasoning.clear();
        self.deferred_text.clear();
        self.content_has_text = false;
        self.tool_seen = false;
        self.block_started = false;
        self.raw_tail.clear();
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
        self.watch_block_start(delta);
        let mut raw = UnifiedParserOutput::default();
        let result = self.inner.parse_into(delta, &mut raw);
        self.shape(raw.events, output);
        result
    }

    fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
        let raw = self.inner.finish()?;
        let mut out = UnifiedParserOutput::default();
        self.shape(raw.events, &mut out);
        if !self.deferred_text.is_empty() && !self.tool_seen && !self.block_started {
            out.push_text(std::mem::take(&mut self.deferred_text));
        }
        Ok(out)
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

struct DeepSeekV41;

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
        anyhow::ensure!(!name.is_empty(), "empty DeepSeek V4.1 tool name");
        let mut body = body
            .strip_suffix(INVOKE_END)
            .context("incomplete DeepSeek V4.1 invocation")?;
        let mut arguments = Map::new();
        while !body.trim().is_empty() {
            let (name, string, value) = parameter_header(body.trim_start())
                .context("invalid DeepSeek V4.1 parameter header")?;
            let (raw, remainder) = value
                .split_once(PARAMETER_END)
                .context("incomplete DeepSeek V4.1 parameter")?;
            let value = if string {
                Value::String(raw.to_string())
            } else {
                serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
            };
            // A repeated name keeps its first position with the last value (a Python dict).
            arguments.insert(name.to_string(), value);
            body = remainder;
        }
        Ok(Some(ToolCallDelta {
            tool_index,
            name: Some(name.to_string()),
            arguments: python_json(&Value::Object(arguments)),
            complete: true,
        }))
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
                vec![reasoning("plan")],
            ),
            ("plan</think> \n answer".to_string(), vec![reasoning("plan"), text(" \n answer")]),
            ("r</think>Use <think> tags </think> ok".to_string(), vec![reasoning("r"), text("Use "), reasoning(" tags"), text(" ok")]),
        ] {
            assert_every_split(&input, UnifiedParserStartingState::Reasoning, expected);
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

    #[test]
    fn incomplete_arguments_do_not_emit_calls() {
        let input = "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\">unfinished";
        assert_every_split(input, UnifiedParserStartingState::None, vec![]);
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
    fn invocation_requires_its_complete_closing_tag() {
        for suffix in ["", " invoke", " banana>"] {
            let input = format!("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"></｜DSML｜{suffix}");
            let mut parser = deepseek_v41_unified(&[]);
            assert!(parser.push(&input).unwrap().is_empty());
            assert!(parser.finish().unwrap().events.is_empty());
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

    #[test]
    fn closed_malformed_parameter_is_an_error_at_eof() {
        for input in [
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"value\" string=\"false\">1</｜DSML｜ invoke></｜DSML｜ calls>",
        ] {
            for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
                let mut parser = deepseek_v41_unified(&[]);
                let mut output = UnifiedParserOutput::default();
                let result = parser
                    .parse_into(&input[..split], &mut output)
                    .and_then(|()| parser.parse_into(&input[split..], &mut output))
                    .and_then(|()| parser.finish().map(|_| ()));
                assert!(result.is_err(), "split {split}");
                assert!(output.events.is_empty(), "split {split}");
            }
        }
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
