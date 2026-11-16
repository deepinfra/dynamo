// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DI: DeepSeek V4 tool-call structural tag for `tool_choice=auto`, gated by `DI_DSV4_TOOL_STRUCTURAL_TAG=1`
//! (default off).
//!
//! The tag is xgrammar's `deepseek_v4` builtin (`get_model_structural_tag("deepseek_v4", reasoning="disabled",
//! any_order=True)`): text stays free until `<｜DSML｜tool_calls>`, then one `<｜DSML｜invoke name="..">` tag per
//! declared tool with the tool's `parameters` as `deepseek_xml` content, which renders a string value raw under
//! `string="true"` and every other value as JSON under `string="false"`, as the V4 chat template does. Parameters may
//! come in any order (the model often writes an optional one before a required one); xgrammar then no longer enforces
//! `required` or rejects a repeated parameter. Unlike the builtin, every tool's schema is pinned (also
//! `strict: false`), and free text may not hold a bare `<｜DSML｜invoke name=` outside a calls block, which the V4
//! parser would otherwise recover as a call to any name. Where xgrammar cannot compile a schema, only the offending
//! keywords are relaxed (`xgrammar_tag_schema::relax_for_structural_tag`, logged under target `dsv4_tag_relax`); a
//! tool whose relaxed schema still fails is left unconstrained. The worker's reasoning parser holds the grammar until
//! `</think>`, so reasoning stays free.
//! `required` and named choices keep the guided-JSON array, which streams arguments as they are generated. The golden
//! in the tests is the builtin's output plus that one exclude.
//!
//! With `DI_DSV4_TOOL_TAG_REQUIRED=1` as well, a tool whose schema is a plain object (see [`required_split`]) gets one
//! `json_schema` element per required top-level property, in schema order, then one element for the optional ones in
//! any order, so `required` is enforced without forcing an order on the optional arguments. The prompt then lists
//! that tool's properties required-first ([`prompt_tools`]), so the order the model reads is the order it must write.

use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, Mutex};

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_protocols::types::ChatCompletionRequestMessage;
use dynamo_renderer::{OAIChatLikeRequest, PromptInput, TextInput, TokenInput};
use minijinja::value::Value as JinjaValue;
use serde_json::{Map, Value, json};

use crate::protocols::openai::common_ext::CommonExtProvider;
use crate::protocols::openai::xgrammar_tag_schema;

pub(crate) const ENABLE_ENV: &str = "DI_DSV4_TOOL_STRUCTURAL_TAG";
pub(crate) const REQUIRED_ENV: &str = "DI_DSV4_TOOL_TAG_REQUIRED";
pub(crate) const EMPTY_USER_ENV: &str = "DI_DSV4_EMPTY_USER_TURN";
pub(crate) const TOOL_NONE_FLATTEN_ENV: &str = "DI_DSV4_TOOL_NONE_FLATTEN";
pub(crate) const FORCED_NO_WHITESPACE_ENV: &str = "DI_DSV4_FORCED_NO_WHITESPACE";

const CALLS_TRIGGER: &str = "<｜DSML｜tool_calls>";
const CALLS_BEGIN: &str = "<｜DSML｜tool_calls>\n";
const CALLS_END: &str = "</｜DSML｜tool_calls>";
const BARE_INVOKE: &str = "<｜DSML｜invoke name=";
const INVOKE_END: &str = "</｜DSML｜invoke>\n";
const ARGUMENT_STYLE: &str = "deepseek_xml";
const DSML_MARKER: &str = "｜DSML｜";

#[cfg(test)]
thread_local! {
    /// Tests flip the gates per thread instead of mutating the process environment.
    pub(crate) static GATE_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    pub(crate) static REQUIRED_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    pub(crate) static EMPTY_USER_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    pub(crate) static FLATTEN_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    pub(crate) static FORCED_WHITESPACE_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn gate_override() -> Option<bool> {
    GATE_FOR_TEST.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn gate_override() -> Option<bool> {
    None
}

#[cfg(test)]
fn required_override() -> Option<bool> {
    REQUIRED_FOR_TEST.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn required_override() -> Option<bool> {
    None
}

#[cfg(test)]
fn empty_user_override() -> Option<bool> {
    EMPTY_USER_FOR_TEST.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn empty_user_override() -> Option<bool> {
    None
}

#[cfg(test)]
fn flatten_override() -> Option<bool> {
    FLATTEN_FOR_TEST.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn flatten_override() -> Option<bool> {
    None
}

#[cfg(test)]
fn forced_whitespace_override() -> Option<bool> {
    FORCED_WHITESPACE_FOR_TEST.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn forced_whitespace_override() -> Option<bool> {
    None
}

/// Most consecutive whitespace characters the forced tool-call JSON may hold: room for `": "`, `", "` and a
/// pretty-printed newline plus indentation, never a run to `max_tokens`.
const FORCED_MAX_WHITESPACE: u64 = 16;

/// Largest forced-call schema (serialized JSON bytes) that gets the whitespace bound. Default for
/// `DI_DSV4_FORCED_NO_WHITESPACE_MAX_BYTES`.
///
/// xgrammar's per-token mask cost for the `json_schema` structural tag grows with schema size, while guided JSON stays
/// flat: offline on the DSV4 tokenizer, ~0.02 ms at a few KB but ~0.3 ms mean and ~1 ms worst at ~270 KB, against
/// ~0.01 ms for guided JSON. The mask is filled on the decode step's critical path for every draft position, and a
/// DP+EP decode pod runs its ranks in lock-step, so one forced request with a large tool set slowed a whole prod decode
/// pod ~8x. Small schemas keep the bound; larger ones keep guided JSON.
const FORCED_MAX_SCHEMA_BYTES_DEFAULT: usize = 16 * 1024;
pub(crate) const FORCED_MAX_SCHEMA_BYTES_ENV: &str = "DI_DSV4_FORCED_NO_WHITESPACE_MAX_BYTES";

fn forced_max_schema_bytes() -> usize {
    std::env::var(FORCED_MAX_SCHEMA_BYTES_ENV)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(FORCED_MAX_SCHEMA_BYTES_DEFAULT)
}

/// With `DI_DSV4_FORCED_NO_WHITESPACE` on and the deepseek_v4 parser, the forced (`required`/named) tool-call JSON
/// schema as a structural tag whose `json_schema` content bounds consecutive whitespace; `None` keeps guided JSON.
///
/// vLLM's xgrammar backend compiles guided JSON with `any_whitespace` from its server config only (the request's
/// `disable_any_whitespace`/`whitespace_pattern` are not read, and the dynamo worker forwards only the latter), so a
/// forced call can decode `[` and then whitespace until `length`. A structural tag carries `max_whitespace_cnt` per
/// request. The output is the same JSON array, so the reported guided-JSON constraint and the parsers stay as they
/// are. A schema the tag cannot compile, or one larger than `DI_DSV4_FORCED_NO_WHITESPACE_MAX_BYTES`, keeps guided JSON.
pub(crate) fn bounded_forced_json(tool_call_parser: Option<&str>, schema: &Value) -> Option<Value> {
    let on = forced_whitespace_override().unwrap_or_else(|| env_on(FORCED_NO_WHITESPACE_ENV));
    if !is_v4_parser(tool_call_parser)
        || !on
        || schema.to_string().len() > forced_max_schema_bytes()
        || !xgrammar_tag_schema::compiles_in_structural_tag(schema)
    {
        return None;
    }
    Some(json!({
        "type": "structural_tag",
        "format": {
            "type": "json_schema",
            "json_schema": schema,
            "style": "json",
            "any_order": false,
            "max_whitespace_cnt": FORCED_MAX_WHITESPACE,
            "excludes": [],
        },
    }))
}

fn env_on(name: &str) -> bool {
    matches!(std::env::var(name).as_deref(), Ok("1" | "true"))
}

fn gate_on() -> bool {
    gate_override().unwrap_or_else(|| env_on(ENABLE_ENV))
}

/// Whether the tag splits invoke content on `required`; only consulted where the tag itself applies.
fn required_split_on() -> bool {
    required_override().unwrap_or_else(|| env_on(REQUIRED_ENV))
}

/// Whether this frontend builds V4 tool tags: the `deepseek_v4` tool parser with the env gate on.
pub(crate) fn applies(tool_call_parser: Option<&str>) -> bool {
    is_v4_parser(tool_call_parser) && gate_on()
}

fn is_v4_parser(tool_call_parser: Option<&str>) -> bool {
    matches!(
        tool_call_parser,
        Some("deepseek_v4" | "deepseek-v4" | "deepseekv4")
    )
}

/// The `guided_decoding.structural_tag` for a `tool_choice=auto` request with tools; `None` for any other choice
/// or without tools.
pub(crate) fn structural_tag(
    tool_choice: &ToolChoice,
    tools: &[ToolDefinition],
    parallel_tool_calls: Option<bool>,
) -> Option<Value> {
    if tools.is_empty() || !matches!(tool_choice, ToolChoice::Auto) {
        return None;
    }
    let stop_after_first = parallel_tool_calls == Some(false);
    Some(json!({
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": [CALLS_TRIGGER],
            "tags": [{
                "type": "tag",
                "begin": CALLS_BEGIN,
                "content": invokes(tools, stop_after_first),
                "end": CALLS_END,
            }],
            "at_least_one": false,
            "stop_after_first": stop_after_first,
            "excludes": ["<think>", "</think>", BARE_INVOKE],
        },
    }))
}

fn invokes(tools: &[ToolDefinition], stop_after_first: bool) -> Value {
    json!({
        "type": "tags_with_separator",
        "tags": (tools.iter().map(invoke).collect::<Vec<_>>()),
        "separator": "",
        "at_least_one": true,
        "stop_after_first": stop_after_first,
    })
}

/// The tool's `parameters` as pinned in the tag: with mergeable `allOf`s flattened (xgrammar would otherwise accept
/// any value there), relaxed where xgrammar cannot compile them (or `true`), and with open objects that declare
/// properties closed (`xgrammar_tag_schema::close_declared_objects`; logged as `additionalProperties_closed`). Only
/// the grammar sees this schema; the prompt and response validation keep the one sent.
fn arguments_schema(tool_name: &str, parameters: Option<&Value>) -> Value {
    let Some(parameters) = parameters else {
        return Value::Bool(true);
    };
    let mut keywords = Vec::new();
    let flattened = xgrammar_tag_schema::flatten_all_of(parameters);
    let schema = if xgrammar_tag_schema::compiles_in_structural_tag(&flattened) {
        flattened
    } else {
        let (relaxed, relaxed_keywords) = xgrammar_tag_schema::relax_for_structural_tag(&flattened);
        keywords = relaxed_keywords;
        if relaxed == Value::Bool(true)
            || !xgrammar_tag_schema::compiles_in_structural_tag(&relaxed)
        {
            log_relaxation(tool_name, &keywords, true);
            return Value::Bool(true);
        }
        relaxed
    };
    let (closed, changed) = xgrammar_tag_schema::close_declared_objects(&schema);
    let schema = if changed && xgrammar_tag_schema::compiles_in_structural_tag(&closed) {
        keywords.push("additionalProperties_closed");
        closed
    } else {
        schema
    };
    if !keywords.is_empty() {
        log_relaxation(tool_name, &keywords, false);
    }
    schema
}

/// Distinct (tool, keywords) pairs logged per process; past this, relaxations are no longer logged.
const RELAX_LOG_CAPACITY: usize = 1000;

/// Logs a relaxation once per tool name and keyword set. Only names are logged, never schema values or arguments.
fn log_relaxation(tool: &str, keywords: &[&'static str], unconstrained: bool) {
    static SEEN: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(Default::default);
    let keywords = keywords.join(",");
    {
        let Ok(mut seen) = SEEN.lock() else {
            return;
        };
        if seen.len() >= RELAX_LOG_CAPACITY || !seen.insert(format!("{tool}\u{1f}{keywords}")) {
            return;
        }
    }
    tracing::info!(
        target: "dsv4_tag_relax",
        tool,
        keywords = %keywords,
        unconstrained,
        "adjusted a tool schema for the DSML structural tag (relaxed where xgrammar cannot compile it, or closed)"
    );
}

fn invoke(tool: &ToolDefinition) -> Value {
    let schema = arguments_schema(&tool.name, tool.parameters.as_ref());
    let split = tool
        .parameters
        .as_ref()
        .filter(|_| required_split_on())
        .and_then(|parameters| required_split(parameters, &schema));
    let content = match split {
        Some(split) => split.content,
        None => argument_content(schema),
    };
    json!({
        "type": "tag",
        "begin": format!("<｜DSML｜invoke name=\"{}\">\n", tool.name),
        "content": content,
        "end": INVOKE_END,
    })
}

fn argument_content(schema: Value) -> Value {
    json!({
        "type": "json_schema",
        "json_schema": schema,
        "style": ARGUMENT_STYLE,
        "any_order": true,
        "max_whitespace_cnt": null,
        "excludes": [],
    })
}

/// Root keys a split schema may hold. Anything else at the root (combinators, `$defs`, `patternProperties`,
/// property-count bounds, ...) keeps the single-schema content.
const SPLIT_ROOT_KEYS: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "description",
    "title",
    "$schema",
    "$comment",
    "strict",
    "examples",
    "default",
    "deprecated",
    "readOnly",
    "writeOnly",
];

/// A tool's invoke content split on `required`, and the top-level property order it pins.
struct RequiredSplit {
    order: Vec<String>,
    content: Value,
}

/// The split for a tool's raw `parameters` and its prepared (flattened, relaxed) schema, or `None` to keep the single
/// schema. Both the grammar and the prompt order are decided here, so they always agree.
///
/// Splits only a prepared root `{"type": "object", "properties": {..non-empty..}, "required": [..names in
/// properties..]}` with no `additionalProperties` other than `false`, no root key outside [`SPLIT_ROOT_KEYS`], and no
/// `$ref` anywhere (an element's root is not the tool's root). The raw schema must have the same top-level property
/// order and `required`, since the prompt is rendered from it.
fn required_split(parameters: &Value, prepared: &Value) -> Option<RequiredSplit> {
    let raw = parameters.as_object()?;
    let root = prepared.as_object()?;
    if root.get("type").and_then(Value::as_str) != Some("object")
        || !root
            .keys()
            .all(|key| SPLIT_ROOT_KEYS.contains(&key.as_str()))
        || !matches!(
            root.get("additionalProperties"),
            None | Some(Value::Bool(false))
        )
        || contains_ref(prepared)
    {
        return None;
    }
    let properties = root.get("properties")?.as_object()?;
    let required: Vec<&str> = root
        .get("required")?
        .as_array()?
        .iter()
        .map(Value::as_str)
        .collect::<Option<_>>()?;
    if properties.is_empty() || !required.iter().all(|name| properties.contains_key(*name)) {
        return None;
    }
    let raw_properties = raw.get("properties")?.as_object()?;
    if !raw_properties.keys().eq(properties.keys()) || raw.get("required") != root.get("required") {
        return None;
    }
    let (required, optional): (Vec<String>, Vec<String>) = properties
        .keys()
        .cloned()
        .partition(|name| required.contains(&name.as_str()));
    let element = |names: &[String], required: &[String]| {
        let properties: Map<String, Value> = names
            .iter()
            .map(|name| (name.clone(), properties[name].clone()))
            .collect();
        json!({
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false,
        })
    };
    let mut schemas: Vec<Value> = required
        .iter()
        .map(|name| element(std::slice::from_ref(name), std::slice::from_ref(name)))
        .collect();
    if !optional.is_empty() {
        schemas.push(element(&optional, &[]));
    }
    if !schemas
        .iter()
        .all(xgrammar_tag_schema::compiles_in_structural_tag)
    {
        return None;
    }
    let mut contents: Vec<Value> = schemas.into_iter().map(argument_content).collect();
    let content = if contents.len() == 1 {
        contents.remove(0)
    } else {
        json!({"type": "sequence", "elements": contents})
    };
    let order = required.into_iter().chain(optional).collect();
    Some(RequiredSplit { order, content })
}

/// The tag for a request that cannot call tools (no tools, or `tool_choice=none`) after a tool-calling history: free
/// text that never holds the DSML marker (opening or closing tags alike), which would leak into `content`.
pub(crate) fn dsml_ban_tag() -> Value {
    json!({"type": "structural_tag", "format": {"type": "any_text", "excludes": [DSML_MARKER]}})
}

/// Whether the conversation holds an assistant tool call or a tool result.
pub(crate) fn history_has_tool_calls(messages: &[ChatCompletionRequestMessage]) -> bool {
    messages.iter().any(|message| match message {
        ChatCompletionRequestMessage::Assistant(assistant) => assistant
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty()),
        ChatCompletionRequestMessage::Tool(_) => true,
        _ => false,
    })
}

fn contains_ref(node: &Value) -> bool {
    match node {
        Value::Object(obj) => obj.contains_key("$ref") || obj.values().any(contains_ref),
        Value::Array(list) => list.iter().any(contains_ref),
        _ => false,
    }
}

/// The tools a V4 prompt renders, with each tool whose tag splits on `required` listing its properties
/// required-first. Built by [`prompt_tools`]; rendered through [`PromptToolsRequest`].
pub(crate) struct PromptTools {
    tools: Value,
    raw_tools: Value,
}

/// The reordered tools for a request whose DSML tag will split on `required`: the tag applies (parser and gate), the
/// required switch is on, and the request installs the `auto` tag (auto or unset `tool_choice`, no response format or
/// guided decoding). `None` renders the request's tools as they are.
pub(crate) fn prompt_tools<R: OAIChatLikeRequest + CommonExtProvider>(
    tool_call_parser: Option<&str>,
    request: &R,
) -> Option<PromptTools> {
    if !applies(tool_call_parser) || !required_split_on() {
        return None;
    }
    let auto = request
        .tool_choice()
        .is_none_or(|choice| choice.as_str() == Some("auto"));
    let constrained = request.get_guided_json().is_some()
        || request.get_guided_regex().is_some()
        || request.get_guided_grammar().is_some()
        || request
            .get_guided_choice()
            .is_some_and(|choice| !choice.is_empty());
    if !auto || constrained {
        return None;
    }
    let raw_tools = request.raw_tools()?;
    let tools = serde_json::to_value(request.tools()?).ok()?;
    let tools = required_first(&tools, &raw_tools)?;
    let raw_tools = required_first(&raw_tools, &raw_tools).unwrap_or(raw_tools);
    Some(PromptTools { tools, raw_tools })
}

/// `tools` with the properties of each tool that splits reordered as its split pins them, deciding from the raw tool
/// at the same index; `None` when nothing moves.
fn required_first(tools: &Value, raw_tools: &Value) -> Option<Value> {
    let (list, raw_list) = (tools.as_array()?, raw_tools.as_array()?);
    if list.len() != raw_list.len() {
        return None;
    }
    let mut out = list.clone();
    let mut changed = false;
    for (tool, raw_tool) in out.iter_mut().zip(raw_list) {
        let function = &raw_tool["function"];
        let Some(parameters) = function.get("parameters") else {
            continue;
        };
        let name = function["name"].as_str().unwrap_or_default();
        let prepared = arguments_schema(name, Some(parameters));
        let Some(split) = required_split(parameters, &prepared) else {
            continue;
        };
        let Some(properties) = tool
            .pointer_mut("/function/parameters/properties")
            .and_then(Value::as_object_mut)
        else {
            continue;
        };
        if split.order.len() != properties.len() || split.order.iter().eq(properties.keys()) {
            continue;
        }
        let reordered: Option<Map<String, Value>> = split
            .order
            .iter()
            .map(|name| {
                properties
                    .get(name)
                    .map(|value| (name.clone(), value.clone()))
            })
            .collect();
        if let Some(reordered) = reordered {
            *properties = reordered;
            changed = true;
        }
    }
    changed.then_some(Value::Array(out))
}

/// What a V4 prompt renders in place of the request's own fields: [`prompt_tools`], and the messages with the tool
/// history as plain text ([`flattened_tool_history`]) or with an empty user turn ([`empty_user_turn`]).
pub(crate) struct PromptOverride {
    tools: Option<PromptTools>,
    messages: Option<Value>,
}

/// The V4 prompt overrides for a request, or `None` to render it as is.
pub(crate) fn prompt_override<R: OAIChatLikeRequest + CommonExtProvider>(
    tool_call_parser: Option<&str>,
    request: &R,
) -> Option<PromptOverride> {
    let tools = prompt_tools(tool_call_parser, request);
    let messages = flattened_tool_history(tool_call_parser, request)
        .or_else(|| empty_user_turn(tool_call_parser, request));
    (tools.is_some() || messages.is_some()).then_some(PromptOverride { tools, messages })
}

/// With `DI_DSV4_EMPTY_USER_TURN` on, a V4 conversation of system messages only, rendered with a generation prompt,
/// gets an empty user message after them: the template then closes with `<｜User｜><｜Assistant｜>` instead of
/// leaving the model to continue the system text. A developer message already renders as a user turn, so any
/// non-system message keeps the conversation as is.
fn empty_user_turn<R: OAIChatLikeRequest>(
    tool_call_parser: Option<&str>,
    request: &R,
) -> Option<Value> {
    let on = empty_user_override().unwrap_or_else(|| env_on(EMPTY_USER_ENV));
    if !is_v4_parser(tool_call_parser) || !on || !request.should_add_generation_prompt() {
        return None;
    }
    let mut messages = serde_json::to_value(request.messages()).ok()?;
    let list = messages.as_array_mut()?;
    let system_only = !list.is_empty()
        && list
            .iter()
            .all(|message| message.get("role").and_then(Value::as_str) == Some("system"));
    if !system_only {
        return None;
    }
    list.push(json!({"role": "user", "content": ""}));
    Some(messages)
}

const TOOLS_UNAVAILABLE_NOTE: &str = "\n\n(Tools are not available for this reply. Answer the user now in plain text \
     using only the results above; if something is missing, say so.)";

/// With `DI_DSV4_TOOL_NONE_FLATTEN` on, a V4 request that cannot call tools (`tool_choice=none`, or no tools) after a
/// tool-calling history gets that history as plain text: the DSML calls the history would render teach the model to
/// keep calling. Each assistant tool call becomes a `[tool call] name(arguments)` line under the message's text, each
/// run of tool results one user message of `[tool result] name: content` lines, and when the conversation ends on
/// tool results they get [`TOOLS_UNAVAILABLE_NOTE`]. Other messages are kept as they are.
fn flattened_tool_history<R: OAIChatLikeRequest>(
    tool_call_parser: Option<&str>,
    request: &R,
) -> Option<Value> {
    let on = flatten_override().unwrap_or_else(|| env_on(TOOL_NONE_FLATTEN_ENV));
    let no_callable_tools = request
        .tool_choice()
        .is_some_and(|choice| choice.as_str() == Some("none"))
        || request
            .raw_tools()
            .is_none_or(|tools| tools.as_array().is_none_or(Vec::is_empty));
    if !is_v4_parser(tool_call_parser) || !on || !no_callable_tools {
        return None;
    }
    flatten_tool_history(&serde_json::to_value(request.messages()).ok()?)
}

fn role(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
}

fn tool_calls(message: &Value) -> Option<&Vec<Value>> {
    message
        .get("tool_calls")
        .and_then(Value::as_array)
        .filter(|calls| !calls.is_empty())
}

/// A message `content` as text: a string as is, text parts joined, anything else empty.
fn text_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// `messages` with assistant tool calls and tool results rewritten as plain text, or `None` without either.
fn flatten_tool_history(messages: &Value) -> Option<Value> {
    let list = messages.as_array()?;
    if !list
        .iter()
        .any(|message| tool_calls(message).is_some() || role(message) == Some("tool"))
    {
        return None;
    }
    let mut call_names: HashMap<String, String> = HashMap::new();
    let mut out: Vec<Value> = Vec::with_capacity(list.len());
    let mut last_is_result = false;
    for message in list {
        if let Some(calls) = tool_calls(message) {
            let text = text_content(message.get("content"));
            let mut lines: Vec<String> = Vec::new();
            if !text.is_empty() {
                lines.push(text);
            }
            for call in calls {
                let function = &call["function"];
                let name = function["name"].as_str().unwrap_or("tool");
                let arguments = match &function["arguments"] {
                    Value::String(arguments) => arguments.clone(),
                    Value::Null => String::new(),
                    other => other.to_string(),
                };
                if let Some(id) = call["id"].as_str() {
                    call_names.insert(id.to_string(), name.to_string());
                }
                lines.push(format!("[tool call] {name}({arguments})"));
            }
            let mut rewritten = message.clone();
            let fields = rewritten.as_object_mut()?;
            fields.remove("tool_calls");
            fields.insert("content".to_string(), Value::String(lines.join("\n")));
            out.push(rewritten);
            last_is_result = false;
        } else if role(message) == Some("tool") {
            let name = message["tool_call_id"]
                .as_str()
                .and_then(|id| call_names.get(id).cloned())
                .or_else(|| message["name"].as_str().map(str::to_string))
                .unwrap_or_else(|| "tool".to_string());
            let line = format!(
                "[tool result] {name}: {}",
                text_content(message.get("content"))
            );
            if last_is_result && let Some(previous) = out.last_mut() {
                let content = previous["content"].as_str().unwrap_or_default().to_string();
                previous["content"] = Value::String(format!("{content}\n{line}"));
            } else {
                out.push(json!({"role": "user", "content": line}));
            }
            last_is_result = true;
        } else {
            out.push(message.clone());
            last_is_result = false;
        }
    }
    if last_is_result && let Some(last) = out.last_mut() {
        let content = last["content"].as_str().unwrap_or_default().to_string();
        last["content"] = Value::String(content + TOOLS_UNAVAILABLE_NOTE);
    }
    Some(Value::Array(out))
}

/// A request rendered with a [`PromptOverride`]'s tools and messages in place of its own; every other method
/// delegates.
pub(crate) struct PromptToolsRequest<'a> {
    pub(crate) inner: &'a dyn OAIChatLikeRequest,
    pub(crate) tools: Option<&'a PromptTools>,
    pub(crate) messages: Option<&'a Value>,
}

impl<'a> PromptToolsRequest<'a> {
    pub(crate) fn new(inner: &'a dyn OAIChatLikeRequest, prompt: &'a PromptOverride) -> Self {
        Self {
            inner,
            tools: prompt.tools.as_ref(),
            messages: prompt.messages.as_ref(),
        }
    }
}

impl OAIChatLikeRequest for PromptToolsRequest<'_> {
    fn model(&self) -> String {
        self.inner.model()
    }

    fn messages(&self) -> JinjaValue {
        match self.messages {
            Some(messages) => JinjaValue::from_serialize(messages),
            None => self.inner.messages(),
        }
    }

    fn typed_messages(&self) -> Option<&[ChatCompletionRequestMessage]> {
        self.inner.typed_messages()
    }

    fn tools(&self) -> Option<JinjaValue> {
        match self.tools {
            Some(tools) => Some(JinjaValue::from_serialize(&tools.tools)),
            None => self.inner.tools(),
        }
    }

    fn raw_tools(&self) -> Option<Value> {
        match self.tools {
            Some(tools) => Some(tools.raw_tools.clone()),
            None => self.inner.raw_tools(),
        }
    }

    fn tool_choice(&self) -> Option<JinjaValue> {
        self.inner.tool_choice()
    }

    fn response_format(&self) -> Option<JinjaValue> {
        self.inner.response_format()
    }

    fn reasoning_effort(&self) -> Option<JinjaValue> {
        self.inner.reasoning_effort()
    }

    fn should_add_generation_prompt(&self) -> bool {
        self.inner.should_add_generation_prompt()
    }

    fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
        self.inner.chat_template_args()
    }

    fn prompt_input_type(&self) -> PromptInput {
        self.inner.prompt_input_type()
    }

    fn extract_tokens(&self) -> Option<TokenInput> {
        self.inner.extract_tokens()
    }

    fn extract_text(&self) -> Option<TextInput> {
        self.inner.extract_text()
    }

    fn mm_processor_kwargs(&self) -> Option<&Value> {
        self.inner.mm_processor_kwargs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn golden() -> Value {
        serde_json::from_str(include_str!("deepseek_v4_tool_tag_golden.json")).unwrap()
    }

    fn tool(name: &str, parameters: Option<Value>, strict: Option<bool>) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            parameters,
            strict,
        }
    }

    fn golden_tools() -> Vec<ToolDefinition> {
        golden()["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                let f = &t["function"];
                tool(
                    f["name"].as_str().unwrap(),
                    f.get("parameters").cloned(),
                    f.get("strict").and_then(Value::as_bool),
                )
            })
            .collect()
    }

    fn invoke_schemas(tag: &Value) -> Vec<Value> {
        let content = match tag["format"]["type"].as_str().unwrap() {
            "triggered_tags" => &tag["format"]["tags"][0]["content"],
            _ => &tag["format"]["elements"][1],
        };
        let invokes = match content["type"].as_str().unwrap() {
            "tags_with_separator" => content["tags"].as_array().unwrap().clone(),
            _ => vec![content.clone()],
        };
        invokes
            .iter()
            .map(|i| i["content"]["json_schema"].clone())
            .collect()
    }

    /// The golden is the builtin generated with `any_order=True`, where we deliberately differ from its default.
    #[test]
    fn tags_match_xgrammar_deepseek_v4_builtin() {
        let golden = golden();
        let cases = [
            ("auto", ToolChoice::Auto, None),
            ("auto_single_call", ToolChoice::Auto, Some(false)),
        ];
        for (label, choice, parallel) in cases {
            assert_eq!(
                structural_tag(&choice, &golden_tools(), parallel).unwrap(),
                golden["cases"][label],
                "{label}"
            );
        }
    }

    #[test]
    fn every_tool_schema_is_pinned_not_only_strict() {
        let parameters =
            json!({"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]});
        let tools = [
            tool("unset", Some(parameters.clone()), None),
            tool("lax", Some(parameters.clone()), Some(false)),
            tool("strict", Some(parameters.clone()), Some(true)),
        ];
        let tag = structural_tag(&ToolChoice::Auto, &tools, None).unwrap();
        assert_eq!(invoke_schemas(&tag), vec![parameters.clone(); 3]);
    }

    #[test]
    fn representative_schemas_are_pinned_verbatim() {
        let schemas = [
            json!({"type": "object", "properties": {"n": {"type": "integer", "minimum": 0}}}),
            json!({"type": "object", "properties": {"x": {"type": "number"}, "ok": {"type": "boolean"}}}),
            json!({"type": "object", "properties": {"unit": {"type": "string", "enum": ["c", "f"]}}}),
            json!({"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "integer"}}}}),
            json!({"type": "object", "properties": {"o": {"type": "object", "properties": {"k": {"type": "string"}}}}}),
            json!({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": false}),
            json!({"type": "object", "properties": {"opt": {"type": ["string", "null"]}}, "required": []}),
            json!({"type": "object", "properties": {"r": {"$ref": "#/$defs/R"}}, "$defs": {"R": {"type": "string"}}}),
        ];
        let tools: Vec<_> = schemas
            .iter()
            .enumerate()
            .map(|(i, s)| tool(&format!("t{i}"), Some(s.clone()), None))
            .collect();
        let tag = structural_tag(&ToolChoice::Auto, &tools, None).unwrap();
        assert_eq!(invoke_schemas(&tag), schemas.to_vec());
    }

    #[test]
    fn uncompilable_keywords_are_relaxed_not_the_whole_tool() {
        let good = json!({"type": "object", "properties": {"q": {"type": "string"}}});
        let int = json!({"type": "integer"});
        let tools = [
            tool(
                "tuple",
                Some(json!({"type": "object", "properties": {
                    "xy": {"type": "array", "items": [{"type": "number"}]}, "n": int,
                }})),
                None,
            ),
            tool(
                "lookahead",
                Some(json!({"type": "object", "properties": {
                    "pw": {"type": "string", "pattern": "^(?=.*[A-Z])"}, "n": int,
                }})),
                None,
            ),
            tool(
                "dangling",
                Some(json!({"type": "object", "properties": {
                    "r": {"$ref": "#/$defs/missing"}, "n": int,
                }})),
                None,
            ),
            tool("good", Some(good.clone()), None),
            tool(
                "unique",
                Some(
                    json!({"type": "object", "properties": {"l": {"type": "array", "uniqueItems": true}}}),
                ),
                None,
            ),
        ];
        let tag = structural_tag(&ToolChoice::Auto, &tools, None).unwrap();
        let schemas = invoke_schemas(&tag);
        // Only the offending keyword or subschema goes; the integer sibling stays pinned.
        assert_eq!(
            schemas[0],
            json!({"type": "object", "properties": {"xy": {"type": "array", "items": true}, "n": int}})
        );
        assert_eq!(
            schemas[1],
            json!({"type": "object", "properties": {"pw": {"type": "string"}, "n": int}})
        );
        assert_eq!(
            schemas[2],
            json!({"type": "object", "properties": {"r": true, "n": int}})
        );
        assert_eq!(schemas[3], good);
        assert_eq!(
            schemas[4],
            tools[4].parameters.clone().unwrap(),
            "xgrammar compiles uniqueItems in a tag"
        );
    }

    #[test]
    fn all_of_parameters_are_pinned_flattened() {
        let tools = [tool(
            "count",
            Some(json!({"type": "object", "properties": {
                "n": {"allOf": [{"type": "integer"}, {"minimum": 0}], "description": "how many"},
            }})),
            None,
        )];
        let tag = structural_tag(&ToolChoice::Auto, &tools, None).unwrap();
        assert_eq!(
            invoke_schemas(&tag),
            vec![json!({"type": "object", "properties": {
                "n": {"type": "integer", "minimum": 0, "description": "how many"},
            }})]
        );
    }

    #[test]
    fn a_tool_still_uncompilable_after_relaxing_is_left_unconstrained() {
        // Relaxing the tuple `items` breaks the `$ref` that pointed into it.
        let parameters = json!({"type": "object", "properties": {
            "a": {"$ref": "#/properties/b/items/0"},
            "b": {"type": "array", "items": [{"type": "string"}]},
        }});
        let tools = [tool("broken", Some(parameters), None)];
        let tag = structural_tag(&ToolChoice::Auto, &tools, None).unwrap();
        assert_eq!(invoke_schemas(&tag), vec![json!(true)]);
    }

    #[test]
    fn no_tag_unless_auto_with_tools() {
        assert!(structural_tag(&ToolChoice::Auto, &[], None).is_none());
        for choice in [
            ToolChoice::None,
            ToolChoice::Required,
            ToolChoice::Named("create_task".to_string()),
        ] {
            assert!(
                structural_tag(&choice, &golden_tools(), None).is_none(),
                "{choice:?}"
            );
        }
    }

    #[test]
    fn only_the_deepseek_v4_parser_with_the_gate_on() {
        if std::env::var(ENABLE_ENV).is_err() {
            assert!(!applies(Some("deepseek_v4")), "default off");
        }
        GATE_FOR_TEST.with(|gate| gate.set(Some(true)));
        for parser in ["deepseek_v4", "deepseek-v4", "deepseekv4"] {
            assert!(applies(Some(parser)), "{parser}");
        }
        for parser in [
            Some("deepseek_v41"),
            Some("deepseek_v3_2"),
            Some("qwen3_coder"),
            Some("mimo"),
            None,
        ] {
            assert!(!applies(parser), "{parser:?}");
        }
        GATE_FOR_TEST.with(|gate| gate.set(Some(false)));
        assert!(!applies(Some("deepseek_v4")));
        GATE_FOR_TEST.with(|gate| gate.set(None));
    }

    mod required_split {
        use super::*;
        use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
        use dynamo_renderer::OAIPromptFormatter;
        use dynamo_renderer::deepseek::v4::DeepSeekV4Formatter;

        /// Runs `f` with the tag gate and the required switch set for this thread.
        fn with_gates<T>(tag: bool, required: bool, f: impl FnOnce() -> T) -> T {
            GATE_FOR_TEST.with(|gate| gate.set(Some(tag)));
            REQUIRED_FOR_TEST.with(|gate| gate.set(Some(required)));
            let out = f();
            GATE_FOR_TEST.with(|gate| gate.set(None));
            REQUIRED_FOR_TEST.with(|gate| gate.set(None));
            out
        }

        fn mixed() -> Value {
            json!({
                "type": "object",
                "properties": {
                    "issue_id": {"type": "string"},
                    "text": {"type": "string"},
                    "n": {"type": "integer"},
                    "author": {"type": "string"},
                },
                "required": ["issue_id", "n"],
            })
        }

        fn element(properties: Value, required: Value) -> Value {
            argument_content(json!({
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": false,
            }))
        }

        fn invoke_content(parameters: Value, required: bool) -> Value {
            let tools = [tool("post", Some(parameters), None)];
            let tag = with_gates(true, required, || {
                structural_tag(&ToolChoice::Auto, &tools, None)
            })
            .unwrap();
            tag["format"]["tags"][0]["content"]["tags"][0]["content"].clone()
        }

        #[test]
        fn mixed_schema_splits_required_first_then_optional() {
            let content = invoke_content(mixed(), true);
            assert_eq!(
                content,
                json!({"type": "sequence", "elements": [
                    element(json!({"issue_id": {"type": "string"}}), json!(["issue_id"])),
                    element(json!({"n": {"type": "integer"}}), json!(["n"])),
                    element(
                        json!({"text": {"type": "string"}, "author": {"type": "string"}}),
                        json!([])
                    ),
                ]})
            );
            let optional = &content["elements"][2]["json_schema"]["properties"];
            let order: Vec<&String> = optional.as_object().unwrap().keys().collect();
            assert_eq!(order, ["text", "author"]);
        }

        #[test]
        fn a_single_element_is_used_directly() {
            let all_required = json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]});
            assert_eq!(
                invoke_content(all_required, true),
                element(json!({"a": {"type": "string"}}), json!(["a"]))
            );
            let all_optional =
                json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": []});
            assert_eq!(
                invoke_content(all_optional, true),
                element(json!({"a": {"type": "string"}}), json!([]))
            );
        }

        #[test]
        fn excluded_root_shapes_keep_the_single_schema() {
            let base = mixed();
            let with = |key: &str, value: Value| {
                let mut schema = base.clone();
                schema[key] = value;
                schema
            };
            let without = |key: &str| {
                let mut schema = base.clone();
                schema.as_object_mut().unwrap().remove(key);
                schema
            };
            let excluded = [
                without("required"),
                without("type"),
                with("required", json!(["issue_id", "missing"])),
                with("properties", json!({})),
                with("patternProperties", json!({"^x": {"type": "string"}})),
                with("anyOf", json!([{"required": ["text"]}])),
                with("oneOf", json!([{"required": ["text"]}])),
                with(
                    "allOf",
                    json!([{"required": ["text"]}, {"type": "integer"}]),
                ),
                with("if", json!({"required": ["text"]})),
                with("not", json!({"required": ["text"]})),
                with("dependentRequired", json!({"text": ["author"]})),
                with("minProperties", json!(1)),
                with("maxProperties", json!(4)),
                with("propertyNames", json!({"maxLength": 9})),
                with("$defs", json!({"X": {"type": "string"}})),
                with(
                    "properties",
                    json!({"issue_id": {"$ref": "#/properties/n"}, "n": {"type": "integer"}}),
                ),
            ];
            for schema in excluded {
                assert_eq!(
                    invoke_content(schema.clone(), true),
                    argument_content(arguments_schema("post", Some(&schema))),
                    "{schema}"
                );
            }
        }

        #[test]
        fn an_open_root_is_closed_and_then_splits() {
            for open in [json!(true), json!({}), json!({"type": "string"})] {
                let mut schema = mixed();
                schema["additionalProperties"] = open.clone();
                assert_eq!(
                    invoke_content(schema, true),
                    invoke_content(mixed(), true),
                    "{open}"
                );
            }
        }

        #[test]
        fn open_objects_that_declare_properties_are_pinned_closed() {
            let schema = json!({
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "cfg": {
                        "type": "object",
                        "properties": {"a": {"type": "string"}},
                        "additionalProperties": true,
                    },
                    "meta": {"type": "object", "additionalProperties": true},
                },
                "required": ["status"],
                "additionalProperties": true,
            });
            let mut closed = schema.clone();
            closed["additionalProperties"] = json!(false);
            closed["properties"]["cfg"]["additionalProperties"] = json!(false);
            assert_eq!(invoke_content(schema, false), argument_content(closed));
        }

        #[test]
        fn required_switch_off_is_todays_tag() {
            let golden = golden();
            for (label, parallel) in [("auto", None), ("auto_single_call", Some(false))] {
                let tag = with_gates(true, false, || {
                    structural_tag(&ToolChoice::Auto, &golden_tools(), parallel)
                });
                assert_eq!(tag.unwrap(), golden["cases"][label], "{label}");
            }
            assert_eq!(
                invoke_content(mixed(), false),
                argument_content(mixed()),
                "switch off pins the schema whole"
            );
        }

        fn chat_request(extra: Value) -> NvCreateChatCompletionRequest {
            let mut body = json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "post", "parameters": mixed()}}],
            });
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            serde_json::from_value(body).unwrap()
        }

        fn rendered(request: &NvCreateChatCompletionRequest, tag: bool, required: bool) -> String {
            let tools = with_gates(tag, required, || prompt_tools(Some("deepseek_v4"), request));
            let formatter = DeepSeekV4Formatter::new_chat();
            match &tools {
                Some(tools) => formatter
                    .render(&PromptToolsRequest {
                        inner: request,
                        tools: Some(tools),
                        messages: None,
                    })
                    .unwrap(),
                None => formatter.render(request).unwrap(),
            }
        }

        fn property_order(prompt: &str) -> Vec<&'static str> {
            let mut names = ["issue_id", "text", "n", "author"];
            names.sort_by_key(|name| {
                prompt
                    .find(&format!("\"{name}\": {{"))
                    .expect("property in the prompt")
            });
            names.to_vec()
        }

        #[test]
        fn prompt_lists_required_first_only_with_the_switch_on() {
            let request = chat_request(json!({}));
            let today = DeepSeekV4Formatter::new_chat().render(&request).unwrap();
            assert_eq!(property_order(&today), ["issue_id", "text", "n", "author"]);
            assert_eq!(rendered(&request, true, false), today, "switch off");
            assert_eq!(rendered(&request, false, true), today, "tag gate off");
            let on = rendered(&request, true, true);
            assert_eq!(property_order(&on), ["issue_id", "n", "text", "author"]);
            assert_eq!(on.len(), today.len(), "only the order changes");
        }

        #[test]
        fn prompt_keeps_its_order_when_no_auto_tag_is_installed() {
            for extra in [
                json!({"tool_choice": "required"}),
                json!({"tool_choice": "none"}),
                json!({"tool_choice": {"type": "function", "function": {"name": "post"}}}),
                json!({"response_format": {"type": "json_object"}}),
                json!({"guided_regex": "a+"}),
            ] {
                let request = chat_request(extra.clone());
                let today = DeepSeekV4Formatter::new_chat().render(&request).unwrap();
                assert_eq!(rendered(&request, true, true), today, "{extra}");
            }
            let text_format = chat_request(json!({"response_format": {"type": "text"}}));
            assert_eq!(
                property_order(&rendered(&text_format, true, true)),
                ["issue_id", "n", "text", "author"]
            );
        }
    }

    mod empty_user_turn {
        use super::*;
        use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
        use dynamo_renderer::OAIPromptFormatter;
        use dynamo_renderer::deepseek::v4::DeepSeekV4Formatter;

        const USER: &str = "<｜User｜>";
        const ASSISTANT: &str = "<｜Assistant｜>";
        const THINK_END: &str = "</think>";

        fn request(messages: Value) -> NvCreateChatCompletionRequest {
            serde_json::from_value(json!({"model": "m", "messages": messages})).unwrap()
        }

        fn rendered(request: &NvCreateChatCompletionRequest, parser: &str, on: bool) -> String {
            EMPTY_USER_FOR_TEST.with(|gate| gate.set(Some(on)));
            let prompt = prompt_override(Some(parser), request);
            EMPTY_USER_FOR_TEST.with(|gate| gate.set(None));
            let formatter = DeepSeekV4Formatter::new_chat();
            match &prompt {
                Some(prompt) => formatter
                    .render(&PromptToolsRequest::new(request, prompt))
                    .unwrap(),
                None => formatter.render(request).unwrap(),
            }
        }

        #[test]
        fn system_only_gets_an_empty_user_turn_before_the_assistant() {
            let request = request(json!([{"role": "system", "content": "You are terse."}]));
            let today = rendered(&request, "deepseek_v4", false);
            assert_eq!(
                today,
                DeepSeekV4Formatter::new_chat().render(&request).unwrap()
            );
            assert!(!today.contains(USER), "{today}");
            let on = rendered(&request, "deepseek_v4", true);
            assert!(
                on.ends_with(&format!("You are terse.{USER}{ASSISTANT}{THINK_END}")),
                "{on}"
            );
        }

        #[test]
        fn conversations_with_a_user_turn_or_other_parsers_are_untouched() {
            let unchanged = [
                json!([{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]),
                json!([{"role": "user", "content": "hi"}]),
                json!([{"role": "system", "content": "s"}, {"role": "developer", "content": "d"}]),
                json!([
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "weather?"},
                    {"role": "assistant", "content": "", "tool_calls": [{
                        "id": "c1", "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }]},
                    {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
                ]),
            ];
            for messages in unchanged {
                let request = request(messages.clone());
                assert_eq!(
                    rendered(&request, "deepseek_v4", true),
                    rendered(&request, "deepseek_v4", false),
                    "{messages}"
                );
            }
            let system_only = request(json!([{"role": "system", "content": "s"}]));
            for parser in ["deepseek_v41", "qwen3_coder", "glm47"] {
                EMPTY_USER_FOR_TEST.with(|gate| gate.set(Some(true)));
                assert!(
                    prompt_override(Some(parser), &system_only).is_none(),
                    "{parser}"
                );
                EMPTY_USER_FOR_TEST.with(|gate| gate.set(None));
            }
        }
    }

    mod tool_none_flatten {
        use super::*;
        use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
        use dynamo_renderer::OAIPromptFormatter;
        use dynamo_renderer::deepseek::v4::DeepSeekV4Formatter;

        const NOTE: &str = "\n\n(Tools are not available for this reply. Answer the user now in plain text using \
                            only the results above; if something is missing, say so.)";

        fn weather_tool() -> Value {
            json!([{"type": "function", "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }}])
        }

        fn history() -> Value {
            json!([
                {"role": "user", "content": "Weather in Paris and Rome?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}},
                    {"id": "c2", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\": \"Rome\"}"}},
                ]},
                {"role": "tool", "tool_call_id": "c1", "content": "18C sunny"},
                {"role": "tool", "tool_call_id": "c2", "content": "22C clear"},
            ])
        }

        fn request(messages: Value, extra: Value) -> NvCreateChatCompletionRequest {
            let mut body = json!({"model": "m", "messages": messages});
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            serde_json::from_value(body).unwrap()
        }

        fn overridden(
            request: &NvCreateChatCompletionRequest,
            parser: &str,
            on: bool,
        ) -> Option<Value> {
            FLATTEN_FOR_TEST.with(|gate| gate.set(Some(on)));
            let prompt = prompt_override(Some(parser), request);
            FLATTEN_FOR_TEST.with(|gate| gate.set(None));
            prompt.and_then(|prompt| prompt.messages)
        }

        fn rendered(request: &NvCreateChatCompletionRequest, on: bool) -> String {
            FLATTEN_FOR_TEST.with(|gate| gate.set(Some(on)));
            let prompt = prompt_override(Some("deepseek_v4"), request);
            FLATTEN_FOR_TEST.with(|gate| gate.set(None));
            let formatter = DeepSeekV4Formatter::new_chat();
            match &prompt {
                Some(prompt) => formatter
                    .render(&PromptToolsRequest::new(request, prompt))
                    .unwrap(),
                None => formatter.render(request).unwrap(),
            }
        }

        /// The (role, content) of each message, in order.
        fn turns(messages: &Value) -> Vec<(String, String)> {
            messages
                .as_array()
                .unwrap()
                .iter()
                .map(|m| {
                    (
                        m["role"].as_str().unwrap().to_string(),
                        m["content"].as_str().unwrap_or_default().to_string(),
                    )
                })
                .collect()
        }

        #[test]
        fn tool_history_becomes_plain_text_when_no_tools_are_callable() {
            let expected = vec![
                ("user".to_string(), "Weather in Paris and Rome?".to_string()),
                (
                    "assistant".to_string(),
                    "[tool call] get_weather({\"city\": \"Paris\"})\n[tool call] get_weather({\"city\": \"Rome\"})"
                        .to_string(),
                ),
                (
                    "user".to_string(),
                    format!("[tool result] get_weather: 18C sunny\n[tool result] get_weather: 22C clear{NOTE}"),
                ),
            ];
            for extra in [
                json!({}),
                json!({"tools": weather_tool(), "tool_choice": "none"}),
            ] {
                let request = request(history(), extra.clone());
                let messages = overridden(&request, "deepseek_v4", true).expect("flattened");
                assert_eq!(turns(&messages), expected, "{extra}");
                assert!(messages[1].get("tool_calls").is_none(), "{extra}");
                let prompt = rendered(&request, true);
                assert!(!prompt.contains("｜DSML｜"), "{prompt}");
                assert!(
                    prompt.contains("[tool result] get_weather: 18C sunny"),
                    "{prompt}"
                );
            }
        }

        #[test]
        fn results_followed_by_a_user_turn_get_no_note() {
            let mut messages = history();
            messages
                .as_array_mut()
                .unwrap()
                .push(json!({"role": "user", "content": "And tomorrow?"}));
            let request = request(messages, json!({}));
            let flattened = overridden(&request, "deepseek_v4", true).unwrap();
            let turns = turns(&flattened);
            assert_eq!(
                turns[2].1,
                "[tool result] get_weather: 18C sunny\n[tool result] get_weather: 22C clear"
            );
            assert_eq!(turns[3], ("user".to_string(), "And tomorrow?".to_string()));
        }

        #[test]
        fn a_result_names_its_call_then_its_name_then_tool() {
            let messages = json!([
                {"role": "user", "content": "q"},
                {"role": "tool", "tool_call_id": "unknown", "content": "a"},
                {"role": "user", "content": "q2"},
                {"role": "tool", "tool_call_id": "x", "content": "b"},
            ]);
            let request = request(messages, json!({}));
            let flattened = overridden(&request, "deepseek_v4", true).unwrap();
            assert_eq!(turns(&flattened)[1].1, "[tool result] tool: a");
            assert_eq!(
                turns(&flattened)[3].1,
                format!("[tool result] tool: b{NOTE}")
            );
        }

        #[test]
        fn untouched_without_tool_history_with_callable_tools_or_switch_off() {
            let plain = request(
                json!([{"role": "user", "content": "hi"}]),
                json!({"tool_choice": "none"}),
            );
            assert!(overridden(&plain, "deepseek_v4", true).is_none());
            for extra in [
                json!({"tools": weather_tool(), "tool_choice": "auto"}),
                json!({"tools": weather_tool()}),
                json!({"tools": weather_tool(), "tool_choice": "required"}),
            ] {
                let request = request(history(), extra.clone());
                assert!(
                    overridden(&request, "deepseek_v4", true).is_none(),
                    "{extra}"
                );
            }
            let request = request(history(), json!({}));
            assert_eq!(
                rendered(&request, false),
                DeepSeekV4Formatter::new_chat().render(&request).unwrap(),
                "switch off"
            );
            for parser in ["deepseek_v41", "qwen3_coder", "glm47"] {
                assert!(overridden(&request, parser, true).is_none(), "{parser}");
            }
        }
    }
}
