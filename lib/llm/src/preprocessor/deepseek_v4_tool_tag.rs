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

use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use serde_json::{Value, json};

use crate::protocols::openai::xgrammar_tag_schema;

pub(crate) const ENABLE_ENV: &str = "DI_DSV4_TOOL_STRUCTURAL_TAG";

const CALLS_TRIGGER: &str = "<｜DSML｜tool_calls>";
const CALLS_BEGIN: &str = "<｜DSML｜tool_calls>\n";
const CALLS_END: &str = "</｜DSML｜tool_calls>";
const BARE_INVOKE: &str = "<｜DSML｜invoke name=";
const INVOKE_END: &str = "</｜DSML｜invoke>\n";
const ARGUMENT_STYLE: &str = "deepseek_xml";

#[cfg(test)]
thread_local! {
    /// Tests flip the gate per thread instead of mutating the process environment.
    pub(crate) static GATE_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn gate_override() -> Option<bool> {
    GATE_FOR_TEST.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn gate_override() -> Option<bool> {
    None
}

fn gate_on() -> bool {
    gate_override()
        .unwrap_or_else(|| matches!(std::env::var(ENABLE_ENV).as_deref(), Ok("1" | "true")))
}

/// Whether this frontend builds V4 tool tags: the `deepseek_v4` tool parser with the env gate on.
pub(crate) fn applies(tool_call_parser: Option<&str>) -> bool {
    matches!(
        tool_call_parser,
        Some("deepseek_v4" | "deepseek-v4" | "deepseekv4")
    ) && gate_on()
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

/// The tool's `parameters` as pinned in the tag: as sent, relaxed where xgrammar cannot compile them, or `true`.
fn arguments_schema(tool: &ToolDefinition) -> Value {
    let Some(parameters) = &tool.parameters else {
        return Value::Bool(true);
    };
    if xgrammar_tag_schema::compiles_in_structural_tag(parameters) {
        return parameters.clone();
    }
    let (relaxed, keywords) = xgrammar_tag_schema::relax_for_structural_tag(parameters);
    let unconstrained =
        relaxed == Value::Bool(true) || !xgrammar_tag_schema::compiles_in_structural_tag(&relaxed);
    log_relaxation(&tool.name, &keywords, unconstrained);
    if unconstrained {
        Value::Bool(true)
    } else {
        relaxed
    }
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
        "relaxed a tool schema xgrammar cannot compile in the DSML structural tag"
    );
}

fn invoke(tool: &ToolDefinition) -> Value {
    let schema = arguments_schema(tool);
    json!({
        "type": "tag",
        "begin": format!("<｜DSML｜invoke name=\"{}\">\n", tool.name),
        "content": {
            "type": "json_schema",
            "json_schema": schema,
            "style": ARGUMENT_STYLE,
            "any_order": true,
            "max_whitespace_cnt": null,
            "excludes": [],
        },
        "end": INVOKE_END,
    })
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
}
