// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DI: DeepSeek V4 tool-call structural tags, gated by `DI_DSV4_TOOL_STRUCTURAL_TAG=1` (default off).
//!
//! The tag is xgrammar's `deepseek_v4` builtin (`get_model_structural_tag("deepseek_v4", reasoning="disabled")`):
//! the DSML `<｜DSML｜tool_calls>` envelope, one `<｜DSML｜invoke name="..">` tag per declared tool, and each tool's
//! `parameters` as `deepseek_xml` content, which renders a string value raw under `string="true"` and every other value
//! as JSON under `string="false"`, as the V4 chat template does. `auto` leaves text free until the calls trigger,
//! `required` must open one calls block, a named choice must call that tool. Unlike the builtin, every tool's schema is
//! pinned (also `strict: false`), and free text may not hold a bare `<｜DSML｜invoke name=` outside a calls block,
//! which the V4 parser would otherwise recover as a call to any name. A schema xgrammar cannot compile is left
//! unconstrained for that tool only. The worker's reasoning parser holds the grammar until `</think>`, so reasoning
//! stays free. The golden in the tests is the builtin's output plus that one exclude.

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use serde_json::{Value, json};

use crate::protocols::openai::xgrammar_tag_schema;

pub(crate) const ENABLE_ENV: &str = "DI_DSV4_TOOL_STRUCTURAL_TAG";

const CALLS_TRIGGER: &str = "<｜DSML｜tool_calls>";
const CALLS_BEGIN: &str = "<｜DSML｜tool_calls>\n";
const CALLS_END: &str = "</｜DSML｜tool_calls>";
const CALLS_PREFIX: &str = "\n\n";
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

/// The `guided_decoding.structural_tag` for a request with tools; `None` when no tag applies (`tool_choice=none`,
/// no tools, or a named tool that is not among them, which the caller rejects).
pub(crate) fn structural_tag(
    tool_choice: &ToolChoice,
    tools: &[ToolDefinition],
    parallel_tool_calls: Option<bool>,
) -> Option<Value> {
    if tools.is_empty() {
        return None;
    }
    let stop_after_first = parallel_tool_calls == Some(false);
    let format = match tool_choice {
        ToolChoice::None => return None,
        ToolChoice::Auto => json!({
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
        }),
        ToolChoice::Required => calls_block(invokes(tools, stop_after_first)),
        ToolChoice::Named(name) => {
            let tool = tools.iter().find(|tool| &tool.name == name)?;
            calls_block(invoke(tool))
        }
    };
    Some(json!({"type": "structural_tag", "format": format}))
}

fn calls_block(content: Value) -> Value {
    json!({
        "type": "sequence",
        "elements": [
            {"type": "const_string", "value": ([CALLS_PREFIX, CALLS_BEGIN].concat())},
            content,
            {"type": "const_string", "value": CALLS_END},
        ],
    })
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

fn invoke(tool: &ToolDefinition) -> Value {
    let schema = match &tool.parameters {
        Some(parameters) if xgrammar_tag_schema::compiles_in_structural_tag(parameters) => {
            parameters.clone()
        }
        Some(_) => {
            tracing::debug!(tool = %tool.name, "tool schema not compilable by xgrammar; arguments left unconstrained");
            Value::Bool(true)
        }
        None => Value::Bool(true),
    };
    json!({
        "type": "tag",
        "begin": format!("<｜DSML｜invoke name=\"{}\">\n", tool.name),
        "content": {
            "type": "json_schema",
            "json_schema": schema,
            "style": ARGUMENT_STYLE,
            "any_order": false,
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

    #[test]
    fn tags_match_xgrammar_deepseek_v4_builtin() {
        let golden = golden();
        let cases = [
            ("auto", ToolChoice::Auto, None),
            ("auto_single_call", ToolChoice::Auto, Some(false)),
            ("required", ToolChoice::Required, None),
            ("named", ToolChoice::Named("create_task".to_string()), None),
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
        for choice in [ToolChoice::Auto, ToolChoice::Required] {
            let tag = structural_tag(&choice, &tools, None).unwrap();
            assert_eq!(invoke_schemas(&tag), schemas.to_vec());
        }
    }

    #[test]
    fn uncompilable_schema_frees_only_that_tools_arguments() {
        let good = json!({"type": "object", "properties": {"q": {"type": "string"}}});
        let tools = [
            tool(
                "tuple",
                Some(
                    json!({"type": "object", "properties": {"xy": {"type": "array", "items": [{"type": "number"}]}}}),
                ),
                None,
            ),
            tool(
                "lookahead",
                Some(
                    json!({"type": "object", "properties": {"pw": {"type": "string", "pattern": "^(?=.*[A-Z])"}}}),
                ),
                None,
            ),
            tool(
                "dangling",
                Some(json!({"type": "object", "properties": {"r": {"$ref": "#/$defs/missing"}}})),
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
        let tag = structural_tag(&ToolChoice::Required, &tools, None).unwrap();
        let schemas = invoke_schemas(&tag);
        assert_eq!(&schemas[..3], &[json!(true), json!(true), json!(true)]);
        assert_eq!(schemas[3], good);
        assert_eq!(
            schemas[4],
            tools[4].parameters.clone().unwrap(),
            "xgrammar compiles uniqueItems in a tag"
        );
    }

    #[test]
    fn no_tag_for_none_no_tools_or_unknown_named_tool() {
        assert!(structural_tag(&ToolChoice::Auto, &[], None).is_none());
        assert!(structural_tag(&ToolChoice::None, &golden_tools(), None).is_none());
        assert!(
            structural_tag(
                &ToolChoice::Named("missing".to_string()),
                &golden_tools(),
                None
            )
            .is_none()
        );
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
