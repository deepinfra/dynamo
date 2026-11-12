// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DI: DeepSeek V4.1 tool-call structural tags, as vLLM builds them for `--tool-strict-level=function`.
//!
//! vLLM's `structural_tag_registry.get_model_structural_tag("deepseek_v4_1", ..., reasoning=False,
//! strict_level=FUNCTION)` constrains the DSML tool-call envelope of every request with tools: `auto` may answer in
//! text or open a `<｜DSML｜ calls>` block, `required` must open one with at least one invoke, a named choice must
//! call that tool. Arguments are pinned to the tool's schema only for `strict: true` tools. The worker's reasoning
//! parser holds the grammar until `</think>`, so the tag covers the answer only. The goldens in the tests are that
//! function's output.

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use serde_json::{Value, json};

use crate::protocols::openai::xgrammar_schema;

const CALLS_TRIGGER: &str = "<｜DSML｜ calls>";
const CALLS_BEGIN: &str = "<｜DSML｜ calls>\n";
const CALLS_END: &str = "</｜DSML｜ calls>";
const CALLS_PREFIX: &str = "\n\n";
const INVOKE_END: &str = "</｜DSML｜ invoke>\n";
const ARGUMENT_STYLE: &str = "deepseek_v4_1_xml";

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
            "stop_after_first": false,
            "excludes": ["<think>", "</think>"],
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
    let opener = [CALLS_PREFIX, CALLS_BEGIN].concat();
    json!({
        "type": "sequence",
        "elements": [
            {"type": "const_string", "value": opener},
            content,
            {"type": "const_string", "value": CALLS_END},
        ],
    })
}

fn invokes(tools: &[ToolDefinition], stop_after_first: bool) -> Value {
    json!({
        "type": "tags_with_separator",
        "tags": tools.iter().map(invoke).collect::<Vec<_>>(),
        "separator": "",
        "at_least_one": true,
        "stop_after_first": stop_after_first,
    })
}

fn invoke(tool: &ToolDefinition) -> Value {
    let schema = if tool.strict == Some(true) {
        match &tool.parameters {
            Some(parameters) => xgrammar_schema::pinnable_or(parameters.clone(), || Value::Bool(true), &tool.name),
            None => json!({"type": "object", "properties": {}}),
        }
    } else {
        Value::Bool(true)
    };
    json!({
        "type": "tag",
        "begin": format!("<｜DSML｜ invoke name=\"{}\">\n", tool.name),
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
        serde_json::from_str(include_str!("deepseek_v41_tool_tag_golden.json")).unwrap()
    }

    fn tools() -> Vec<ToolDefinition> {
        golden()["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| ToolDefinition {
                name: tool["function"]["name"].as_str().unwrap().to_string(),
                parameters: tool["function"].get("parameters").cloned(),
                strict: tool["function"].get("strict").and_then(Value::as_bool),
            })
            .collect()
    }

    #[test]
    fn tags_match_vllm_for_every_tool_choice() {
        let golden = golden();
        let cases = [
            ("auto", ToolChoice::Auto),
            ("required", ToolChoice::Required),
            ("named", ToolChoice::Named("calculate".to_string())),
        ];
        for (label, choice) in cases {
            assert_eq!(structural_tag(&choice, &tools(), None).unwrap(), golden["cases"][label], "{label}");
        }
    }

    #[test]
    fn no_tag_without_tools_for_none_or_for_an_unknown_named_tool() {
        assert!(structural_tag(&ToolChoice::Required, &[], None).is_none());
        assert!(structural_tag(&ToolChoice::None, &tools(), None).is_none());
        assert!(structural_tag(&ToolChoice::Named("missing".to_string()), &tools(), None).is_none());
    }

    #[test]
    fn strict_tool_with_an_uncompilable_schema_leaves_its_arguments_free() {
        let tools = vec![
            ToolDefinition {
                name: "plot".to_string(),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"xy": {"type": "array", "items": [{"type": "number"}, {"type": "number"}]}},
                })),
                strict: Some(true),
            },
            ToolDefinition {
                name: "unit".to_string(),
                parameters: Some(json!({"type": "object", "properties": {"u": {"enum": ["c", "f"]}}})),
                strict: Some(true),
            },
        ];
        let tag = structural_tag(&ToolChoice::Required, &tools, None).unwrap();
        let invokes = &tag["format"]["elements"][1]["tags"];
        assert_eq!(invokes[0]["content"]["json_schema"], json!(true));
        assert_eq!(invokes[1]["content"]["json_schema"], tools[1].parameters.clone().unwrap());
    }

    #[test]
    fn parallel_tool_calls_false_stops_after_the_first_invoke() {
        let tag = structural_tag(&ToolChoice::Required, &tools(), Some(false)).unwrap();
        assert_eq!(tag["format"]["elements"][1]["stop_after_first"], json!(true));
    }
}
