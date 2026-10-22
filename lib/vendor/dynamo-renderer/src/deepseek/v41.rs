// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Text prompt encoding for DeepSeek V4.1.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::common::{ThinkingMode, to_json};
use super::v4::{Encoding, encode_messages_with_encoding};

/// vLLM `deepseek_v41_encoding.REASONING_EFFORT_MAPPINGS`, the same table the
/// checkpoint's reference `encoding/encoding.py` ships.
const REASONING_EFFORT_MAPPINGS: [(&str, u8); 4] =
    [("low", 25), ("high", 50), ("xhigh", 75), ("max", 100)];
const DEFAULT_REASONING_EFFORT: u8 = 50;
const EFFORT_ERROR: &str = concat!(
    "DeepSeek V4.1 reasoning_effort must be low, high, xhigh, max, ",
    "or an integer within [1, 100] in chat_template_kwargs"
);

pub(super) fn find_last_user_index(messages: &[Value]) -> Option<usize> {
    messages.iter().enumerate().rposition(|(index, message)| {
        let role = message.get("role").and_then(Value::as_str);
        matches!(role, Some("user" | "developer")) || (role == Some("system") && index > 0)
    })
}

/// Before the last user turn, assistant reasoning is removed and every role
/// other than user/system/tool/latest_reminder/direct_search_results (in
/// practice: developer) is dropped, as in vLLM's `_drop_thinking_messages`.
pub(super) fn drop_thinking_messages(messages: Vec<Value>) -> Vec<Value> {
    let Some(last_user) = find_last_user_index(&messages) else {
        return messages;
    };
    messages
        .into_iter()
        .enumerate()
        .filter_map(|(index, mut message)| {
            let role = message.get("role").and_then(Value::as_str).unwrap_or("");
            let keep = matches!(
                role,
                "user" | "system" | "tool" | "latest_reminder" | "direct_search_results"
            );
            if keep || index >= last_user {
                Some(message)
            } else if role == "assistant" {
                message.as_object_mut().unwrap().remove("reasoning_content");
                Some(message)
            } else {
                None
            }
        })
        .collect()
}

fn python_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// vLLM's `FunctionDefinition.model_dump()`: name, description, parameters
/// (both null when absent), then strict/defer_loading only when set.
fn vllm_tool(tool: &Value) -> Option<Value> {
    let function = tool.get("function")?.as_object()?;
    let mut out = serde_json::Map::new();
    out.insert("name".into(), function.get("name").cloned().unwrap_or(Value::Null));
    for key in ["description", "parameters"] {
        out.insert(key.into(), function.get(key).cloned().unwrap_or(Value::Null));
    }
    for key in ["strict", "defer_loading"] {
        if let Some(value) = function.get(key).filter(|v| !v.is_null()) {
            out.insert(key.into(), value.clone());
        }
    }
    Some(serde_json::json!({"type": "function", "function": Value::Object(out)}))
}

pub(super) fn encode_arguments(tool_call: &Value) -> Result<String> {
    let original = tool_call
        .get("arguments")
        .context("Missing tool arguments")?;
    let mut arguments = original.clone();
    for _ in 0..2 {
        let Some(text) = arguments.as_str() else {
            break;
        };
        let Ok(decoded) = serde_json::from_str(text) else {
            break;
        };
        arguments = decoded;
    }
    if !arguments.is_object() {
        arguments = serde_json::json!({"arguments": original});
    }
    let parameters = arguments
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, value)| {
            let content = value
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| to_json(value));
            format!(
                "<｜DSML｜ parameter name=\"{name}\" string=\"{}\">{content}</｜DSML｜ parameter>",
                value.is_string()
            )
        })
        .collect::<Vec<_>>();
    Ok(parameters.join("\n"))
}

fn normalize_text(messages: &mut [Value]) -> Result<()> {
    for message in messages {
        for field in ["tools", "tool_calls"] {
            for tool in message
                .get(field)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                ensure!(
                    tool.get("namespace").is_none_or(Value::is_null)
                        && tool
                            .get("function")
                            .and_then(|function| function.get("namespace"))
                            .is_none_or(Value::is_null),
                    "DeepSeek V4.1 native formatter does not support explicit tool namespaces; use a qualified function name"
                );
            }
        }
        ensure!(
            message.get("content_blocks").is_none(),
            "DeepSeek V4.1 expects OpenAI text content blocks in content"
        );
        ensure!(
            !message
                .get("reasoning_content")
                .and_then(Value::as_str)
                .is_some_and(|text| text.contains("<｜deepseek_image｜>")),
            "DeepSeek V4.1 native formatter supports text content only"
        );
        if let Some(reasoning) = message.get("reasoning").cloned() {
            message["reasoning_content"] = reasoning;
        }
        if let Some(content) = message.get("content") {
            let text = match content {
                Value::Null => String::new(),
                Value::String(text) => text.clone(),
                Value::Array(blocks) => {
                    let mut texts = Vec::with_capacity(blocks.len());
                    for block in blocks {
                        ensure!(
                            block.get("type").and_then(Value::as_str) == Some("text"),
                            "DeepSeek V4.1 native formatter supports text content only"
                        );
                        texts.push(
                            block
                                .get("text")
                                .and_then(Value::as_str)
                                .context("Text block requires text")?,
                        );
                    }
                    texts.join("\n\n")
                }
                _ => anyhow::bail!("DeepSeek V4.1 message content must be text"),
            };
            ensure!(
                !text.contains("<｜deepseek_image｜>"),
                "DeepSeek V4.1 native formatter supports text content only"
            );
            message["content"] = Value::String(text);
        }
    }
    Ok(())
}

/// Encode text messages with the model's numeric reasoning effort (1–100).
pub fn encode_messages(
    messages: &[Value],
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
    reasoning_effort: u8,
) -> Result<String> {
    ensure!(
        (1..=100).contains(&reasoning_effort),
        "DeepSeek V4.1 reasoning effort must be within 1–100"
    );
    let mut messages = messages.to_vec();
    normalize_text(&mut messages)?;
    encode_messages_with_encoding(
        &messages,
        thinking_mode,
        true,
        drop_thinking,
        Encoding::V41(reasoning_effort),
    )
}

/// Native text formatter with reasoning-effort names mapped as in the DeepSeek
/// V4.1 reference encoder and vLLM (low 25, high 50, xhigh 75, max 100; default high).
#[derive(Debug, Default)]
pub struct DeepSeekV41Formatter;

impl crate::OAIPromptFormatter for DeepSeekV41Formatter {
    fn supports_add_generation_prompt(&self) -> bool {
        false
    }

    fn render(&self, req: &dyn crate::OAIChatLikeRequest) -> Result<String> {
        // Serving controls follow vLLM's deepseek_v41 tokenizer wrapper, the
        // reference this frontend must match token for token.
        let args = req.chat_template_args();
        let arg = |key: &str| args.and_then(|args| args.get(key));
        let effort = req
            .reasoning_effort()
            .map(|value| serde_json::to_value(value).context("Serialize reasoning effort"))
            .transpose()?
            .or_else(|| arg("reasoning_effort").cloned());
        let mut thinking = if arg("thinking").is_none() && arg("enable_thinking").is_none() {
            true
        } else {
            arg("thinking").is_some_and(python_truthy)
                || arg("enable_thinking").is_some_and(python_truthy)
        };
        let budget = match effort.as_ref() {
            None | Some(Value::Null) => DEFAULT_REASONING_EFFORT,
            Some(Value::String(name)) if name == "none" => {
                thinking = false;
                DEFAULT_REASONING_EFFORT
            }
            Some(Value::String(name)) => REASONING_EFFORT_MAPPINGS
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, budget)| *budget)
                .context(EFFORT_ERROR)?,
            Some(value) => value
                .as_u64()
                .filter(|v| (1..=100).contains(v))
                .context(EFFORT_ERROR)? as u8,
        };
        let thinking_mode = if thinking {
            ThinkingMode::Thinking
        } else {
            ThinkingMode::Chat
        };
        let drop_thinking = arg("drop_thinking").is_none_or(python_truthy);
        let mut messages: Vec<Value> =
            serde_json::from_value(serde_json::to_value(req.messages())?)?;
        let tools_enabled =
            req.tool_choice().as_ref().and_then(|value| value.as_str()) != Some("none");
        let raw_tools = match req.raw_tools() {
            Some(tools) => Some(tools),
            None => req.tools().map(serde_json::to_value).transpose()?,
        };
        let tools: Vec<Value> = raw_tools
            .as_ref()
            .and_then(Value::as_array)
            .filter(|_| tools_enabled)
            .map(|tools| tools.iter().filter_map(vllm_tool).collect())
            .unwrap_or_default();
        // response_format is enforced by guided decoding; vLLM never renders it.
        if !tools.is_empty() {
            let system = match messages
                .iter()
                .position(|m| m.get("role").and_then(Value::as_str) == Some("system"))
            {
                Some(index) => index,
                None => {
                    messages.insert(0, serde_json::json!({"role": "system", "content": ""}));
                    0
                }
            };
            messages[system]["tools"] = Value::Array(tools);
        }
        encode_messages(&messages, thinking_mode, drop_thinking, budget)
    }
}
