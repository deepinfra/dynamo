// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_renderer::{OAIPromptFormatter, deepseek::v41::DeepSeekV41Formatter};
use serde_json::{Value, json};

struct Request {
    request: dynamo_protocols::types::CreateChatCompletionRequest,
    args: Option<std::collections::HashMap<String, Value>>,
}

impl dynamo_renderer::OAIChatLikeRequest for Request {
    fn model(&self) -> String {
        self.request.model()
    }
    fn messages(&self) -> minijinja::Value {
        self.request.messages()
    }
    fn tools(&self) -> Option<minijinja::Value> {
        self.request.tools()
    }
    fn raw_tools(&self) -> Option<Value> {
        self.request.raw_tools()
    }
    fn tool_choice(&self) -> Option<minijinja::Value> {
        self.request.tool_choice()
    }
    fn response_format(&self) -> Option<minijinja::Value> {
        self.request.response_format()
    }
    fn reasoning_effort(&self) -> Option<minijinja::Value> {
        self.request.reasoning_effort()
    }
    fn should_add_generation_prompt(&self) -> bool {
        true
    }
    fn chat_template_args(&self) -> Option<&std::collections::HashMap<String, Value>> {
        self.args.as_ref()
    }
}

fn render(fields: Value) -> anyhow::Result<String> {
    let mut request =
        json!({"model":"deepseek-v4.1", "messages":[{"role":"user","content":"Hello"}]});
    request
        .as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    let args = request
        .get("chat_template_args")
        .or_else(|| request.get("chat_template_kwargs"))
        .map(|value| serde_json::from_value(value.clone()))
        .transpose()?;
    let request = Request {
        request: serde_json::from_value(request)?,
        args,
    };
    DeepSeekV41Formatter.render(&request)
}

#[test]
fn api_reasoning_effort_matches_the_reference_encoder() {
    for (effort, budget) in [("low", 25), ("high", 50), ("max", 100)] {
        let output = render(json!({"reasoning_effort":effort})).unwrap();
        assert!(output.starts_with(&format!(
            "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: {budget} "
        )));
        assert!(output.ends_with("<｜Assistant｜><think>"));
    }
    assert!(render(json!({})).unwrap().contains("Reasoning Effort: 50 "));
    assert!(
        render(json!({"chat_template_kwargs":{"reasoning_effort":"xhigh"}}))
            .unwrap()
            .contains("Reasoning Effort: 75 ")
    );
    assert_eq!(
        render(json!({"reasoning_effort":"none"})).unwrap(),
        "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"
    );
}

#[test]
fn template_options_set_numeric_effort_and_disable_thinking() {
    for field in ["chat_template_args", "chat_template_kwargs"] {
        assert!(
            render(json!({(field):{"reasoning_effort":37}}))
                .unwrap()
                .contains("Reasoning Effort: 37 ")
        );
        let output = render(json!({(field):{"thinking":false}})).unwrap();
        assert!(!output.contains("Reasoning Effort:"));
        assert!(output.ends_with("</think>"));
        for invalid in [
            json!(0),
            json!(101),
            json!(1.5),
            json!(true),
            json!("medium"),
        ] {
            assert!(render(json!({(field):{"reasoning_effort":invalid}})).is_err());
        }
    }
}

#[test]
fn text_only_boundary_rejects_media() {
    for block in [
        json!({"type":"image_url","image_url":{"url":"https://example.com/image.png"}}),
        json!({"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}),
    ] {
        assert!(render(json!({"messages":[{"role":"user","content":[block]}]})).is_err());
    }
    assert!(
        render(json!({"messages":[{"role":"user","content":"see <｜deepseek_image｜> here"}]}))
            .unwrap()
            .contains("<｜User｜>see <｜deepseek_image｜> here<｜Assistant｜>")
    );
}

#[test]
fn mid_system_preserves_generation_header_and_reasoning_cutoff() {
    let output = render(json!({"messages":[
        {"role":"user","content":"First"},
        {"role":"assistant","content":"Reply","reasoning_content":"Old reasoning"},
        {"role":"system","content":"Next instruction"}
    ]}))
    .unwrap();
    assert!(!output.contains("Old reasoning"));
    assert!(output.ends_with("<｜System｜>Next instruction<｜Assistant｜><think>"));
    assert!(!output.contains("<think></think>"));
}

#[test]
fn tool_choice_none_omits_schema_and_response_format() {
    let output = render(
        json!({"reasoning_effort":"none", "tool_choice":"none", "tools":[
        {"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}
    ],"response_format":{"type":"json_object"}}),
    )
    .unwrap();
    assert!(!output.contains("weather"));
    assert!(!output.contains("## Response Format:"));
}

#[test]
fn official_python_prompt_fixtures() {
    use dynamo_renderer::deepseek::{common::ThinkingMode, v41::encode_messages};
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("fixtures/deepseek_v41.json")).unwrap();
    for case in cases {
        let mode = if case["thinking"].as_bool().unwrap() {
            ThinkingMode::Thinking
        } else {
            ThinkingMode::Chat
        };
        let output = encode_messages(
            case["messages"].as_array().unwrap(),
            mode,
            case["drop_thinking"].as_bool().unwrap(),
            case["effort"].as_u64().unwrap() as u8,
        )
        .unwrap();
        assert_eq!(
            output,
            case["expected"].as_str().unwrap(),
            "{}",
            case["name"]
        );
    }
}

#[test]
fn native_detection_selects_v41_before_v4_and_honors_model_type() {
    use dynamo_renderer::{PromptFormatter, deepseek_formatter_for};
    for (model_type, name, v41) in [
        (Some("deepseek_v41"), "alias", true),
        (None, "deepseek-v4.1-flash", true),
        (Some("deepseek_v4"), "deepseek-v4.1-flash", false),
        (None, "deepseek-v4-flash", false),
    ] {
        let PromptFormatter::OAI(formatter) =
            deepseek_formatter_for(&model_type.map(str::to_owned), name).unwrap();
        let request: dynamo_protocols::types::CreateChatCompletionRequest = serde_json::from_value(
            json!({"model":name,"messages":[{"role":"user","content":"Hello"}]}),
        )
        .unwrap();
        assert_eq!(
            formatter.render(&request).unwrap().contains("<｜System｜>"),
            v41
        );
    }
    assert!(deepseek_formatter_for(&Some("llama".into()), "deepseek-v4.1").is_none());
}

#[test]
fn absent_and_empty_tools_have_identical_prompts() {
    let plain = render(json!({"reasoning_effort":"none"})).unwrap();
    assert_eq!(
        render(json!({"reasoning_effort":"none","tools":[]})).unwrap(),
        plain
    );
    assert_eq!(render(json!({"reasoning_effort":"none","tool_choice":"none","tools":[{"type":"function","function":{"name":"lookup"}}]})).unwrap(), plain);
}

#[test]
fn fixed_generation_header_is_not_advertised_as_configurable() {
    assert!(!DeepSeekV41Formatter.supports_add_generation_prompt());
}

#[test]
fn raw_encoder_rejects_explicit_tool_namespaces() {
    use dynamo_renderer::deepseek::{common::ThinkingMode, v41::encode_messages};
    for message in [
        json!({"role":"system","content":"","tools":[{"namespace":"calendar","function":{"name":"lookup"}}]}),
        json!({"role":"system","content":"","tools":[{"function":{"namespace":{"name":"calendar"},"name":"lookup"}}]}),
        json!({"role":"assistant","tool_calls":[{"namespace":"calendar","function":{"name":"lookup","arguments":"{}"}}]}),
        json!({"role":"assistant","tool_calls":[{"function":{"namespace":"calendar","name":"lookup","arguments":"{}"}}]}),
    ] {
        let error = encode_messages(&[message], ThinkingMode::Thinking, true, 50).unwrap_err();
        assert!(error.to_string().contains("namespace"));
    }
}

#[test]
fn qualified_tool_names_remain_verbatim() {
    use dynamo_renderer::deepseek::{common::ThinkingMode, v41::encode_messages};
    let output = encode_messages(
        &[json!({"role":"assistant","tool_calls":[{"function":{"name":"calendar::lookup","arguments":"{}"}}]})],
        ThinkingMode::Thinking,
        true,
        50,
    ).unwrap();
    assert!(output.contains("name=\"calendar::lookup\""));
}

#[test]
fn normalized_effort_uses_the_api_mapping() {
    for (effort, budget) in [("low", 25), ("high", 50), ("xhigh", 75), ("max", 100)] {
        assert!(
            render(json!({"chat_template_args": {"reasoning_effort": effort}}))
                .unwrap()
                .contains(&format!("Reasoning Effort: {budget} "))
        );
    }
    let output = render(json!({
        "reasoning_effort": "low",
        "chat_template_args": {"reasoning_effort": "max"}
    }))
    .unwrap();
    assert!(output.contains("Reasoning Effort: 25 "));
    assert!(!output.contains("Reasoning Effort: 100 "));
    let output = render(json!({"reasoning_effort": "none", "chat_template_args": {"reasoning_effort": "none", "thinking": false}})).unwrap();
    assert!(!output.contains("Reasoning Effort:"));
    assert!(output.ends_with("</think>"));
}

#[test]
fn vllm_parity_tools_developer_and_floats() {
    let tools = json!([
        {"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{
            "x":{"type":"number","minimum":1e-05,"maximum":1e16,"default":0.0001}}}}},
        {"type":"function","function":{"name":"strict_one","description":"d","parameters":{},"strict":true}}
    ]);
    let output = render(json!({"tools":tools,"messages":[
        {"role":"user","content":"Hello"},
        {"role":"system","content":"Mid"}
    ]}))
    .unwrap();
    assert!(output.contains(
        r#"{"name": "lookup", "description": null, "parameters": {"type": "object", "properties": {"x": {"type": "number", "minimum": 1e-05, "maximum": 1e+16, "default": 0.0001}}}}"#
    ));
    assert!(output.contains(
        r#"{"name": "strict_one", "description": "d", "parameters": {}, "strict": true}"#
    ));
    assert!(output.contains("<｜System｜>Mid\n\n## Tools"));
    assert!(output.starts_with("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 50 "));

    let output = render(json!({"messages":[
        {"role":"developer","content":"Dev"},
        {"role":"user","content":"Q1"},
        {"role":"assistant","content":"A1","reasoning_content":"R1"},
        {"role":"user","content":"Q2"}
    ]}))
    .unwrap();
    assert!(!output.contains("Dev"));
    assert!(!output.contains("R1"));
    let output = render(json!({"chat_template_kwargs":{"thinking":false},"messages":[
        {"role":"developer","content":"Dev"},
        {"role":"user","content":"Q1"}
    ]}))
    .unwrap();
    assert_eq!(
        output,
        "<｜begin▁of▁sentence｜><｜User｜>Dev<｜User｜>Q1<｜Assistant｜></think>"
    );
}
