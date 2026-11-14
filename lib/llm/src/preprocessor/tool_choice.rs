// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-choice guided decoding policy for OpenAI chat requests.

use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};
use crate::protocols::openai::GuidedToolConstraint;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use crate::protocols::openai::tools::{
    ToolChoiceGuidance, get_tool_choice_guidance_from_tools, validate_openai_tool_choice,
};

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_protocols::types::{ChatCompletionTool, ChatCompletionToolChoiceOption, ResponseFormat};
use dynamo_runtime::error::{DynamoError, ErrorType};

fn invalid_argument(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message)
        .build()
}

impl OpenAIPreprocessor {
    /// Whether this request permits model output to be interpreted as tool calls.
    ///
    /// A configured parser describes the model's wire format; it does not grant
    /// every request permission to return tool calls. Permission depends only on
    /// whether the request supplies tools and whether `tool_choice` forbids them.
    /// Assistant-output constraints apply to assistant content and do not revoke
    /// an `auto` request's ability to choose a tool call.
    pub(crate) fn tool_call_parsing_enabled(request: &NvCreateChatCompletionRequest) -> bool {
        if request.inner.tools.as_ref().is_none_or(Vec::is_empty) {
            return false;
        }

        match request
            .inner
            .tool_choice
            .as_ref()
            .unwrap_or(&ChatCompletionToolChoiceOption::Auto)
        {
            ChatCompletionToolChoiceOption::None => false,
            ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_) => {
                true
            }
            ChatCompletionToolChoiceOption::Auto => true,
        }
    }

    /// Apply guided decoding for OpenAI tool-choice requests.
    ///
    /// Structural tags are preferred when enabled and supported by the configured
    /// tool-call parser. Supported K2 forced requests and named K3 requests
    /// intrinsically use their native structural tags because generic JSON cannot
    /// represent their tool calls. Other forced choices fall back to the legacy
    /// JSON-schema constraint when structural tags are not applied, except K3
    /// required requests, which stay on the prompt-level XTML path.
    pub(super) fn apply_tool_choice_guided_decoding(
        &self,
        request: &NvCreateChatCompletionRequest,
        common_request: &mut PreprocessedRequest,
        prompt_injected_reasoning: bool,
    ) -> Result<GuidedToolConstraint, DynamoError> {
        let tool_choice = request
            .inner
            .tool_choice
            .as_ref()
            .unwrap_or(&ChatCompletionToolChoiceOption::Auto);
        let tools = request.inner.tools.as_deref().unwrap_or(&[]);
        let is_forced_tool_choice = matches!(
            tool_choice,
            ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_)
        );
        let has_explicit_guided_decoding = has_explicit_guided_decoding(request);
        let has_response_format_constraint = has_response_format_constraint(request);

        if is_forced_tool_choice && has_explicit_guided_decoding {
            return Err(invalid_argument(concat!(
                "guided decoding cannot be used in the same request as ",
                "tool_choice=\"required\" or a named tool_choice.",
            )));
        }

        // For non-forced tool choice, explicit guided decoding and response_format
        // constrain assistant content, so tool-choice guided decoding stays inactive.
        let has_assistant_constraint =
            has_explicit_guided_decoding || has_response_format_constraint;
        if !is_forced_tool_choice && has_assistant_constraint {
            return Ok(GuidedToolConstraint::None);
        }

        if is_forced_tool_choice
            && has_response_format_constraint
            && let Some(gd) = common_request.sampling_options.guided_decoding.as_mut()
        {
            // OpenAI `response_format` applies to assistant content, not tool calls.
            gd.json = None;
        }

        // DI: DeepSeek V4.1 gets vLLM's --tool-strict-level=function structural tag (see deepseek_v41_tool_tag). The
        // unified parser still reads native DSML, so the reported constraint stays None.
        if self.tool_call_parser.as_deref() == Some("deepseek_v41")
            && std::env::var("DI_V41_TOOL_STRUCTURAL_TAG").as_deref() != Ok("0")
            && let Some(tag) = super::deepseek_v41_tool_tag::structural_tag(
                &convert_tool_choice(tool_choice),
                &convert_tools(tools),
                request.inner.parallel_tool_calls,
            )
        {
            common_request
                .sampling_options
                .guided_decoding
                .get_or_insert_default()
                .structural_tag = Some(tag);
            return Ok(GuidedToolConstraint::None);
        }

        // DI: DeepSeek V4 DSML structural tag for `auto` behind DI_DSV4_TOOL_STRUCTURAL_TAG (see deepseek_v4_tool_tag).
        // The output stays native DSML for today's stream parser, so no constraint is reported. Forced choices get no
        // tag and keep the guided-JSON array below.
        if super::deepseek_v4_tool_tag::applies(self.tool_call_parser.as_deref())
            && let Some(tag) = super::deepseek_v4_tool_tag::structural_tag(
                &convert_tool_choice(tool_choice),
                &convert_tools(tools),
                request.inner.parallel_tool_calls,
            )
        {
            common_request
                .sampling_options
                .guided_decoding
                .get_or_insert_default()
                .structural_tag = Some(tag);
            return Ok(GuidedToolConstraint::None);
        }

        // DI: a V4 request that cannot call tools (none, or `tool_choice=none`) after a tool-calling history must not
        // write DSML, which nothing would parse and would leak into `content` (see deepseek_v4_tool_tag::dsml_ban_tag).
        if super::deepseek_v4_tool_tag::applies(self.tool_call_parser.as_deref())
            && !is_forced_tool_choice
            && (tools.is_empty() || matches!(tool_choice, ChatCompletionToolChoiceOption::None))
            && super::deepseek_v4_tool_tag::history_has_tool_calls(&request.inner.messages)
        {
            let gd = common_request
                .sampling_options
                .guided_decoding
                .get_or_insert_default();
            if gd.structural_tag.is_none() {
                gd.structural_tag = Some(super::deepseek_v4_tool_tag::dsml_ban_tag());
            }
            return Ok(GuidedToolConstraint::None);
        }

        if self.apply_tool_choice_structural_tag(
            &convert_tool_choice(tool_choice),
            &convert_tools(tools),
            request.inner.parallel_tool_calls,
            prompt_injected_reasoning,
            common_request,
        )? {
            return Ok(GuidedToolConstraint::StructuralTag);
        }

        if is_forced_tool_choice
            && parses_forced_tool_calls_natively(self.tool_call_parser.as_deref())
        {
            // vLLM's DeepSeek V4.1 tool parser sets `supports_required_and_named = False`:
            // the model answers a forced choice in native DSML, which the unified parser
            // extracts, and no tool-call grammar is installed (vLLM's adjust_request and
            // Dynamo's vLLM processor both skip it). A generic JSON schema here would
            // constrain the wrong wire format and send the worker grammars it rejects.
            drop_constraintless_guided_decoding(common_request);
            return Ok(GuidedToolConstraint::None);
        }

        let uses_kimi_k3_parser = uses_kimi_k3_parser(
            self.tool_call_parser.as_deref(),
            self.runtime_config.reasoning_parser.as_deref(),
        );
        if is_forced_tool_choice && uses_kimi_k3_parser {
            if matches!(tool_choice, ChatCompletionToolChoiceOption::Named(_)) {
                return Err(invalid_argument(
                    "named tool choice for Kimi K3 requires --dyn-tool-call-parser kimi_k3 \
                     with XTML structural-tag support",
                ));
            }

            // K3's prompt-level required instruction produces an XTML `tools`
            // channel. Generic JSON guided decoding would constrain the wrong
            // wire format and prevent the Rust K3 parser from seeing it. No JSON
            // schema is installed, so this is NOT a guided-JSON request.
            drop_constraintless_guided_decoding(common_request);
            return Ok(GuidedToolConstraint::None);
        }

        match get_tool_choice_guidance_from_tools(
            Some(tool_choice),
            Some(tools),
            request.inner.parallel_tool_calls,
        ) {
            Ok(Some(guidance)) => {
                let gd = common_request
                    .sampling_options
                    .guided_decoding
                    .get_or_insert_default();
                match guidance {
                    ToolChoiceGuidance::Json(schema) => gd.json = Some(schema),
                    ToolChoiceGuidance::Regex(regex) => gd.regex = Some(regex),
                }

                // Report the parser constraint implied by the installed grammar,
                // not merely the tool_choice that requested it. Both JSON and
                // regex guidance still use the guided-JSON tool-output parser.
                return Ok(installed_json_constraint(tool_choice));
            }
            Ok(None) => {}
            Err(err) => {
                return Err(invalid_argument(err.to_string()));
            }
        }

        // Auto/None requests can reach here when neither structural tags nor a
        // tool-choice JSON fallback were needed.
        Ok(GuidedToolConstraint::None)
    }
}

/// Clearing a forced choice's `response_format` JSON can leave guided decoding with
/// no constraint at all; vLLM rejects such options ("You must use one kind of
/// structured outputs constraint"), and the Python processor sends none.
fn drop_constraintless_guided_decoding(common_request: &mut PreprocessedRequest) {
    let options = &mut common_request.sampling_options.guided_decoding;
    if options.as_ref().is_some_and(|gd| {
        gd.json.is_none()
            && gd.regex.is_none()
            && gd.choice.as_ref().is_none_or(Vec::is_empty)
            && gd.grammar.is_none()
            && gd.structural_tag.is_none()
    }) {
        *options = None;
    }
}

fn has_explicit_guided_decoding(request: &NvCreateChatCompletionRequest) -> bool {
    request.common.guided_json.is_some()
        || request.common.guided_regex.is_some()
        || request
            .common
            .guided_choice
            .as_ref()
            .is_some_and(|v| !v.is_empty())
        || request.common.guided_grammar.is_some()
}

fn has_response_format_constraint(request: &NvCreateChatCompletionRequest) -> bool {
    request
        .inner
        .response_format
        .as_ref()
        .is_some_and(|format| !matches!(format, ResponseFormat::Text))
}

pub(crate) fn convert_tool_choice(tool_choice: &ChatCompletionToolChoiceOption) -> ToolChoice {
    match tool_choice {
        ChatCompletionToolChoiceOption::None => ToolChoice::None,
        ChatCompletionToolChoiceOption::Auto => ToolChoice::Auto,
        ChatCompletionToolChoiceOption::Required => ToolChoice::Required,
        ChatCompletionToolChoiceOption::Named(named) => {
            ToolChoice::Named(named.function.name.clone())
        }
    }
}

pub(crate) fn convert_tools(tools: &[ChatCompletionTool]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .map(|tool| ToolDefinition {
            name: tool.function.name.clone(),
            parameters: tool.function.parameters.clone(),
            strict: tool.function.strict,
        })
        .collect()
}

/// The guided-tool constraint a request implies, given what the structural-tag stage
/// already decided.
///
/// Direct postprocessor callers and remote frontends use this compatibility helper
/// when they did not run request preprocessing. The worker streaming path carries the
/// enum returned by `apply_tool_choice_guided_decoding`; topology-B aggregation may
/// reconstruct a structural-tag request as guided JSON, so its complete-output parser
/// also checks the generated wire shape.
pub(crate) fn guided_tool_constraint(
    request: &NvCreateChatCompletionRequest,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    uses_structural_tag: bool,
) -> Result<GuidedToolConstraint, DynamoError> {
    if uses_structural_tag {
        return Ok(GuidedToolConstraint::StructuralTag);
    }
    let tool_choice = request
        .inner
        .tool_choice
        .as_ref()
        .unwrap_or(&ChatCompletionToolChoiceOption::Auto);
    validate_openai_tool_choice(Some(tool_choice), request.inner.tools.as_deref())
        .map_err(|error| invalid_argument(error.to_string()))?;
    let is_forced_tool_choice = matches!(
        tool_choice,
        ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_)
    );
    // Only a forced choice installs a tool-level JSON schema. Auto/None leave any
    // JSON constraint to `response_format`, which governs assistant CONTENT.
    if !is_forced_tool_choice {
        return Ok(GuidedToolConstraint::None);
    }
    // Forced + explicit guided decoding is rejected at request time, so reaching here
    // with both means the request never ran.
    if has_explicit_guided_decoding(request) {
        return Ok(GuidedToolConstraint::None);
    }
    // K3 forced requests are served by a prompt-level XTML instruction; no JSON schema.
    if uses_kimi_k3_parser(tool_call_parser, reasoning_parser) {
        return Ok(GuidedToolConstraint::None);
    }
    if parses_forced_tool_calls_natively(tool_call_parser) {
        return Ok(GuidedToolConstraint::None);
    }
    // Validate the forced choice against the actual tools the same way
    // `apply_tool_choice_guided_decoding` does, instead of blindly installing a
    // constraint for a `tool_choice` that names a tool absent from `tools` (or an
    // empty `tools` list under `tool_choice: "required"`).
    let tools = request.inner.tools.as_deref().unwrap_or(&[]);
    match get_tool_choice_guidance_from_tools(
        Some(tool_choice),
        Some(tools),
        request.inner.parallel_tool_calls,
    ) {
        Ok(Some(_)) => Ok(installed_json_constraint(tool_choice)),
        Ok(None) => Ok(GuidedToolConstraint::None),
        Err(e) => Err(invalid_argument(e.to_string())),
    }
}

/// Parsers whose forced (`required` / named) tool calls stay in the model's native
/// syntax, mirroring vLLM tool parsers with `supports_required_and_named = False`.
fn parses_forced_tool_calls_natively(tool_call_parser: Option<&str>) -> bool {
    tool_call_parser
        == Some(
            crate::protocols::openai::chat_completions::unified_parser::DEEPSEEK_V41_UNIFIED_FAMILY,
        )
}

/// True when either configured parser is Kimi K3.
///
/// K3 forced requests are served by a prompt-level XTML instruction, so they must
/// NOT be reported as guided JSON even though their `tool_choice` is forced.
fn uses_kimi_k3_parser(tool_call_parser: Option<&str>, reasoning_parser: Option<&str>) -> bool {
    let is_k3 = |parser: &str| matches!(parser, "kimi_k3" | "kimi-k3");
    tool_call_parser.is_some_and(is_k3) || reasoning_parser.is_some_and(is_k3)
}

/// Map a forced `tool_choice` onto the guided-JSON parser constraint for its grammar.
///
/// Only reachable once `get_tool_choice_guidance_from_tools` produced either JSON or
/// regex guidance. It returns `None` for `Auto`/`None`, so those arms are unreachable
/// in practice and report [`GuidedToolConstraint::None`] rather than guessing.
fn installed_json_constraint(tool_choice: &ChatCompletionToolChoiceOption) -> GuidedToolConstraint {
    match tool_choice {
        ChatCompletionToolChoiceOption::Named(named) => GuidedToolConstraint::GuidedJsonNamed {
            tool_name: named.function.name.clone(),
        },
        ChatCompletionToolChoiceOption::Required => GuidedToolConstraint::GuidedJsonRequired,
        ChatCompletionToolChoiceOption::Auto | ChatCompletionToolChoiceOption::None => {
            GuidedToolConstraint::None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::{ChatCompletionNamedToolChoice, FunctionName};
    use serde_json::{Value, json};

    fn request(extra: Value) -> NvCreateChatCompletionRequest {
        let mut value = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}]
        });
        value
            .as_object_mut()
            .expect("base request is an object")
            .extend(extra.as_object().expect("extra is an object").clone());
        serde_json::from_value(value).expect("request must deserialize")
    }

    fn tools() -> Value {
        json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        }])
    }

    #[test]
    fn tool_call_parsing_requires_tools() {
        assert!(!OpenAIPreprocessor::tool_call_parsing_enabled(&request(
            json!({})
        )));
        assert!(!OpenAIPreprocessor::tool_call_parsing_enabled(&request(
            json!({"tool_choice": "required"})
        )));
    }

    #[test]
    fn deepseek_v41_forced_tool_choice_installs_no_json_constraint() {
        for tool_choice in [
            json!("required"),
            json!({"type": "function", "function": {"name": "get_weather"}}),
        ] {
            let request = request(json!({"tools": tools(), "tool_choice": tool_choice}));
            assert_eq!(
                guided_tool_constraint(&request, Some("deepseek_v41"), Some("deepseek_v41"), false)
                    .unwrap(),
                GuidedToolConstraint::None
            );
            assert_ne!(
                guided_tool_constraint(&request, None, None, false).unwrap(),
                GuidedToolConstraint::None
            );
        }
    }

    #[test]
    fn tool_call_parsing_honors_each_tool_choice() {
        for (tool_choice, expected) in [
            (json!(null), true),
            (json!("none"), false),
            (json!("auto"), true),
            (json!("required"), true),
            (
                json!({"type": "function", "function": {"name": "get_weather"}}),
                true,
            ),
        ] {
            let mut extra = json!({"tools": tools()});
            if !tool_choice.is_null() {
                extra["tool_choice"] = tool_choice;
            }
            assert_eq!(
                OpenAIPreprocessor::tool_call_parsing_enabled(&request(extra)),
                expected,
            );
        }
    }

    #[test]
    fn assistant_constraints_do_not_revoke_auto_tool_permission() {
        let constraints = [
            json!({"response_format": {"type": "json_object"}}),
            json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "event",
                        "schema": {"type": "object"}
                    }
                }
            }),
            json!({"guided_json": {"type": "object"}}),
            json!({"guided_regex": "[a-z]+"}),
            json!({"guided_choice": ["a", "b"]}),
            json!({"guided_grammar": "root ::= 'a'"}),
        ];

        for constraint in constraints {
            let mut auto = json!({"tools": tools(), "tool_choice": "auto"});
            auto.as_object_mut()
                .unwrap()
                .extend(constraint.as_object().unwrap().clone());
            assert!(OpenAIPreprocessor::tool_call_parsing_enabled(&request(
                auto
            )));

            let mut required = json!({"tools": tools(), "tool_choice": "required"});
            required
                .as_object_mut()
                .unwrap()
                .extend(constraint.as_object().unwrap().clone());
            assert!(OpenAIPreprocessor::tool_call_parsing_enabled(&request(
                required
            )));
        }
    }

    #[test]
    fn text_response_format_does_not_disable_auto_tool_parsing() {
        assert!(OpenAIPreprocessor::tool_call_parsing_enabled(&request(
            json!({
                "tools": tools(),
                "tool_choice": "auto",
                "response_format": {"type": "text"}
            })
        )));
    }

    fn named(name: &str) -> ChatCompletionToolChoiceOption {
        ChatCompletionToolChoiceOption::Named(ChatCompletionNamedToolChoice {
            r#type: dynamo_protocols::types::ChatCompletionToolType::Function,
            function: FunctionName {
                name: name.to_string(),
            },
        })
    }

    #[test]
    fn only_structural_tag_reports_a_structural_tag() {
        assert!(GuidedToolConstraint::StructuralTag.uses_structural_tag());
        assert!(!GuidedToolConstraint::None.uses_structural_tag());
        assert!(!GuidedToolConstraint::GuidedJsonRequired.uses_structural_tag());
        assert!(
            !GuidedToolConstraint::GuidedJsonNamed {
                tool_name: "get_weather".to_string(),
            }
            .uses_structural_tag()
        );
    }

    #[test]
    fn required_reports_the_array_shape() {
        assert_eq!(
            installed_json_constraint(&ChatCompletionToolChoiceOption::Required),
            GuidedToolConstraint::GuidedJsonRequired
        );
    }

    #[test]
    fn named_carries_the_tool_name_from_the_request() {
        assert_eq!(
            installed_json_constraint(&named("get_weather")),
            GuidedToolConstraint::GuidedJsonNamed {
                tool_name: "get_weather".to_string(),
            }
        );
    }

    #[test]
    fn named_closed_zero_arg_tool_keeps_the_named_parser_constraint() {
        let request = request(json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_server_time",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_server_time"}
            }
        }));

        assert_eq!(
            guided_tool_constraint(&request, None, None, false).expect("constraint is valid"),
            GuidedToolConstraint::GuidedJsonNamed {
                tool_name: "get_server_time".to_string(),
            }
        );
    }

    #[test]
    fn unforced_choices_install_no_tool_constraint() {
        assert_eq!(
            installed_json_constraint(&ChatCompletionToolChoiceOption::Auto),
            GuidedToolConstraint::None
        );
        assert_eq!(
            installed_json_constraint(&ChatCompletionToolChoiceOption::None),
            GuidedToolConstraint::None
        );
    }

    #[test]
    fn kimi_k3_is_detected_from_either_parser_slot() {
        assert!(uses_kimi_k3_parser(Some("kimi_k3"), None));
        assert!(uses_kimi_k3_parser(Some("kimi-k3"), None));
        assert!(uses_kimi_k3_parser(None, Some("kimi_k3")));
        assert!(uses_kimi_k3_parser(None, Some("kimi-k3")));
    }

    #[test]
    fn non_k3_parsers_are_not_mistaken_for_k3() {
        assert!(!uses_kimi_k3_parser(None, None));
        assert!(!uses_kimi_k3_parser(Some("kimi_k2"), Some("qwen3")));
        assert!(!uses_kimi_k3_parser(Some("qwen3_coder"), None));
    }

    #[test]
    fn kimi_k3_required_without_tools_is_rejected_before_the_xtml_return() {
        for extra in [
            json!({"tool_choice": "required"}),
            json!({"tool_choice": "required", "tools": []}),
        ] {
            let request = request(extra);
            let result = guided_tool_constraint(&request, Some("kimi_k3"), None, false);
            assert!(
                result.is_err(),
                "Kimi K3 required must reject both missing and empty tools"
            );
        }
    }

    mod deepseek_v4_tag {
        use std::path::PathBuf;
        use std::sync::Arc;

        use super::*;
        use crate::model_card::ModelDeploymentCard;
        use crate::preprocessor::deepseek_v4_tool_tag::GATE_FOR_TEST;
        use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};

        fn preprocessor(parser: &str) -> Arc<OpenAIPreprocessor> {
            let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
            let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
            mdc.runtime_config.tool_call_parser = Some(parser.to_string());
            if parser.starts_with("deepseek") {
                mdc.runtime_config.reasoning_parser = Some(parser.to_string());
            }
            OpenAIPreprocessor::new(mdc).unwrap()
        }

        fn apply(
            parser: &str,
            gate: bool,
            extra: Value,
        ) -> (GuidedToolConstraint, PreprocessedRequest) {
            GATE_FOR_TEST.with(|g| g.set(Some(gate)));
            let mut common = PreprocessedRequest::builder()
                .model("test-model".to_string())
                .token_ids(Vec::new())
                .stop_conditions(StopConditions::default())
                .sampling_options(SamplingOptions::default())
                .output_options(OutputOptions::default())
                .build()
                .unwrap();
            let constraint = preprocessor(parser)
                .apply_tool_choice_guided_decoding(&request(extra), &mut common, false)
                .unwrap();
            GATE_FOR_TEST.with(|g| g.set(None));
            (constraint, common)
        }

        fn structural_tag(common: &PreprocessedRequest) -> Option<Value> {
            common
                .sampling_options
                .guided_decoding
                .as_ref()
                .and_then(|gd| gd.structural_tag.clone())
        }

        fn guided_decoding(common: &PreprocessedRequest) -> Value {
            serde_json::to_value(&common.sampling_options.guided_decoding).unwrap()
        }

        fn reconstructed(extra: Value, gate: bool) -> GuidedToolConstraint {
            GATE_FOR_TEST.with(|g| g.set(Some(gate)));
            let constraint = guided_tool_constraint(
                &request(extra),
                Some("deepseek_v4"),
                Some("deepseek_v4"),
                false,
            )
            .unwrap();
            GATE_FOR_TEST.with(|g| g.set(None));
            constraint
        }

        fn forced_choices() -> [Value; 2] {
            [
                json!("required"),
                json!({"type": "function", "function": {"name": "get_weather"}}),
            ]
        }

        fn choices() -> [Value; 3] {
            let [required, named] = forced_choices();
            [json!("auto"), required, named]
        }

        #[test]
        fn gate_on_installs_the_dsml_tag_for_auto_only() {
            let extra = json!({"tools": tools(), "tool_choice": "auto"});
            let (constraint, common) = apply("deepseek_v4", true, extra.clone());
            assert_eq!(constraint, GuidedToolConstraint::None);
            let tag = structural_tag(&common).expect("tag installed");
            assert!(
                tag.to_string()
                    .contains("<｜DSML｜invoke name=\\\"get_weather\\\">")
            );
            let gd = common.sampling_options.guided_decoding.as_ref().unwrap();
            assert!(gd.json.is_none() && gd.regex.is_none());
            assert_eq!(reconstructed(extra, true), GuidedToolConstraint::None);
        }

        #[test]
        fn gate_on_forced_choices_match_gate_off_exactly() {
            for tool_choice in forced_choices() {
                let extra = json!({"tools": tools(), "tool_choice": tool_choice});
                let (on, on_common) = apply("deepseek_v4", true, extra.clone());
                let (off, off_common) = apply("deepseek_v4", false, extra.clone());
                assert_eq!(on, off, "{tool_choice}");
                assert_eq!(
                    guided_decoding(&on_common),
                    guided_decoding(&off_common),
                    "{tool_choice}"
                );
                assert!(structural_tag(&on_common).is_none(), "{tool_choice}");
                assert_eq!(
                    reconstructed(extra.clone(), true),
                    reconstructed(extra, false),
                    "HTTP-side reconstruction: {tool_choice}"
                );
            }
        }

        #[test]
        fn gate_off_keeps_todays_behavior() {
            for tool_choice in choices() {
                let extra = json!({"tools": tools(), "tool_choice": tool_choice});
                let (constraint, common) = apply("deepseek_v4", false, extra.clone());
                assert!(structural_tag(&common).is_none(), "{tool_choice}");
                let today =
                    guided_tool_constraint(&request(extra), Some("deepseek_v4"), None, false)
                        .unwrap();
                assert_eq!(constraint, today, "{tool_choice}");
            }
        }

        #[test]
        fn gate_on_leaves_other_families_alone() {
            for parser in ["deepseek_v41", "qwen3_coder", "glm47"] {
                for gate in [false, true] {
                    for tool_choice in choices() {
                        let extra = json!({"tools": tools(), "tool_choice": tool_choice});
                        let (constraint, common) = apply(parser, gate, extra);
                        let tag = structural_tag(&common)
                            .map(|t| t.to_string())
                            .unwrap_or_default();
                        assert!(
                            !tag.contains("<｜DSML｜tool_calls>"),
                            "{parser} {gate} {tool_choice}"
                        );
                        let (baseline, _) = apply(
                            parser,
                            false,
                            json!({"tools": tools(), "tool_choice": tool_choice}),
                        );
                        assert_eq!(constraint, baseline, "{parser} {gate} {tool_choice}");
                    }
                }
            }
        }

        #[test]
        fn auto_with_a_response_format_gets_no_tag() {
            let extra = json!({
                "tools": tools(),
                "tool_choice": "auto",
                "response_format": {"type": "json_object"},
            });
            let (constraint, common) = apply("deepseek_v4", true, extra);
            assert_eq!(constraint, GuidedToolConstraint::None);
            assert!(structural_tag(&common).is_none());
        }

        /// The prompt is reordered exactly when the installed tag splits on `required`.
        #[test]
        fn prompt_reorders_exactly_when_the_tag_splits_on_required() {
            use crate::preprocessor::deepseek_v4_tool_tag::{REQUIRED_FOR_TEST, prompt_tools};

            let tools = json!([{"type": "function", "function": {"name": "post", "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "id": {"type": "string"}},
                "required": ["id"],
            }}}]);
            let cases = [
                (json!({}), true),
                (json!({"tool_choice": "auto"}), true),
                (
                    json!({"tool_choice": "auto", "response_format": {"type": "text"}}),
                    true,
                ),
                (
                    json!({"tool_choice": "auto", "response_format": {"type": "json_object"}}),
                    false,
                ),
                (
                    json!({"tool_choice": "auto", "guided_json": {"type": "object"}}),
                    false,
                ),
                (json!({"tool_choice": "required"}), false),
                (
                    json!({"tool_choice": {"type": "function", "function": {"name": "post"}}}),
                    false,
                ),
                (json!({"tool_choice": "none"}), false),
            ];
            for (extra, expected) in cases {
                let mut extra = extra;
                extra["tools"] = tools.clone();
                REQUIRED_FOR_TEST.with(|g| g.set(Some(true)));
                let (_, common) = apply("deepseek_v4", true, extra.clone());
                GATE_FOR_TEST.with(|g| g.set(Some(true)));
                let reordered =
                    prompt_tools(Some("deepseek_v4"), &request(extra.clone())).is_some();
                GATE_FOR_TEST.with(|g| g.set(None));
                REQUIRED_FOR_TEST.with(|g| g.set(None));
                let split = structural_tag(&common).is_some_and(|tag| {
                    tag["format"]["tags"][0]["content"]["tags"][0]["content"]["type"] == "sequence"
                });
                assert_eq!(split, expected, "tag split: {extra}");
                assert_eq!(reordered, expected, "prompt reordered: {extra}");
            }
        }

        fn tool_history() -> Value {
            json!([
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"},
                }]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
                {"role": "user", "content": "thanks"},
            ])
        }

        #[test]
        fn no_callable_tools_after_a_tool_history_bans_dsml() {
            use crate::preprocessor::deepseek_v4_tool_tag::dsml_ban_tag;

            let banned = [
                json!({"messages": tool_history()}),
                json!({"messages": tool_history(), "tools": tools(), "tool_choice": "none"}),
                json!({"messages": tool_history(), "tool_choice": "auto"}),
            ];
            for extra in banned {
                let (constraint, common) = apply("deepseek_v4", true, extra.clone());
                assert_eq!(constraint, GuidedToolConstraint::None, "{extra}");
                assert_eq!(structural_tag(&common), Some(dsml_ban_tag()), "{extra}");
            }

            let response_format = json!({"type": "json_schema", "json_schema": {
                "name": "answer", "schema": {"type": "object"},
            }});
            let untouched = [
                json!({}),
                json!({"tools": tools(), "tool_choice": "none"}),
                json!({"messages": tool_history(), "response_format": response_format}),
                json!({"messages": tool_history(), "guided_regex": "a+"}),
            ];
            for extra in untouched {
                let (on, on_common) = apply("deepseek_v4", true, extra.clone());
                let (off, off_common) = apply("deepseek_v4", false, extra.clone());
                assert_eq!(on, off, "{extra}");
                assert_eq!(
                    guided_decoding(&on_common),
                    guided_decoding(&off_common),
                    "{extra}"
                );
            }
        }

        #[test]
        fn the_dsml_ban_is_gated_and_v4_only() {
            let cases = [
                json!({"messages": tool_history()}),
                json!({"messages": tool_history(), "tools": tools(), "tool_choice": "none"}),
            ];
            for extra in cases {
                let (constraint, common) = apply("deepseek_v4", false, extra.clone());
                assert!(structural_tag(&common).is_none(), "gate off: {extra}");
                let today = guided_tool_constraint(
                    &request(extra.clone()),
                    Some("deepseek_v4"),
                    None,
                    false,
                )
                .unwrap();
                assert_eq!(constraint, today, "gate off: {extra}");
                for parser in ["deepseek_v41", "qwen3_coder", "glm47"] {
                    let (on, on_common) = apply(parser, true, extra.clone());
                    let (off, off_common) = apply(parser, false, extra.clone());
                    assert_eq!(on, off, "{parser}: {extra}");
                    assert_eq!(
                        guided_decoding(&on_common),
                        guided_decoding(&off_common),
                        "{parser}: {extra}"
                    );
                }
            }
        }
    }
}
