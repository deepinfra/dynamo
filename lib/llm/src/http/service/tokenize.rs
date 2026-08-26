// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `POST /tokenize` and `POST /detokenize` HTTP endpoints.
//!
//! Field names and request/response shapes mirror vLLM's
//! `vllm/entrypoints/serve/tokenize/protocol.py` so clients written against
//! vLLM's tokenize API work unchanged against Dynamo.
//!
//! `POST /tokenize` accepts either of two request shapes (untagged union):
//! * `TokenizeCompletionRequest` — `{ "prompt": ... }`
//! * `TokenizeChatRequest`       — `{ "messages": [...] }`, which renders the
//!   model's chat template before tokenizing.
//!
//! Both hand the work to the model's own [`OpenAIPreprocessor`] — the same instance
//! `/v1/chat/completions` and `/v1/completions` run — so the count reported here is
//! the count the model is actually sent. This module deliberately owns no template
//! and no tokenizer of its own: anything it re-derived could drift from the
//! generate path, and has.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{Json, Router, extract::State, extract::rejection::JsonRejection, routing::post};
use serde::{Deserialize, Serialize};

use super::RouteDoc;
use super::openai::{ErrorMessage, ErrorResponse, check_ready};
use super::service_v2;
use crate::model_card::ModelDeploymentCard;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

// ---------------------------------------------------------------------------
// Request / response types (vLLM-compatible)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizeCompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    /// Accepted for vLLM API compatibility, but the default is `false` rather than vLLM's
    /// `true`, because Dynamo's `/v1/completions` tokenizes with special tokens off and
    /// applies no template — the prompt reaches the engine with no BOS. Reporting vLLM's
    /// number here would mean reporting a count this deployment never produces. An explicit
    /// `true` returns 501 rather than silently answering with the other one.
    #[serde(default)]
    pub add_special_tokens: bool,
    #[serde(default)]
    pub return_token_strs: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizeChatRequest {
    pub model: Option<String>,
    pub messages: Vec<serde_json::Value>,
    #[serde(default = "default_true")]
    pub add_generation_prompt: bool,
    #[serde(default)]
    pub continue_final_message: bool,
    /// Default differs from the completion form: chat templates already insert
    /// special tokens, so vLLM defaults this to `false`.
    #[serde(default)]
    pub add_special_tokens: bool,
    pub chat_template: Option<String>,
    pub chat_template_kwargs: Option<HashMap<String, serde_json::Value>>,
    pub tools: Option<Vec<serde_json::Value>>,
    /// The text each token contributes. Decoded per id rather than the byte-level
    /// vocabulary spelling vLLM returns; see [`OpenAIPreprocessor::token_strings`].
    #[serde(default)]
    pub return_token_strs: Option<bool>,
    /// Accepted for vLLM API compatibility. Both only affect multimodal
    /// preprocessing, which this endpoint does not perform.
    pub media_io_kwargs: Option<serde_json::Value>,
    pub mm_processor_kwargs: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum TokenizeRequest {
    /// Boxed: the chat form is ~5x the size of the completion form, and clippy's
    /// `large_enum_variant` would otherwise make every request pay for the bigger one.
    Chat(Box<TokenizeChatRequest>),
    Completion(TokenizeCompletionRequest),
}

#[derive(Debug, Serialize)]
pub struct TokenizeResponse {
    pub count: usize,
    pub max_model_len: u32,
    pub tokens: Vec<u32>,
    /// Always serialized, `null` when not requested — vLLM dumps the field either way.
    pub token_strs: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DetokenizeRequest {
    pub model: Option<String>,
    pub tokens: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub struct DetokenizeResponse {
    pub prompt: String,
}

fn default_true() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Turn an extractor rejection into the same error shape the rest of the API uses.
/// Worth spelling out for `/tokenize`: a body matching neither arm of the untagged
/// union otherwise surfaces as a bare `data did not match any variant` 422.
fn invalid_request(rejection: JsonRejection, expected: &str) -> ErrorResponse {
    bad_request(format!("{rejection}. Expected {expected}."))
}

fn bad_request(message: String) -> ErrorResponse {
    ErrorMessage::from_http_error(super::error::HttpError { code: 400, message })
}

/// Resolve the model name a request targets. vLLM serves one model per process, so a
/// body without `model` is the common case there; honor it only when unambiguous.
fn resolve_model(
    state: &Arc<service_v2::State>,
    requested: Option<&str>,
) -> Result<ModelDeploymentCard, ErrorResponse> {
    let mut cards = state.manager().get_model_cards();
    match requested {
        Some(name) => cards
            .into_iter()
            .find(|card| card.display_name == name)
            .ok_or_else(ErrorMessage::model_not_found),
        None if cards.len() == 1 => Ok(cards.remove(0)),
        None => Err(ErrorMessage::model_not_found()),
    }
}

fn no_preprocessor(model: &str) -> ErrorResponse {
    ErrorMessage::not_implemented_error(format!(
        "Model '{model}' has no Rust tokenizer pipeline, so it cannot be tokenized here"
    ))
}

fn tokenizer_failed(what: &str, error: impl std::fmt::Display) -> ErrorResponse {
    ErrorMessage::internal_server_error(&format!("Tokenizer {what} failed: {error}"))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Multimodal parts expand to many more tokens at inference than the single placeholder
/// a chat template emits, and this endpoint does no media preprocessing — so a count for
/// such a request would be wrong rather than merely approximate. Detect and refuse.
fn first_non_text_part(messages: &[serde_json::Value]) -> Option<&str> {
    messages
        .iter()
        .filter_map(|message| message.get("content")?.as_array())
        .flatten()
        .filter_map(|part| part.get("type")?.as_str())
        .find(|kind| *kind != "text")
}

/// Build the request the generate path would have received. Deserializing into the real
/// request type is deliberate: it validates the messages the same way
/// `/v1/chat/completions` does, so a body this endpoint accepts is one the model
/// would accept.
fn chat_request(
    req: &TokenizeChatRequest,
    model: &str,
) -> Result<NvCreateChatCompletionRequest, ErrorResponse> {
    serde_json::from_value(serde_json::json!({
        "model": model,
        "messages": req.messages,
        "tools": req.tools,
        "chat_template_kwargs": req.chat_template_kwargs,
    }))
    .map_err(|e| bad_request(format!("Invalid chat request: {e}")))
}

async fn tokenize(
    State(state): State<Arc<service_v2::State>>,
    request: Result<Json<TokenizeRequest>, JsonRejection>,
) -> Result<Json<TokenizeResponse>, ErrorResponse> {
    check_ready(&state)?;
    let Json(request) = request.map_err(|rejection| {
        invalid_request(
            rejection,
            "either `prompt` (completion form) or `messages` (chat form)",
        )
    })?;

    let (card, preprocessor, encoding, return_token_strs) = match request {
        TokenizeRequest::Completion(req) => {
            let card = resolve_model(&state, req.model.as_deref())?;
            let preprocessor = state
                .manager()
                .get_preprocessor(&card.display_name)
                .ok_or_else(|| no_preprocessor(&card.display_name))?;
            if req.add_special_tokens {
                return Err(ErrorMessage::not_implemented_error(
                    "`add_special_tokens` is not supported: this model's /v1/completions \
                     tokenizes without them, and /tokenize reports what it would send",
                ));
            }
            let encoding = preprocessor
                .tokenize_completion(&req.prompt)
                .await
                .map_err(|e| tokenizer_failed("encode", e))?;
            (card, preprocessor, encoding, req.return_token_strs)
        }
        TokenizeRequest::Chat(req) => {
            // Accepted by the schema for vLLM API compatibility, but not yet applied.
            // Fail loudly rather than return a count for a prompt we did not build.
            if req.continue_final_message {
                return Err(ErrorMessage::not_implemented_error(
                    "`continue_final_message` is not yet supported",
                ));
            }
            if req.chat_template.is_some() {
                return Err(ErrorMessage::not_implemented_error(
                    "Per-request `chat_template` override is not yet supported",
                ));
            }
            if req.media_io_kwargs.is_some() || req.mm_processor_kwargs.is_some() {
                return Err(ErrorMessage::not_implemented_error(
                    "`media_io_kwargs` and `mm_processor_kwargs` are not yet supported",
                ));
            }
            if let Some(kind) = first_non_text_part(&req.messages) {
                return Err(ErrorMessage::not_implemented_error(format!(
                    "Multimodal content is not yet supported by /tokenize (message part type '{kind}')"
                )));
            }
            if req.add_special_tokens {
                return Err(ErrorMessage::not_implemented_error(
                    "`add_special_tokens` on the chat form is not yet supported; the chat \
                     template already carries the model's special tokens",
                ));
            }

            let card = resolve_model(&state, req.model.as_deref())?;
            let preprocessor = state
                .manager()
                .get_chat_preprocessor(&card.display_name)
                .ok_or_else(|| {
                    ErrorMessage::not_implemented_error(format!(
                        "Model '{}' has no Rust chat pipeline to render a chat template with",
                        card.display_name
                    ))
                })?;
            let mut chat = chat_request(&req, &card.display_name)?;
            let encoding = preprocessor
                .tokenize_chat(&mut chat, req.add_generation_prompt)
                .await
                .map_err(|e| bad_request(format!("Chat template rendering failed: {e}")))?;
            (card, preprocessor, encoding, req.return_token_strs)
        }
    };

    let tokens = encoding.token_ids().to_vec();
    let token_strs = return_token_strs
        .unwrap_or(false)
        .then(|| preprocessor.token_strings(&tokens))
        .transpose()
        .map_err(|e| tokenizer_failed("decode", e))?;

    Ok(Json(TokenizeResponse {
        count: tokens.len(),
        max_model_len: card.effective_context_length(),
        tokens,
        token_strs,
    }))
}

async fn detokenize(
    State(state): State<Arc<service_v2::State>>,
    request: Result<Json<DetokenizeRequest>, JsonRejection>,
) -> Result<Json<DetokenizeResponse>, ErrorResponse> {
    check_ready(&state)?;
    let Json(request) =
        request.map_err(|rejection| invalid_request(rejection, "`tokens`, a list of token ids"))?;
    let card = resolve_model(&state, request.model.as_deref())?;
    let prompt = state
        .manager()
        .get_preprocessor(&card.display_name)
        .ok_or_else(|| no_preprocessor(&card.display_name))?
        .detokenize(&request.tokens, false)
        .map_err(|e| tokenizer_failed("decode", e))?;
    Ok(Json(DetokenizeResponse { prompt }))
}

// ---------------------------------------------------------------------------
// Routers
// ---------------------------------------------------------------------------

pub fn tokenize_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| "/tokenize".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new().route(&path, post(tokenize)).with_state(state);
    (vec![doc], router)
}

pub fn detokenize_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| "/detokenize".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(detokenize))
        .with_state(state);
    (vec![doc], router)
}
