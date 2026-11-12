// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DI: which tool argument schemas the vLLM worker can turn into a grammar.
//!
//! A tool-derived constraint (the `tool_choice=required`/named guided JSON, or a DeepSeek V4.1 `strict` structural tag)
//! pins each tool's `parameters`. vLLM's `auto` structured-output backend sends a schema with a feature xgrammar does
//! not support to guidance and then outlines, which reject many of the same features (`Unimplemented keys:
//! ["uniqueItems"]`, `Unknown format: phone`), and xgrammar itself refuses draft-04 tuple `items`. Each of those is a
//! 400 for the whole request, while vLLM without Dynamo answers it. A schema this check flags is left unconstrained
//! instead: the tool-call envelope and tool names stay enforced, only that tool's arguments are free.
//!
//! The rules are vLLM's `has_xgrammar_unsupported_json_features` (backend_xgrammar.py, in the DSV4/V4.1 worker
//! images) plus deepapi's `xgrammar_schema_support.is_xgrammar_compilable`.

use serde_json::{Map, Value};

/// Set to `0` to pin every tool schema as sent (the previous behavior).
pub const RELAX_ENV: &str = "DI_RELAX_UNCOMPILABLE_TOOL_SCHEMAS";

const STRING_FORMATS: &[&str] = &[
    "email",
    "date",
    "time",
    "date-time",
    "duration",
    "ipv4",
    "ipv6",
    "hostname",
    "uuid",
    "uri",
    "uri-reference",
    "uri-template",
    "json-pointer",
    "relative-json-pointer",
];
const ARRAY_KEYWORDS: &[&str] = &["uniqueItems", "contains", "minContains", "maxContains"];

pub fn relax_enabled() -> bool {
    std::env::var(RELAX_ENV).as_deref() != Ok("0")
}

/// False when the schema uses a feature the worker's structured-output backends refuse to compile.
pub fn is_xgrammar_compilable(schema: &Value) -> bool {
    !has_unsupported_feature(schema)
}

/// The schema to pin for a tool's arguments: `schema` itself, or `relaxed` when the worker could not compile it.
pub fn pinnable_or(schema: Value, relaxed: impl FnOnce() -> Value, tool_name: &str) -> Value {
    if !relax_enabled() || is_xgrammar_compilable(&schema) {
        return schema;
    }
    tracing::debug!(tool = tool_name, "tool schema not compilable by the worker; arguments left unconstrained");
    relaxed()
}

fn has_unsupported_feature(node: &Value) -> bool {
    match node {
        Value::Object(obj) => is_unsupported_here(obj) || obj.values().any(has_unsupported_feature),
        Value::Array(items) => items.iter().any(has_unsupported_feature),
        _ => false,
    }
}

fn schema_types(obj: &Map<String, Value>) -> Vec<&str> {
    match obj.get("type") {
        Some(Value::String(t)) => vec![t.as_str()],
        Some(Value::Array(ts)) => ts.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

fn has_pattern_and_length_bounds(obj: &Map<String, Value>) -> bool {
    (obj.contains_key("pattern") || obj.contains_key("format"))
        && (obj.contains_key("minLength") || obj.contains_key("maxLength"))
}

fn is_unsupported_here(obj: &Map<String, Value>) -> bool {
    let types = schema_types(obj);
    let has_type = |t: &str| types.contains(&t);

    if (has_type("integer") || has_type("number")) && obj.contains_key("multipleOf") {
        return true;
    }
    if has_type("array") && ARRAY_KEYWORDS.iter().any(|k| obj.contains_key(*k)) {
        return true;
    }
    if obj.get("items").is_some_and(Value::is_array) {
        return true;
    }
    if has_type("string") {
        if let Some(format) = obj.get("format")
            && !format.as_str().is_some_and(|f| STRING_FORMATS.contains(&f))
        {
            return true;
        }
        if has_pattern_and_length_bounds(obj) {
            return true;
        }
    }
    if has_type("object") {
        if let Some(names) = obj.get("propertyNames") {
            if names.as_object().is_some_and(has_pattern_and_length_bounds) {
                return true;
            }
            if obj.contains_key("properties")
                || obj.contains_key("patternProperties")
                || obj.get("additionalProperties").is_some_and(Value::is_object)
                || obj.get("unevaluatedProperties").is_some_and(|v| v != &Value::Bool(true))
            {
                return true;
            }
        }
        // An empty `patternProperties` constrains nothing; schema generators emit it next to `properties`.
        if let Some(Value::Object(pattern_properties)) = obj.get("patternProperties")
            && !pattern_properties.is_empty()
            && (obj.contains_key("properties") || pattern_properties.len() > 1)
        {
            return true;
        }
    }
    obj.get("allOf").and_then(Value::as_array).is_some_and(|branches| branches.len() >= 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn wrap(property: Value) -> Value {
        json!({"type": "object", "properties": {"p": property}, "required": ["p"]})
    }

    #[test]
    fn features_the_worker_rejects_are_flagged_at_any_depth() {
        let rejected = [
            json!({"type": "array", "items": {"type": "string"}, "uniqueItems": true}),
            json!({"type": "array", "contains": {"const": "main-hall"}}),
            json!({"type": "integer", "multipleOf": 6}),
            json!({"type": "string", "format": "phone"}),
            json!({"type": ["string", "null"], "format": "regex"}),
            json!({"type": "array", "items": [{"type": "number"}, {"type": "number"}]}),
            json!({"type": "string", "pattern": "^[a-z]+$", "maxLength": 5}),
            json!({"type": "string", "format": "email", "minLength": 3}),
            json!({"allOf": [{"type": "string"}, {"enum": ["x"]}]}),
            json!({"type": "object", "properties": {"a": {}}, "propertyNames": {"pattern": "^a"}}),
            json!({"type": "object", "properties": {"a": {}}, "patternProperties": {"^x": {}}}),
        ];
        for property in rejected {
            assert!(!is_xgrammar_compilable(&wrap(property.clone())), "{property}");
            let nested = wrap(json!({"type": "object", "properties": {"q": property.clone()}}));
            assert!(!is_xgrammar_compilable(&nested), "nested {property}");
            assert!(!is_xgrammar_compilable(&wrap(json!({"anyOf": [property.clone(), {"type": "null"}]}))));
        }
    }

    #[test]
    fn common_strict_schemas_stay_pinned() {
        let accepted = [
            json!({"type": "string", "enum": ["c", "f"], "description": "unit"}),
            json!({"type": ["string", "null"]}),
            json!({"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3}),
            json!({"type": "array", "prefixItems": [{"type": "number"}], "items": false}),
            json!({"type": "string", "format": "date-time"}),
            json!({"type": "string", "pattern": "^[A-Z]{3}$"}),
            json!({"type": "number", "minimum": 0, "exclusiveMaximum": 10}),
            json!({"$ref": "#/$defs/x"}),
            json!({"allOf": [{"type": "string"}]}),
            json!({"type": "object", "additionalProperties": {"type": "integer"}}),
            json!({"type": "integer", "format": "int32"}),
            json!({"type": "object", "properties": {}, "patternProperties": {}}),
        ];
        for property in accepted {
            assert!(is_xgrammar_compilable(&wrap(property.clone())), "{property}");
        }
        let mut with_defs = wrap(json!({"$ref": "#/$defs/x"}));
        with_defs["$defs"] = json!({"x": {"type": "string"}});
        with_defs["additionalProperties"] = json!(false);
        assert!(is_xgrammar_compilable(&with_defs));
    }

    #[test]
    fn pinnable_or_relaxes_only_uncompilable_schemas() {
        let ok = wrap(json!({"type": "string"}));
        assert_eq!(pinnable_or(ok.clone(), || json!(true), "t"), ok);
        let bad = wrap(json!({"type": "string", "format": "phone"}));
        assert_eq!(pinnable_or(bad, || json!(true), "t"), json!(true));
    }
}
