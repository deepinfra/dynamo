// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DI: whether xgrammar can compile a tool's `parameters` inside a structural tag's `json_schema` content.
//!
//! A structural tag is compiled by xgrammar alone (vLLM sends no structural tag to guidance), and xgrammar ignores
//! keywords it does not implement (`uniqueItems`, unknown `format`s, `multipleOf`, `not`, ...). What it refuses, and
//! what would turn a tool call into a 400 for the whole request, is a schema it cannot read: tuple or null `items`,
//! unknown `type`s, local `$ref`s that do not resolve, regex syntax its converter lacks (lookaround, backreferences,
//! `\b`, `\p{..}`, inline flags), malformed keyword values and empty ranges. `xgrammar_tag_schema_cases.json` holds
//! each case with the DI xgrammar wheel's (0.2.8+di.e085851aec) verdict; the rules here only need to reject every case
//! that wheel rejects. Unknown failure modes still 400, so callers gate the tag.

use serde_json::{Map, Value};

const JSON_TYPES: &[&str] = &[
    "string", "number", "integer", "boolean", "object", "array", "null",
];
const NUMERIC_KEYWORDS: &[&str] = &[
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
];
const COUNT_KEYWORDS: &[&str] = &[
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minProperties",
    "maxProperties",
];
const SCHEMA_MAPS: &[&str] = &[
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
];
const SCHEMA_LISTS: &[&str] = &["anyOf", "oneOf", "allOf", "prefixItems"];
const SCHEMA_VALUES: &[&str] = &[
    "additionalProperties",
    "unevaluatedProperties",
    "items",
    "contains",
    "propertyNames",
    "not",
    "if",
    "then",
    "else",
];

/// True when xgrammar can compile `schema` as structural-tag `json_schema` content.
pub fn compiles_in_structural_tag(schema: &Value) -> bool {
    schema_ok(schema, schema)
}

/// A copy of `schema` with only what xgrammar cannot compile inside a structural tag relaxed, and the keywords that
/// were relaxed. Every move widens what the schema accepts: an offending keyword is dropped (bad `pattern`, unknown
/// `type`, empty or non-array `enum`, non-array `required`, malformed or empty-range bounds, impossible property
/// counts); a non-schema child (tuple or null `items`, ...) or a node whose local `$ref` does not resolve becomes
/// `true`; a node with non-object `properties` or an unsupported `patternProperties` regex becomes `true` (dropping
/// either would admit fewer keys, as xgrammar closes objects by default); and `not` or `if` (with `then`/`else`) is
/// dropped when anything inside it needs relaxing, since widening under them can narrow the whole. Callers re-check
/// the result with [`compiles_in_structural_tag`].
pub fn relax_for_structural_tag(schema: &Value) -> (Value, Vec<&'static str>) {
    let mut relaxed = Vec::new();
    let value = relax_schema(schema, schema, &mut relaxed);
    relaxed.sort_unstable();
    relaxed.dedup();
    (value, relaxed)
}

fn relax_child(
    node: &Value,
    root: &Value,
    keyword: &'static str,
    relaxed: &mut Vec<&'static str>,
) -> Value {
    match node {
        Value::Bool(_) | Value::Object(_) => relax_schema(node, root, relaxed),
        _ => {
            relaxed.push(keyword);
            Value::Bool(true)
        }
    }
}

fn relax_schema(node: &Value, root: &Value, relaxed: &mut Vec<&'static str>) -> Value {
    let Value::Object(obj) = node else {
        return node.clone();
    };
    if let Some(Value::String(reference)) = obj.get("$ref")
        && reference.starts_with('#')
        && resolve_local_ref(root, reference).is_none()
    {
        relaxed.push("$ref");
        return Value::Bool(true);
    }
    if obj.get("properties").is_some_and(|p| !p.is_object()) {
        relaxed.push("properties");
        return Value::Bool(true);
    }
    if obj.get("patternProperties").is_some_and(|p| {
        p.as_object()
            .is_none_or(|p| !p.keys().all(|k| regex_supported(k)))
    }) {
        relaxed.push("patternProperties");
        return Value::Bool(true);
    }

    let mut out = obj.clone();
    relax_keywords(&mut out, relaxed);
    for &key in SCHEMA_MAPS {
        let Some(current) = out.get(key).cloned() else {
            continue;
        };
        match current {
            Value::Object(map) => {
                let map: Map<String, Value> = map
                    .iter()
                    .map(|(name, child)| (name.clone(), relax_child(child, root, key, relaxed)))
                    .collect();
                out.insert(key.to_string(), Value::Object(map));
            }
            _ => drop_if(&mut out, key, true, relaxed),
        }
    }
    for &key in SCHEMA_LISTS {
        let Some(current) = out.get(key).cloned() else {
            continue;
        };
        match current {
            Value::Array(list) => {
                let list = list
                    .iter()
                    .map(|child| relax_child(child, root, key, relaxed))
                    .collect();
                out.insert(key.to_string(), Value::Array(list));
            }
            _ => drop_if(&mut out, key, true, relaxed),
        }
    }
    for &key in SCHEMA_VALUES {
        let Some(current) = out.get(key).cloned() else {
            continue;
        };
        match key {
            "not" | "if" => {
                let mut inner = Vec::new();
                relax_child(&current, root, key, &mut inner);
                if !inner.is_empty() {
                    drop_if(&mut out, key, true, relaxed);
                    if key == "if" {
                        out.remove("then");
                        out.remove("else");
                    }
                }
            }
            "contains" if !(current.is_object() || current.is_boolean()) => {
                drop_if(&mut out, key, true, relaxed);
            }
            _ => {
                let child = relax_child(&current, root, key, relaxed);
                out.insert(key.to_string(), child);
            }
        }
    }
    Value::Object(out)
}

fn relax_keywords(obj: &mut Map<String, Value>, relaxed: &mut Vec<&'static str>) {
    let type_ok = match obj.get("type") {
        None => true,
        Some(Value::String(t)) => JSON_TYPES.contains(&t.as_str()),
        Some(Value::Array(ts)) => ts
            .iter()
            .all(|t| t.as_str().is_some_and(|t| JSON_TYPES.contains(&t))),
        Some(_) => false,
    };
    drop_if(obj, "type", !type_ok, relaxed);
    let required_bad = obj.get("required").is_some_and(|v| !v.is_array());
    drop_if(obj, "required", required_bad, relaxed);
    let enum_bad = obj
        .get("enum")
        .is_some_and(|v| v.as_array().is_none_or(Vec::is_empty));
    drop_if(obj, "enum", enum_bad, relaxed);
    for &key in NUMERIC_KEYWORDS {
        let bad = obj.get(key).is_some_and(|v| !v.is_number());
        drop_if(obj, key, bad, relaxed);
    }
    for &key in COUNT_KEYWORDS {
        let bad = obj.get(key).is_some_and(|v| v.as_u64().is_none());
        drop_if(obj, key, bad, relaxed);
    }
    let pattern_bad = obj
        .get("pattern")
        .is_some_and(|p| p.as_str().is_none_or(|p| !regex_supported(p)));
    drop_if(obj, "pattern", pattern_bad, relaxed);
    let integer = match obj.get("type") {
        Some(Value::String(t)) => t == "integer",
        Some(Value::Array(ts)) => ts.iter().any(|t| t.as_str() == Some("integer")),
        _ => false,
    };
    if integer {
        for &key in &NUMERIC_KEYWORDS[..4] {
            let bad = obj
                .get(key)
                .and_then(Value::as_f64)
                .is_some_and(|v| v.fract() != 0.0);
            drop_if(obj, key, bad, relaxed);
        }
    }
    for (lo, hi) in [
        ("minLength", "maxLength"),
        ("minItems", "maxItems"),
        ("minProperties", "maxProperties"),
    ] {
        let inverted = matches!(
            (obj.get(lo).and_then(Value::as_u64), obj.get(hi).and_then(Value::as_u64)),
            (Some(lo), Some(hi)) if lo > hi
        );
        drop_if(obj, lo, inverted, relaxed);
        drop_if(obj, hi, inverted, relaxed);
    }
    if !ranges_ok(obj, integer) {
        for &key in &NUMERIC_KEYWORDS[..4] {
            let present = obj.contains_key(key);
            drop_if(obj, key, present, relaxed);
        }
    }
    if !property_counts_ok(obj) {
        for key in ["minProperties", "maxProperties"] {
            let present = obj.contains_key(key);
            drop_if(obj, key, present, relaxed);
        }
    }
}

fn drop_if(
    obj: &mut Map<String, Value>,
    key: &'static str,
    bad: bool,
    relaxed: &mut Vec<&'static str>,
) {
    if bad {
        obj.remove(key);
        relaxed.push(key);
    }
}

fn schema_ok(node: &Value, root: &Value) -> bool {
    let obj = match node {
        Value::Bool(_) => return true,
        Value::Object(obj) => obj,
        _ => return false,
    };
    node_ok(obj, root)
        && SCHEMA_MAPS.iter().all(|key| match obj.get(*key) {
            None => true,
            Some(Value::Object(map)) => map.values().all(|child| schema_ok(child, root)),
            Some(_) => false,
        })
        && SCHEMA_LISTS.iter().all(|key| match obj.get(*key) {
            None => true,
            Some(Value::Array(list)) => list.iter().all(|child| schema_ok(child, root)),
            Some(_) => false,
        })
        && SCHEMA_VALUES
            .iter()
            .all(|key| obj.get(*key).is_none_or(|child| schema_ok(child, root)))
}

fn node_ok(obj: &Map<String, Value>, root: &Value) -> bool {
    let types = match obj.get("type") {
        None => Vec::new(),
        Some(Value::String(t)) => vec![t.as_str()],
        Some(Value::Array(ts)) => match ts.iter().map(Value::as_str).collect::<Option<Vec<_>>>() {
            Some(ts) => ts,
            None => return false,
        },
        Some(_) => return false,
    };
    if !types.iter().all(|t| JSON_TYPES.contains(t)) {
        return false;
    }
    if obj.get("required").is_some_and(|v| !v.is_array())
        || obj
            .get("enum")
            .is_some_and(|v| v.as_array().is_none_or(Vec::is_empty))
        || NUMERIC_KEYWORDS
            .iter()
            .any(|k| obj.get(*k).is_some_and(|v| !v.is_number()))
        || COUNT_KEYWORDS
            .iter()
            .any(|k| obj.get(*k).is_some_and(|v| v.as_u64().is_none()))
    {
        return false;
    }
    if let Some(Value::String(reference)) = obj.get("$ref")
        && reference.starts_with('#')
        && resolve_local_ref(root, reference).is_none()
    {
        return false;
    }
    if obj
        .get("pattern")
        .is_some_and(|p| p.as_str().is_none_or(|p| !regex_supported(p)))
    {
        return false;
    }
    if let Some(Value::Object(patterns)) = obj.get("patternProperties")
        && !patterns.keys().all(|p| regex_supported(p))
    {
        return false;
    }
    if types.contains(&"integer")
        && NUMERIC_KEYWORDS[..4].iter().any(|k| {
            obj.get(*k)
                .and_then(Value::as_f64)
                .is_some_and(|v| v.fract() != 0.0)
        })
    {
        return false;
    }
    ranges_ok(obj, types.contains(&"integer")) && property_counts_ok(obj)
}

fn ranges_ok(obj: &Map<String, Value>, integer: bool) -> bool {
    let num = |k: &str| obj.get(k).and_then(Value::as_f64);
    let count = |k: &str| obj.get(k).and_then(Value::as_u64);
    let ordered =
        |lo: Option<u64>, hi: Option<u64>| !matches!((lo, hi), (Some(lo), Some(hi)) if lo > hi);
    if !ordered(count("minLength"), count("maxLength"))
        || !ordered(count("minItems"), count("maxItems"))
        || !ordered(count("minProperties"), count("maxProperties"))
    {
        return false;
    }
    let lower = [(num("minimum"), false), (num("exclusiveMinimum"), true)];
    let upper = [(num("maximum"), false), (num("exclusiveMaximum"), true)];
    lower.iter().all(|&(lo, lo_excl)| {
        upper.iter().all(|&(hi, hi_excl)| match (lo, hi) {
            (Some(lo), Some(hi)) if integer => {
                let lo = if lo_excl { lo.floor() + 1.0 } else { lo.ceil() };
                let hi = if hi_excl { hi.ceil() - 1.0 } else { hi.floor() };
                lo <= hi
            }
            (Some(lo), Some(hi)) => lo < hi || (lo == hi && !lo_excl && !hi_excl),
            _ => true,
        })
    })
}

fn property_counts_ok(obj: &Map<String, Value>) -> bool {
    let declared = obj
        .get("properties")
        .and_then(Value::as_object)
        .map_or(0, Map::len) as u64;
    let required = obj
        .get("required")
        .and_then(Value::as_array)
        .map_or(0, Vec::len) as u64;
    // xgrammar treats an unspecified additionalProperties as false.
    let open = matches!(
        obj.get("additionalProperties"),
        Some(Value::Bool(true) | Value::Object(_))
    ) || obj
        .get("patternProperties")
        .and_then(Value::as_object)
        .is_some_and(|p| !p.is_empty());
    let min_ok = open
        || obj
            .get("minProperties")
            .and_then(Value::as_u64)
            .is_none_or(|min| min <= declared);
    let max_ok = obj
        .get("maxProperties")
        .and_then(Value::as_u64)
        .is_none_or(|max| max >= required);
    min_ok && max_ok
}

fn resolve_local_ref<'a>(root: &'a Value, reference: &str) -> Option<&'a Value> {
    let pointer = reference.strip_prefix('#')?;
    if pointer.is_empty() {
        return Some(root);
    }
    root.pointer(pointer)
}

/// Regex syntax xgrammar's converter accepts: no lookaround, backreferences, word boundaries, Unicode property
/// classes or group flags (only `(?:` and `(?<name>` groups), with balanced brackets and parentheses.
fn regex_supported(pattern: &str) -> bool {
    let chars: Vec<char> = pattern.chars().collect();
    let mut depth = 0i32;
    let mut in_class = false;
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '\\' => {
                let Some(&next) = chars.get(i + 1) else {
                    return false;
                };
                if matches!(next, 'b' | 'B' | 'p' | 'P' | 'k' | '1'..='9') {
                    return false;
                }
                i += 2;
                continue;
            }
            '[' if !in_class => in_class = true,
            ']' if in_class => in_class = false,
            '(' if !in_class => {
                if chars.get(i + 1) == Some(&'?') {
                    let named = chars.get(i + 2) == Some(&'<')
                        && chars
                            .get(i + 3)
                            .is_some_and(|c| c.is_ascii_alphabetic() || *c == '_');
                    if chars.get(i + 2) != Some(&':') && !named {
                        return false;
                    }
                }
                depth += 1;
            }
            ')' if !in_class => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => {}
        }
        i += 1;
    }
    depth == 0 && !in_class
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `relaxed*` are set for the cases the wheel rejects: the relaxed schema and keywords, and the wheel's verdict on
    /// the relaxed schema.
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        schema: Value,
        xgrammar_compiles: bool,
        relaxed: Option<Value>,
        relaxed_keywords: Option<Vec<String>>,
        relaxed_xgrammar_compiles: Option<bool>,
    }

    fn cases() -> Vec<Case> {
        serde_json::from_str(include_str!("xgrammar_tag_schema_cases.json")).unwrap()
    }

    #[test]
    fn rejects_every_schema_xgrammar_rejects() {
        for case in cases().into_iter().filter(|c| !c.xgrammar_compiles) {
            assert!(
                !compiles_in_structural_tag(&case.schema),
                "{} must be relaxed: {}",
                case.name,
                case.schema
            );
        }
    }

    #[test]
    fn keeps_schemas_xgrammar_compiles() {
        let kept: Vec<_> = cases()
            .into_iter()
            .filter(|c| c.xgrammar_compiles)
            .collect();
        assert!(kept.len() > 30);
        for case in kept {
            assert!(
                compiles_in_structural_tag(&case.schema),
                "{} should stay pinned: {}",
                case.name,
                case.schema
            );
        }
    }

    #[test]
    fn relax_leaves_compiling_schemas_alone() {
        for case in cases().into_iter().filter(|c| c.xgrammar_compiles) {
            let (relaxed, keywords) = relax_for_structural_tag(&case.schema);
            assert_eq!(relaxed, case.schema, "{}", case.name);
            assert!(keywords.is_empty(), "{}: {keywords:?}", case.name);
        }
    }

    #[test]
    fn relax_matches_what_the_wheel_compiled() {
        for case in cases().into_iter().filter(|c| !c.xgrammar_compiles) {
            let (relaxed, keywords) = relax_for_structural_tag(&case.schema);
            assert_eq!(Some(&relaxed), case.relaxed.as_ref(), "{}", case.name);
            let expected: Vec<&str> = case
                .relaxed_keywords
                .iter()
                .flatten()
                .map(String::as_str)
                .collect();
            assert_eq!(keywords, expected, "{}", case.name);
            if case.relaxed_xgrammar_compiles != Some(true) {
                assert!(
                    !compiles_in_structural_tag(&relaxed),
                    "{}: the wheel rejects the relaxed schema, so it must fall back",
                    case.name
                );
            }
        }
    }

    #[test]
    fn relax_drops_only_the_offending_keyword() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "code": {"type": "string", "pattern": "^(?=.*[A-Z])[A-Za-z]+$", "maxLength": 8},
                "count": {"type": "integer", "minimum": 0},
                "point": {"type": "array", "items": [{"type": "number"}]},
                "kind": {"type": "str"},
            },
            "required": ["code", "count"],
        });
        let (relaxed, keywords) = relax_for_structural_tag(&schema);
        assert_eq!(
            relaxed,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "code": {"type": "string", "maxLength": 8},
                    "count": {"type": "integer", "minimum": 0},
                    "point": {"type": "array", "items": true},
                    "kind": {},
                },
                "required": ["code", "count"],
            })
        );
        assert_eq!(keywords, vec!["items", "pattern", "type"]);
        assert!(compiles_in_structural_tag(&relaxed));
    }

    #[test]
    fn relax_drops_not_and_if_when_their_insides_need_relaxing() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "not": {"properties": {"a": {"pattern": "(?=x)"}}},
            "if": {"properties": {"a": {"enum": []}}},
            "then": {"required": ["a"]},
            "else": {"required": []},
        });
        let (relaxed, keywords) = relax_for_structural_tag(&schema);
        assert_eq!(
            relaxed,
            serde_json::json!({"type": "object", "properties": {"a": {"type": "string"}}})
        );
        assert_eq!(keywords, vec!["if", "not"]);
    }

    #[test]
    fn regex_subset() {
        for ok in [
            r"^[a-z]+$",
            r"^\d{3}-\d{4}$",
            r"^(?:ab)+$",
            r"^(?<x>a)$",
            r"^\$[0-9]+(\.[0-9]{2})?$",
            r"[(]",
        ] {
            assert!(regex_supported(ok), "{ok}");
        }
        for bad in [
            r"^(?=a)",
            r"(?<=a)b",
            r"(?<!a)b",
            r"^(a)\1$",
            r"\bfoo",
            r"\p{L}",
            r"(?i)a",
            r"(?P<x>a)",
            r"([a-z",
            r"a)",
        ] {
            assert!(!regex_supported(bad), "{bad}");
        }
    }
}
