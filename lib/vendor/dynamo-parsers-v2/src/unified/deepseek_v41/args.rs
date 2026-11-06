// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DI: DeepSeek V4.1 tool-call arguments exactly as vLLM's `deepseek_v41` parser builds them.
//!
//! vLLM turns an invoke body into the client's `arguments` string in three steps, all ported here:
//! `_dsml_arg_converter` (parameter regex, `string="true"` or else `json.loads`), `_unwrap_wrapper_args`
//! (a lone `arguments`/`input` wrapper the schema does not declare), and `ParserEngine._fix_arg_types`
//! (`coerce_to_schema_type` against the tool's JSON schema). Values are Python objects ([`Py`]) rather than
//! `serde_json::Value` so integers of any size, `NaN`/`Infinity` and float-vs-int survive to `json.dumps`.

use serde_json::{Map, Value};

use crate::tool_calling::traits::Tool;

const PARAMETER_START: &str = "<｜DSML｜ parameter";
const PARAMETER_END: &str = "</｜DSML｜ parameter>";

/// A Python JSON value. `Int` holds the canonical decimal text, so it has no size limit.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Py {
    None,
    Bool(bool),
    Int(String),
    Float(f64),
    Str(String),
    List(Vec<Py>),
    Dict(Vec<(String, Py)>),
}

impl Py {
    fn dict_insert(entries: &mut Vec<(String, Py)>, key: String, value: Py) {
        match entries.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1 = value,
            None => entries.push((key, value)),
        }
    }

    fn is_finite(&self) -> bool {
        match self {
            Py::Float(f) => f.is_finite(),
            Py::List(items) => items.iter().all(Py::is_finite),
            Py::Dict(entries) => entries.iter().all(|(_, v)| v.is_finite()),
            _ => true,
        }
    }
}

/// The `arguments` string vLLM returns for a call to `name` whose invoke body (between the
/// `">` of the header and the invoke close, or the end of output for a truncated call) is `body`.
pub(crate) fn invoke_arguments(body: &str, name: &str, tools: &[Tool]) -> String {
    let mut args = Py::Dict(dsml_parameters(body));
    let properties = tool_properties(tools, name);
    if let Some(properties) = properties.filter(|p| !p.is_empty()) {
        args = unwrap_wrapper(args, properties);
        if let Py::Dict(entries) = &mut args {
            coerce_dict(entries, properties);
        }
    }
    dumps(&args)
}

/// `find_tool_properties`: the first tool named `name`, its `parameters.properties`.
fn tool_properties<'a>(tools: &'a [Tool], name: &str) -> Option<&'a Map<String, Value>> {
    let tool = tools.iter().find(|tool| tool.name == name)?;
    tool.parameters.get("properties")?.as_object()
}

/// `_dsml_arg_converter(raw, partial=False)`: every parameter matched by vLLM's regex
/// `<｜DSML｜ parameter\s+name="([^"]+)"(?:\s+string="([^"]*)")?\s*>(.*?)(?:</｜DSML｜ parameter>|(?=<｜DSML｜ parameter\s+name=))`.
pub(crate) fn dsml_parameters(raw: &str) -> Vec<(String, Py)> {
    let mut params = Vec::new();
    let mut pos = 0;
    while let Some(found) = raw[pos..].find(PARAMETER_START) {
        let at = pos + found;
        let Some((name, string, value_start)) = parameter_head(raw, at) else {
            pos = at + PARAMETER_START.len();
            continue;
        };
        let rest = &raw[value_start..];
        let close = rest.find(PARAMETER_END);
        let next = next_parameter_head(rest);
        let (value_end, resume) = match (close, next) {
            (Some(c), Some(n)) if n < c => (n, value_start + n),
            (Some(c), _) => (c, value_start + c + PARAMETER_END.len()),
            (None, Some(n)) => (n, value_start + n),
            (None, None) => break,
        };
        let value = &rest[..value_end];
        let value = if string == Some("true") {
            Py::Str(value.to_string())
        } else {
            loads(value).unwrap_or_else(|| Py::Str(value.to_string()))
        };
        Py::dict_insert(&mut params, name.to_string(), value);
        pos = resume;
    }
    params
}

fn skip_ws(text: &str, from: usize) -> usize {
    text[from..]
        .char_indices()
        .find(|(_, c)| !c.is_whitespace())
        .map_or(text.len(), |(i, _)| from + i)
}

/// The header at `at`: name, optional `string` attribute and the offset where the value starts.
fn parameter_head(raw: &str, at: usize) -> Option<(&str, Option<&str>, usize)> {
    let mut pos = at + PARAMETER_START.len();
    let after_ws = skip_ws(raw, pos);
    if after_ws == pos {
        return None;
    }
    pos = after_ws;
    let rest = raw[pos..].strip_prefix("name=\"")?;
    let name_len = rest.find('"')?;
    if name_len == 0 {
        return None;
    }
    let name = &rest[..name_len];
    pos = raw.len() - rest.len() + name_len + 1;
    let mut string = None;
    let after_ws = skip_ws(raw, pos);
    if after_ws > pos
        && let Some(attr) = raw[after_ws..].strip_prefix("string=\"")
        && let Some(len) = attr.find('"')
    {
        string = Some(&attr[..len]);
        pos = raw.len() - attr.len() + len + 1;
    }
    pos = skip_ws(raw, pos);
    raw[pos..]
        .starts_with('>')
        .then_some((name, string, pos + 1))
}

/// Offset of the lookahead `<｜DSML｜ parameter\s+name=` in `text`, if any.
fn next_parameter_head(text: &str) -> Option<usize> {
    let mut pos = 0;
    while let Some(found) = text[pos..].find(PARAMETER_START) {
        let at = pos + found;
        let after = at + PARAMETER_START.len();
        let ws_end = skip_ws(text, after);
        if ws_end > after && text[ws_end..].starts_with("name=") {
            return Some(at);
        }
        pos = after;
    }
    None
}

/// `_unwrap_wrapper_args`: `{"arguments": {...}}` / `{"input": "{...}"}` → the inner object when the
/// schema does not declare the wrapper and declares every inner key.
fn unwrap_wrapper(args: Py, properties: &Map<String, Value>) -> Py {
    let Py::Dict(entries) = &args else {
        return args;
    };
    for wrapper in ["arguments", "input"] {
        if entries.len() != 1 || entries[0].0 != wrapper || properties.contains_key(wrapper) {
            continue;
        }
        let inner = match &entries[0].1 {
            Py::Str(text) => match loads(text) {
                Some(inner) => inner,
                None => return args,
            },
            other => other.clone(),
        };
        if let Py::Dict(inner_entries) = &inner
            && inner_entries
                .iter()
                .all(|(k, _)| properties.contains_key(k))
        {
            return inner;
        }
    }
    args
}

fn coerce_dict(entries: &mut [(String, Py)], properties: &Map<String, Value>) -> bool {
    let mut changed = false;
    for (key, value) in entries.iter_mut() {
        if let Some(schema) = properties.get(key).filter(|s| s.is_object()) {
            changed |= coerce_value(value, schema);
        }
    }
    changed
}

/// `ParserEngine._coerce_value`; returns whether the value changed.
fn coerce_value(value: &mut Py, schema: &Value) -> bool {
    match value {
        Py::Str(text) => match coerce_to_schema_type(text, &schema_types(schema)) {
            Some(coerced) => {
                *value = coerced;
                true
            }
            None => false,
        },
        Py::Dict(entries) => match schema.get("properties").and_then(Value::as_object) {
            Some(properties) => coerce_dict(entries, properties),
            None => false,
        },
        Py::List(items) => match schema.get("items").filter(|s| s.is_object()) {
            Some(items_schema) => {
                let mut changed = false;
                for item in items.iter_mut() {
                    changed |= coerce_value(item, items_schema);
                }
                changed
            }
            None => false,
        },
        _ => {
            let as_str = dumps(value);
            let coerced =
                coerce_to_schema_type(&as_str, &schema_types(schema)).unwrap_or(Py::Str(as_str));
            if coerced == *value {
                false
            } else {
                *value = coerced;
                true
            }
        }
    }
}

/// `extract_types_from_schema`.
fn schema_types(schema: &Value) -> Vec<String> {
    let mut types = Vec::new();
    collect_schema_types(schema, &mut types);
    if types.is_empty() {
        types.push("string".to_string());
    }
    types
}

fn collect_schema_types(schema: &Value, types: &mut Vec<String>) {
    let Some(schema) = schema.as_object() else {
        types.push("string".to_string());
        return;
    };
    let mut add = |t: &str| {
        if !types.iter().any(|known| known == t) {
            types.push(t.to_string());
        }
    };
    match schema.get("type") {
        Some(Value::String(t)) => add(t),
        Some(Value::Array(list)) => list.iter().filter_map(Value::as_str).for_each(&mut add),
        _ => {}
    }
    if let Some(Value::Array(values)) = schema.get("enum") {
        for value in values {
            add(match value {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(n) if n.is_f64() => "number",
                Value::Number(_) => "integer",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            });
        }
    }
    for choice_field in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(choices)) = schema.get(choice_field) {
            for choice in choices {
                let mut nested = Vec::new();
                collect_schema_types(choice, &mut nested);
                if nested.is_empty() {
                    nested.push("string".to_string());
                }
                for t in nested {
                    if !types.contains(&t) {
                        types.push(t);
                    }
                }
            }
        }
    }
}

fn type_alias(name: &str) -> &str {
    match name {
        "str" | "text" | "varchar" | "char" | "enum" => "string",
        "int" | "int32" | "int64" | "uint" | "uint32" | "uint64" | "long" | "short"
        | "unsigned" => "integer",
        "float" | "float32" | "float64" | "double" => "number",
        "bool" => "boolean",
        "dict" => "object",
        "arr" | "list" | "sequence" => "array",
        other => other,
    }
}

/// `coerce_to_schema_type`; `None` means the input string itself is returned.
fn coerce_to_schema_type(value: &str, types: &[String]) -> Option<Py> {
    let normalized: Vec<String> = types
        .iter()
        .map(|t| type_alias(&t.trim().to_lowercase()).to_string())
        .collect();
    let has = |t: &str| normalized.iter().any(|n| n == t);
    for candidate in [
        "null", "integer", "number", "boolean", "object", "array", "string",
    ] {
        if !has(candidate) {
            continue;
        }
        match candidate {
            "null" => {
                if value.to_lowercase() == "null" {
                    return Some(Py::None);
                }
            }
            "string" => return None,
            "integer" => {
                if let Some(int) = py_int(value) {
                    return Some(Py::Int(int));
                }
            }
            "number" => {
                if let Some(int) = py_int(value) {
                    return Some(Py::Int(int));
                }
                if let Some(float) = py_float(value).filter(|f| f.is_finite()) {
                    return Some(if float.fract() != 0.0 {
                        Py::Float(float)
                    } else {
                        Py::Int(integral_float_text(float))
                    });
                }
            }
            "boolean" => match value.trim().to_lowercase().as_str() {
                "true" | "1" => return Some(Py::Bool(true)),
                "false" | "0" => return Some(Py::Bool(false)),
                _ => {}
            },
            _ => {
                if let Some(parsed) = loads(value).filter(Py::is_finite) {
                    return Some(parsed);
                }
            }
        }
    }
    loads(value).filter(Py::is_finite)
}

fn canonical_int(negative: bool, digits: &str) -> String {
    let digits = digits.trim_start_matches('0');
    match (digits.is_empty(), negative) {
        (true, _) => "0".to_string(),
        (false, true) => format!("-{digits}"),
        (false, false) => digits.to_string(),
    }
}

fn integral_float_text(value: f64) -> String {
    let text = format!("{value:.0}");
    match text.strip_prefix('-') {
        Some(digits) => canonical_int(true, digits),
        None => canonical_int(false, &text),
    }
}

/// Digits with Python's single underscores between digits removed; `None` if malformed.
fn py_digits(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let mut out = String::with_capacity(text.len());
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'0'..=b'9' => out.push(b as char),
            b'_' if i > 0
                && i + 1 < bytes.len()
                && bytes[i - 1].is_ascii_digit()
                && bytes[i + 1].is_ascii_digit() => {}
            _ => return None,
        }
    }
    Some(out)
}

/// Python `int(text)` for base 10, as canonical decimal text.
fn py_int(text: &str) -> Option<String> {
    let text = text.trim();
    let (negative, digits) = match text.as_bytes().first()? {
        b'-' => (true, &text[1..]),
        b'+' => (false, &text[1..]),
        _ => (false, text),
    };
    Some(canonical_int(negative, &py_digits(digits)?))
}

/// Python `float(text)`.
fn py_float(text: &str) -> Option<f64> {
    let text = text.trim();
    let (sign, body) = match text.as_bytes().first()? {
        b'-' => ("-", &text[1..]),
        b'+' => ("", &text[1..]),
        _ => ("", text),
    };
    let lower = body.to_ascii_lowercase();
    if matches!(lower.as_str(), "inf" | "infinity" | "nan") {
        return format!("{sign}{lower}").parse().ok();
    }
    let (mantissa, exponent) = match body.find(['e', 'E']) {
        Some(at) => (&body[..at], Some(&body[at + 1..])),
        None => (body, None),
    };
    let (int_part, frac_part) = match mantissa.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (mantissa, None),
    };
    let int_digits = if int_part.is_empty() {
        String::new()
    } else {
        py_digits(int_part)?
    };
    let frac_digits = match frac_part {
        Some(f) if !f.is_empty() => py_digits(f)?,
        _ => String::new(),
    };
    if int_digits.is_empty() && frac_digits.is_empty() {
        return None;
    }
    let mut normalized = format!(
        "{sign}{}.{}",
        if int_digits.is_empty() {
            "0"
        } else {
            &int_digits
        },
        if frac_digits.is_empty() {
            "0"
        } else {
            &frac_digits
        }
    );
    if let Some(exponent) = exponent {
        let (exp_sign, exp_digits) = match exponent.as_bytes().first()? {
            b'-' => ("-", &exponent[1..]),
            b'+' => ("", &exponent[1..]),
            _ => ("", exponent),
        };
        normalized.push('e');
        normalized.push_str(exp_sign);
        normalized.push_str(&py_digits(exp_digits)?);
    }
    normalized.parse().ok()
}

/// Python `json.loads` (strict): `None` on any error.
pub(crate) fn loads(text: &str) -> Option<Py> {
    let mut parser = JsonParser {
        text: text.as_bytes(),
        src: text,
        pos: 0,
    };
    parser.ws();
    let value = parser.value()?;
    parser.ws();
    (parser.pos == parser.text.len()).then_some(value)
}

struct JsonParser<'a> {
    text: &'a [u8],
    src: &'a str,
    pos: usize,
}

impl JsonParser<'_> {
    fn ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.text.get(self.pos) {
            self.pos += 1;
        }
    }

    fn eat(&mut self, literal: &str) -> bool {
        if self.src[self.pos..].starts_with(literal) {
            self.pos += literal.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Option<Py> {
        match *self.text.get(self.pos)? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => self.string().map(Py::Str),
            b'n' if self.eat("null") => Some(Py::None),
            b't' if self.eat("true") => Some(Py::Bool(true)),
            b'f' if self.eat("false") => Some(Py::Bool(false)),
            b'N' if self.eat("NaN") => Some(Py::Float(f64::NAN)),
            b'I' if self.eat("Infinity") => Some(Py::Float(f64::INFINITY)),
            b'-' if self.eat("-Infinity") => Some(Py::Float(f64::NEG_INFINITY)),
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.pos;
        while self.text.get(self.pos).is_some_and(u8::is_ascii_digit) {
            self.pos += 1;
        }
        self.pos - start
    }

    fn number(&mut self) -> Option<Py> {
        let start = self.pos;
        let negative = self.text[self.pos] == b'-';
        if negative {
            self.pos += 1;
        }
        let int_start = self.pos;
        match self.text.get(self.pos) {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return None,
        }
        let int_end = self.pos;
        let mut is_float = false;
        if self.text.get(self.pos) == Some(&b'.')
            && self.text.get(self.pos + 1).is_some_and(u8::is_ascii_digit)
        {
            self.pos += 1;
            self.digits();
            is_float = true;
        }
        if let Some(b'e' | b'E') = self.text.get(self.pos) {
            let save = self.pos;
            self.pos += 1;
            if let Some(b'+' | b'-') = self.text.get(self.pos) {
                self.pos += 1;
            }
            if self.digits() == 0 {
                self.pos = save;
            } else {
                is_float = true;
            }
        }
        if is_float {
            self.src[start..self.pos].parse().ok().map(Py::Float)
        } else {
            Some(Py::Int(canonical_int(
                negative,
                &self.src[int_start..int_end],
            )))
        }
    }

    fn hex4(&mut self) -> Option<u32> {
        let hex = self.src.get(self.pos..self.pos + 4)?;
        let value = u32::from_str_radix(hex, 16).ok()?;
        if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        self.pos += 4;
        Some(value)
    }

    fn string(&mut self) -> Option<String> {
        self.pos += 1;
        let mut out = String::new();
        loop {
            let rest = &self.src[self.pos..];
            let stop = rest.find(|c: char| c == '"' || c == '\\' || (c as u32) < 0x20)?;
            out.push_str(&rest[..stop]);
            self.pos += stop;
            match self.text[self.pos] {
                b'"' => {
                    self.pos += 1;
                    return Some(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let escape = *self.text.get(self.pos)?;
                    self.pos += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let high = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&high)
                                && self.src[self.pos..].starts_with("\\u")
                            {
                                let save = self.pos;
                                self.pos += 2;
                                match self.hex4() {
                                    Some(low) if (0xDC00..0xE000).contains(&low) => {
                                        0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
                                    }
                                    _ => {
                                        self.pos = save;
                                        high
                                    }
                                }
                            } else {
                                high
                            };
                            out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                        }
                        _ => return None,
                    }
                }
                _ => return None,
            }
        }
    }

    fn array(&mut self) -> Option<Py> {
        self.pos += 1;
        let mut items = Vec::new();
        self.ws();
        if self.eat("]") {
            return Some(Py::List(items));
        }
        loop {
            self.ws();
            items.push(self.value()?);
            self.ws();
            if self.eat("]") {
                return Some(Py::List(items));
            }
            if !self.eat(",") {
                return None;
            }
        }
    }

    fn object(&mut self) -> Option<Py> {
        self.pos += 1;
        let mut entries = Vec::new();
        self.ws();
        if self.eat("}") {
            return Some(Py::Dict(entries));
        }
        loop {
            self.ws();
            if self.text.get(self.pos) != Some(&b'"') {
                return None;
            }
            let key = self.string()?;
            self.ws();
            if !self.eat(":") {
                return None;
            }
            self.ws();
            let value = self.value()?;
            Py::dict_insert(&mut entries, key, value);
            self.ws();
            if self.eat("}") {
                return Some(Py::Dict(entries));
            }
            if !self.eat(",") {
                return None;
            }
        }
    }
}

/// Python `json.dumps(value, ensure_ascii=False)`.
pub(crate) fn dumps(value: &Py) -> String {
    let mut out = String::new();
    dump_into(value, &mut out);
    out
}

fn dump_into(value: &Py, out: &mut String) {
    match value {
        Py::None => out.push_str("null"),
        Py::Bool(true) => out.push_str("true"),
        Py::Bool(false) => out.push_str("false"),
        Py::Int(digits) => out.push_str(digits),
        Py::Float(f) if f.is_nan() => out.push_str("NaN"),
        Py::Float(f) if f.is_infinite() => {
            out.push_str(if *f > 0.0 { "Infinity" } else { "-Infinity" })
        }
        Py::Float(f) => out.push_str(&super::python_float_repr(*f)),
        Py::Str(text) => dump_str(text, out),
        Py::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump_into(item, out);
            }
            out.push(']');
        }
        Py::Dict(entries) => {
            out.push('{');
            for (i, (key, item)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump_str(key, out);
                out.push_str(": ");
                dump_into(item, out);
            }
            out.push('}');
        }
    }
}

fn dump_str(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn probe_tools() -> Vec<Tool> {
        vec![Tool {
            name: "f".into(),
            description: None,
            parameters: json!({"type": "object", "properties": {
                "s": {"type": "string"}, "i": {"type": "integer"}, "n": {"type": "number"},
                "b": {"type": "boolean"}, "o": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "string"}}},
                "a": {"type": "array", "items": {"type": "number"}}, "any": {"description": "untyped"},
                "nul": {"type": ["string", "null"]}, "e": {"enum": [1, 2]},
                "u": {"anyOf": [{"type": "integer"}, {"type": "string"}]}, "fl": {"type": "float"},
                "ob": {"type": "object"}, "ar": {"type": "array"}, "nn": {"type": "null"}}}),
            strict: None,
        }]
    }

    fn param(name: &str, value: &str, string: Option<&str>) -> String {
        let attr = string
            .map(|s| format!(" string=\"{s}\""))
            .unwrap_or_default();
        format!("<｜DSML｜ parameter name=\"{name}\"{attr}>{value}</｜DSML｜ parameter>")
    }

    /// Expected outputs recorded from vLLM's deepseek_v41 parser in the prod image (tk3 `u` corpus).
    #[test]
    fn arguments_are_coerced_to_the_tool_schema_like_vllm() {
        let tools = probe_tools();
        for (name, value, string, expected) in [
            ("n", "4.0", Some("false"), r#"{"n": 4}"#),
            ("n", "4.0", Some("true"), r#"{"n": 4}"#),
            ("n", "1.5", Some("true"), r#"{"n": 1.5}"#),
            ("n", "1e3", Some("true"), r#"{"n": 1000}"#),
            ("n", "inf", Some("true"), r#"{"n": "inf"}"#),
            ("n", "nan", Some("true"), r#"{"n": "nan"}"#),
            (
                "n",
                "1e21",
                Some("false"),
                r#"{"n": 1000000000000000000000}"#,
            ),
            ("n", "1e-07", Some("false"), r#"{"n": 1e-07}"#),
            ("n", "-0.0", Some("false"), r#"{"n": 0}"#),
            ("n", "true", Some("false"), r#"{"n": true}"#),
            ("n", "1_000", Some("true"), r#"{"n": 1000}"#),
            ("n", " 2.50 ", Some("true"), r#"{"n": 2.5}"#),
            ("i", "12", Some("true"), r#"{"i": 12}"#),
            ("i", "4.0", Some("false"), r#"{"i": 4.0}"#),
            ("i", " 12 ", Some("true"), r#"{"i": 12}"#),
            ("i", "+7", Some("true"), r#"{"i": 7}"#),
            ("i", "1_0", Some("true"), r#"{"i": 10}"#),
            ("i", "abc", Some("true"), r#"{"i": "abc"}"#),
            ("i", "abc", None, r#"{"i": "abc"}"#),
            ("i", "[1, 2]", Some("true"), r#"{"i": [1, 2]}"#),
            (
                "i",
                "123456789012345678901234",
                Some("true"),
                r#"{"i": 123456789012345678901234}"#,
            ),
            ("s", "5", Some("false"), r#"{"s": "5"}"#),
            ("s", "null", Some("false"), r#"{"s": "null"}"#),
            ("s", "1e-07", Some("false"), r#"{"s": "1e-07"}"#),
            ("s", "true", Some("false"), r#"{"s": "true"}"#),
            ("s", r#"{"k": 1}"#, Some("false"), r#"{"s": {"k": 1}}"#),
            ("s", "[1, 2.0]", Some("false"), r#"{"s": [1, 2.0]}"#),
            ("b", "true", Some("true"), r#"{"b": true}"#),
            ("b", "1", Some("false"), r#"{"b": true}"#),
            ("b", "0", Some("true"), r#"{"b": false}"#),
            ("b", "yes", Some("true"), r#"{"b": "yes"}"#),
            ("b", " TRUE ", Some("true"), r#"{"b": true}"#),
            (
                "o",
                r#"{"x": "3", "y": 4, "z": "5"}"#,
                Some("false"),
                r#"{"o": {"x": 3, "y": "4", "z": "5"}}"#,
            ),
            ("o", r#"{"x": "3"}"#, Some("true"), r#"{"o": {"x": "3"}}"#),
            (
                "a",
                r#"["1", 2.0, "x", 3.5]"#,
                Some("false"),
                r#"{"a": [1, 2, "x", 3.5]}"#,
            ),
            ("a", "[1, 2]", Some("true"), r#"{"a": [1, 2]}"#),
            ("any", "7", Some("false"), r#"{"any": "7"}"#),
            ("any", r#"{"a": 1}"#, Some("false"), r#"{"any": {"a": 1}}"#),
            ("nul", "null", Some("false"), r#"{"nul": null}"#),
            ("nul", "null", Some("true"), r#"{"nul": null}"#),
            ("nul", "3", Some("false"), r#"{"nul": "3"}"#),
            ("e", "2", Some("true"), r#"{"e": 2}"#),
            ("u", "7", Some("true"), r#"{"u": 7}"#),
            ("u", "seven", Some("true"), r#"{"u": "seven"}"#),
            ("fl", "3.0", Some("false"), r#"{"fl": 3}"#),
            ("ob", "5", Some("true"), r#"{"ob": 5}"#),
            ("ar", r#"[1, "a"]"#, Some("true"), r#"{"ar": [1, "a"]}"#),
            ("nn", "NULL", Some("true"), r#"{"nn": null}"#),
            ("zz", "1.0", Some("false"), r#"{"zz": 1.0}"#),
        ] {
            let body = format!("\n{}\n", param(name, value, string));
            assert_eq!(
                invoke_arguments(&body, "f", &tools),
                expected,
                "{name}={value:?} string={string:?}"
            );
        }
        let body = param("n", "4.0", Some("false"));
        assert_eq!(
            invoke_arguments(&body, "nope", &tools),
            r#"{"n": 4.0}"#,
            "unknown tool"
        );
        assert_eq!(
            invoke_arguments(&body, "f", &[]),
            r#"{"n": 4.0}"#,
            "no tools"
        );
    }

    #[test]
    fn wrapper_arguments_are_unwrapped_like_vllm() {
        let tools = probe_tools();
        for (params, expected) in [
            (
                param("arguments", r#"{"s": "x", "n": 2.0}"#, Some("false")),
                r#"{"s": "x", "n": 2}"#,
            ),
            (
                param("input", r#"{"s": "y"}"#, Some("true")),
                r#"{"s": "y"}"#,
            ),
            (
                param("arguments", r#"{"zz": 1}"#, Some("false")),
                r#"{"arguments": {"zz": 1}}"#,
            ),
            (
                param("input", "{nope", Some("true")),
                r#"{"input": "{nope"}"#,
            ),
            (
                param("arguments", r#"{"s": "x"}"#, Some("false")) + &param("s", "y", Some("true")),
                r#"{"arguments": {"s": "x"}, "s": "y"}"#,
            ),
        ] {
            assert_eq!(invoke_arguments(&params, "f", &tools), expected, "{params}");
        }
    }

    #[test]
    fn parameters_follow_vllms_regex() {
        let open = "<｜DSML｜ parameter name=\"city\" string=\"true\">";
        for (body, expected) in [
            (
                format!(
                    "{open}Paris\n<｜DSML｜ parameter name=\"days\" string=\"false\">2</｜DSML｜ parameter>"
                ),
                r#"{"city": "Paris\n", "days": 2}"#,
            ),
            (format!("{open}Par"), "{}"),
            (
                "<｜DSML｜ parameter name=\"days\" string=\"false\">1\n".to_string(),
                "{}",
            ),
            (
                "<｜DSML｜ parameter  name=\"city\"  string=\"true\" >Rome</｜DSML｜ parameter>"
                    .to_string(),
                r#"{"city": "Rome"}"#,
            ),
            (
                format!("junk {}more", param("city", "Oslo", Some("true"))),
                r#"{"city": "Oslo"}"#,
            ),
            (
                param("city", "A", Some("true"))
                    + &param("days", "1", Some("false"))
                    + &param("city", "B", Some("true")),
                r#"{"city": "B", "days": 1}"#,
            ),
            ("<｜DSML｜ parameter name=\"ci".to_string(), "{}"),
            (param("v", "NaN", None), r#"{"v": NaN}"#),
            (
                param("v", "\"a\\u00e9\\ud83d\\ude00\\n\"", Some("false")),
                "{\"v\": \"aé😀\\n\"}",
            ),
            (param("v", "\u{1}x", Some("true")), r#"{"v": "\u0001x"}"#),
        ] {
            assert_eq!(
                dumps(&Py::Dict(dsml_parameters(&body))),
                expected,
                "{body:?}"
            );
        }
    }
}
