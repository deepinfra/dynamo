// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which policy class a prefill request is admitted under, by what the
//! replica set knows about its prompt.
//!
//! - **Seen**: enough of the prompt was routed to the pool recently, so it is
//!   a conversation the pool is serving. It keeps the class it asked for.
//! - **New**: not seen, and a short prompt: a conversation starting here. It
//!   uses `<class>-new` when the policy profile defines it.
//! - **Cold**: not seen, and a long prompt: a conversation arriving
//!   mid-way with history nobody here has computed. It uses `<class>-cold`
//!   when the profile defines it.
//!
//! The profile gives each class its own busy threshold and queue limits, so
//! an operator can always take seen conversations, take new ones while there
//! is headroom, and take cold ones only with plenty of it. A class the
//! profile does not define falls back to the class the request asked for, so
//! without `-new`/`-cold` classes nothing changes.

use std::sync::OnceLock;

pub const NEW_CLASS_SUFFIX: &str = "new";
pub const COLD_CLASS_SUFFIX: &str = "cold";

/// A prompt counts as seen once at least this share of it is known.
const DEFAULT_SEEN_MIN_CACHED_FRACTION: f64 = 0.5;
/// An unseen prompt up to this many tokens is a new conversation; a longer
/// one is treated as arriving mid-conversation.
const DEFAULT_NEW_MAX_PROMPT_TOKENS: usize = 32_768;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmissionClassSettings {
    pub seen_min_cached_fraction: f64,
    pub new_max_prompt_tokens: usize,
}

impl Default for AdmissionClassSettings {
    fn default() -> Self {
        Self {
            seen_min_cached_fraction: DEFAULT_SEEN_MIN_CACHED_FRACTION,
            new_max_prompt_tokens: DEFAULT_NEW_MAX_PROMPT_TOKENS,
        }
    }
}

impl AdmissionClassSettings {
    /// `DYN_ROUTER_SEEN_MIN_CACHED_FRACTION` (0..=1) and
    /// `DYN_ROUTER_NEW_MAX_PROMPT_TOKENS`, read once.
    pub fn from_env() -> Self {
        static SETTINGS: OnceLock<AdmissionClassSettings> = OnceLock::new();
        *SETTINGS.get_or_init(|| {
            Self::parse(
                std::env::var("DYN_ROUTER_SEEN_MIN_CACHED_FRACTION").ok().as_deref(),
                std::env::var("DYN_ROUTER_NEW_MAX_PROMPT_TOKENS").ok().as_deref(),
            )
        })
    }

    fn parse(seen_fraction: Option<&str>, new_max_tokens: Option<&str>) -> Self {
        let defaults = Self::default();
        Self {
            seen_min_cached_fraction: seen_fraction
                .and_then(|raw| raw.trim().parse::<f64>().ok())
                .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
                .unwrap_or(defaults.seen_min_cached_fraction),
            new_max_prompt_tokens: new_max_tokens
                .and_then(|raw| raw.trim().parse::<usize>().ok())
                .unwrap_or(defaults.new_max_prompt_tokens),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptFamiliarity {
    Seen,
    New,
    Cold,
}

impl PromptFamiliarity {
    pub fn of(prompt_tokens: usize, known_tokens: usize, settings: AdmissionClassSettings) -> Self {
        let known = known_tokens.min(prompt_tokens);
        let seen_from = settings.seen_min_cached_fraction * prompt_tokens as f64;
        if prompt_tokens == 0 || (known > 0 && known as f64 >= seen_from) {
            Self::Seen
        } else if prompt_tokens <= settings.new_max_prompt_tokens {
            Self::New
        } else {
            Self::Cold
        }
    }

    pub fn class_suffix(self) -> Option<&'static str> {
        match self {
            Self::Seen => None,
            Self::New => Some(NEW_CLASS_SUFFIX),
            Self::Cold => Some(COLD_CLASS_SUFFIX),
        }
    }
}

/// The class to admit under: `<base>-<suffix>` when the profile defines it,
/// else what the request asked for. `base` is the requested class, or the
/// profile's default class for a request that named none.
pub fn admission_class(
    requested: Option<String>,
    default_class: &str,
    familiarity: PromptFamiliarity,
    class_exists: impl Fn(&str) -> bool,
) -> Option<String> {
    let Some(suffix) = familiarity.class_suffix() else {
        return requested;
    };
    let derived = format!("{}-{suffix}", requested.as_deref().unwrap_or(default_class));
    if class_exists(&derived) {
        Some(derived)
    } else {
        requested
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn familiarity(prompt: usize, known: usize) -> PromptFamiliarity {
        PromptFamiliarity::of(prompt, known, AdmissionClassSettings::default())
    }

    #[test]
    fn a_prompt_is_seen_new_or_cold() {
        // A continuing conversation: almost all of it is known.
        assert_eq!(familiarity(150_000, 148_000), PromptFamiliarity::Seen);
        assert_eq!(familiarity(20_000, 10_000), PromptFamiliarity::Seen);
        // A first turn: nothing known, short.
        assert_eq!(familiarity(9_000, 0), PromptFamiliarity::New);
        assert_eq!(familiarity(32_768, 2_048), PromptFamiliarity::New);
        // A conversation arriving mid-way: long, with at most a shared
        // system prompt known.
        assert_eq!(familiarity(150_000, 0), PromptFamiliarity::Cold);
        assert_eq!(familiarity(150_000, 8_192), PromptFamiliarity::Cold);
        // More known tokens than the prompt has is clamped, not trusted.
        assert_eq!(familiarity(1_000, 9_999), PromptFamiliarity::Seen);
        assert_eq!(familiarity(0, 0), PromptFamiliarity::Seen);
    }

    #[test]
    fn derived_classes_are_used_only_when_the_profile_defines_them() {
        let defined = |name: &str| matches!(name, "no-queue-cold" | "standard-new");
        let class = |requested: Option<&str>, familiarity| {
            admission_class(requested.map(str::to_string), "standard", familiarity, defined)
        };
        assert_eq!(class(Some("no-queue"), PromptFamiliarity::Seen), Some("no-queue".into()));
        assert_eq!(class(Some("no-queue"), PromptFamiliarity::Cold), Some("no-queue-cold".into()));
        // `no-queue-new` is not defined: stay on the requested class.
        assert_eq!(class(Some("no-queue"), PromptFamiliarity::New), Some("no-queue".into()));
        // A request naming no class derives from the default class.
        assert_eq!(class(None, PromptFamiliarity::New), Some("standard-new".into()));
        assert_eq!(class(None, PromptFamiliarity::Cold), None);
        assert_eq!(class(None, PromptFamiliarity::Seen), None);
    }

    #[test]
    fn settings_fall_back_to_defaults_on_bad_values() {
        let defaults = AdmissionClassSettings::default();
        assert_eq!(AdmissionClassSettings::parse(None, None), defaults);
        assert_eq!(AdmissionClassSettings::parse(Some("2"), Some("many")), defaults);
        let custom = AdmissionClassSettings::parse(Some("0.8"), Some("16384"));
        assert_eq!(custom.seen_min_cached_fraction, 0.8);
        assert_eq!(custom.new_max_prompt_tokens, 16_384);
    }
}
