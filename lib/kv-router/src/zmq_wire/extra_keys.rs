// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use xxhash_rust::xxh3;

use crate::protocols::{BlockExtraInfo, BlockMmObjectInfo};

use super::types::ExtraKeyItem;

/// Parse MM hash from extra_keys string:
/// - Only accept canonical vLLM MM identifiers (64-char hex digest)
/// - Convert by taking the first 16 hex chars as u64
pub fn parse_mm_hash_from_extra_key(s: &str) -> Option<u64> {
    // extra_keys mixes MM identifiers with LoRA/cache_salt/prompt-embed metadata.
    // Only MM identifiers should be mapped into BlockExtraInfo.
    if s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit()) {
        return u64::from_str_radix(&s[..16], 16).ok();
    }
    None
}

/// The u64 a vLLM `cache_salt` adds to the first block of its sequence, hashed
/// like an MM id: a 64-char hex salt parses as one, any other salt is its XXH3-64.
pub fn cache_salt_extra_hash(salt: &str) -> u64 {
    parse_mm_hash_from_extra_key(salt).unwrap_or_else(|| xxh3::xxh3_64(salt.as_bytes()))
}

/// Convert vLLM BlockStored extra_keys to block-level MM infos.
/// extra_keys is a list aligned with blocks:
/// - None => no MM content in that block
/// - ["hash1", "hash2", ...] => one or more MM objects in that block
/// - [[hash, start_offset], ...] => one or more MM objects with block-relative
///   start offsets (vLLM 0.19+)
pub fn extra_keys_to_block_mm_infos(
    extra_keys: Option<Vec<Option<Vec<ExtraKeyItem>>>>,
) -> Option<Vec<Option<BlockExtraInfo>>> {
    stored_extra_keys_to_block_infos(extra_keys, false, None)
}

/// [`extra_keys_to_block_mm_infos`] for a whole BlockStored event, which also keeps
/// the request's `cache_salt`.
///
/// vLLM appends the salt only to the extra keys of the sequence's first block
/// (`lora + mm + [cache_salt] + prompt_embeds`), so when the event starts the
/// sequence (no parent), a bare string in block 0 that is neither an MM id nor
/// the LoRA name is the salt. It joins that block's MM objects via
/// [`cache_salt_extra_hash`], so salted chains get their own tree path.
/// Tagged keys (vLLM #59335) name themselves: `("cache_salt", salt)` is the
/// salt and `("mm", id, offset)` an MM id; LoRA and prompt-embeds keys are
/// ignored as before (LoRA is hashed through `lora_name`).
pub fn stored_extra_keys_to_block_infos(
    extra_keys: Option<Vec<Option<Vec<ExtraKeyItem>>>>,
    starts_sequence: bool,
    lora_name: Option<&str>,
) -> Option<Vec<Option<BlockExtraInfo>>> {
    let extra_keys = extra_keys?;
    if extra_keys.is_empty() {
        return None;
    }

    let infos: Vec<Option<BlockExtraInfo>> = extra_keys
        .into_iter()
        .enumerate()
        .map(|(block_idx, block_keys)| {
            let salted_block = starts_sequence && block_idx == 0;
            let mm_objects: Vec<BlockMmObjectInfo> = block_keys
                .unwrap_or_default()
                .iter()
                .filter_map(|key| match key {
                    ExtraKeyItem::Hash(s) if salted_block && Some(s.as_str()) != lora_name => {
                        Some(cache_salt_extra_hash(s))
                    }
                    ExtraKeyItem::Hash(hash)
                    | ExtraKeyItem::HashWithSignedOffset((hash, _))
                    | ExtraKeyItem::HashWithUnsignedOffset((hash, _)) => {
                        parse_mm_hash_from_extra_key(hash)
                    }
                    ExtraKeyItem::Tagged((tag, salt)) if tag == "cache_salt" => {
                        Some(cache_salt_extra_hash(salt))
                    }
                    ExtraKeyItem::TaggedWithSignedOffset((tag, id, _))
                    | ExtraKeyItem::TaggedWithUnsignedOffset((tag, id, _))
                        if tag == "mm" =>
                    {
                        parse_mm_hash_from_extra_key(id)
                    }
                    ExtraKeyItem::Bytes(_)
                    | ExtraKeyItem::Signed(_)
                    | ExtraKeyItem::Unsigned(_)
                    | ExtraKeyItem::Float(_)
                    | ExtraKeyItem::Bool(_)
                    | ExtraKeyItem::Tagged(_)
                    | ExtraKeyItem::TaggedWithSignedOffset(_)
                    | ExtraKeyItem::TaggedWithUnsignedOffset(_)
                    | ExtraKeyItem::TaggedPair(_)
                    | ExtraKeyItem::Other(_) => None,
                })
                .map(|mm_hash| BlockMmObjectInfo {
                    mm_hash,
                    // vLLM extra_keys exposes MM start offsets but not MM lengths.
                    // Dynamo's block hash only depends on mm_hash today, so keep
                    // offsets empty rather than inventing a synthetic range.
                    offsets: vec![],
                })
                .collect();

            if mm_objects.is_empty() {
                None
            } else {
                Some(BlockExtraInfo { mm_objects })
            }
        })
        .collect();

    if infos.iter().all(|i| i.is_none()) {
        return None;
    }

    Some(infos)
}
