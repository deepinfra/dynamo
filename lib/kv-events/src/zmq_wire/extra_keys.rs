// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_hashing::multimodal::{
    BlockExtraInfo, BlockMmObjectInfo, parse_mm_hash_from_extra_key,
};

use super::types::ExtraKeyItem;

// Must match _DYNAMO_CACHE_SALT_PREFIX in components/src/dynamo/vllm/handlers.py.
pub const DYNAMO_CACHE_SALT_PREFIX: &str = "dynamo-cache-salt:";

/// Extract a vLLM cache salt from `extra_keys` when a producer does not emit
/// top-level `cache_salt`. vLLM aligns `extra_keys` with blocks and includes
/// cache salt only in the first block. Dynamo tags the opaque value before
/// passing it to vLLM because bare strings are otherwise ambiguous with LoRA
/// names and legacy multimodal hashes. The first matching LoRA item is skipped
/// before looking for the tag so a salt equal to the LoRA name still works.
pub fn extra_keys_to_cache_namespace(
    extra_keys: Option<&[Option<Vec<ExtraKeyItem>>]>,
    lora_name: Option<&str>,
) -> Option<String> {
    let first_block = extra_keys?.first()?.as_ref()?;
    let mut unmatched_lora = lora_name.filter(|name| !name.is_empty());
    first_block.iter().find_map(|key| {
        let ExtraKeyItem::Hash(value) = key else {
            return None;
        };
        if unmatched_lora.is_some_and(|name| name == value) {
            unmatched_lora = None;
            return None;
        }
        value
            .strip_prefix(DYNAMO_CACHE_SALT_PREFIX)
            .filter(|namespace| !namespace.is_empty())
            .map(str::to_owned)
    })
}

/// DeepInfra: with `DI_KV_NATIVE_MM_HASH=1`, a native vLLM MM identifier in `extra_keys` (any
/// string key that is neither a Dynamo routing marker nor the tagged cache salt) also becomes a
/// block MM object, keyed by its xxh3. Without it, blocks holding different images or audio clips
/// get the same router token hash while vLLM's block hashes differ, which breaks the indexer's
/// one-local-chain-one-external-chain invariant and leaves stale entries (MiMo-V2.6-Flash: 1.5%
/// of stored blocks). Only for models without exact MM routing, where these infos are mixed into
/// the token hash and no placeholder is rewritten.
fn native_mm_hash_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("DI_KV_NATIVE_MM_HASH").ok().as_deref(),
            Some("1" | "true")
        )
    })
}

fn native_mm_hash(s: &str) -> Option<u64> {
    (!s.is_empty() && !s.starts_with(DYNAMO_CACHE_SALT_PREFIX))
        .then(|| xxhash_rust::xxh3::xxh3_64(s.as_bytes()))
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
    extra_keys_to_block_mm_infos_with(extra_keys, native_mm_hash_enabled())
}

fn extra_keys_to_block_mm_infos_with(
    extra_keys: Option<Vec<Option<Vec<ExtraKeyItem>>>>,
    native: bool,
) -> Option<Vec<Option<BlockExtraInfo>>> {
    let extra_keys = extra_keys?;
    if extra_keys.is_empty() {
        return None;
    }

    let infos: Vec<Option<BlockExtraInfo>> = extra_keys
        .into_iter()
        .map(|block_keys| {
            let mm_objects: Vec<BlockMmObjectInfo> = block_keys
                .unwrap_or_default()
                .iter()
                .filter_map(|key| match key {
                    ExtraKeyItem::Hash(hash)
                    | ExtraKeyItem::HashWithSignedOffset((hash, _))
                    | ExtraKeyItem::HashWithUnsignedOffset((hash, _)) => {
                        parse_mm_hash_from_extra_key(hash)
                            .or_else(|| native.then(|| native_mm_hash(hash)).flatten())
                    }
                    ExtraKeyItem::Bytes(_)
                    | ExtraKeyItem::Signed(_)
                    | ExtraKeyItem::Unsigned(_)
                    | ExtraKeyItem::Float(_)
                    | ExtraKeyItem::Bool(_) => None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_hashing::multimodal::mark_mm_hash_for_extra_key;

    #[test]
    fn di_native_mm_hashes_separate_blocks_only_when_enabled() {
        use dynamo_kv_hashing::routing::{BlockHashOptions, compute_block_hash_for_seq};
        let native = |h: &str| Some(vec![Some(vec![ExtraKeyItem::Hash(h.to_string())])]);
        let (img_a, img_b) = ("a".repeat(64), "b".repeat(64));
        // Off: native hashes are ignored, as upstream.
        assert_eq!(extra_keys_to_block_mm_infos_with(native(&img_a), false), None);
        // On: two images give two MM objects, so the same placeholder tokens hash differently.
        let infos_a = extra_keys_to_block_mm_infos_with(native(&img_a), true).unwrap();
        let infos_b = extra_keys_to_block_mm_infos_with(native(&img_b), true).unwrap();
        let tokens = [7u32; 4];
        let hash = |infos: &[Option<BlockExtraInfo>]| {
            compute_block_hash_for_seq(
                &tokens,
                4,
                BlockHashOptions {
                    block_mm_infos: Some(infos),
                    lora_name: None,
                    cache_namespace: None,
                    is_eagle: None,
                },
            )[0]
        };
        assert_ne!(hash(&infos_a), hash(&infos_b));
        // The tagged cache salt and Dynamo routing markers keep their meaning.
        let salt = Some(vec![Some(vec![ExtraKeyItem::Hash(format!("{DYNAMO_CACHE_SALT_PREFIX}t"))])]);
        assert_eq!(extra_keys_to_block_mm_infos_with(salt, true), None);
        let marker = mark_mm_hash_for_extra_key(42);
        let infos = extra_keys_to_block_mm_infos_with(native(&marker), true).unwrap();
        assert_eq!(infos[0].as_ref().unwrap().mm_objects[0].mm_hash, 42);
    }

    #[test]
    fn prompt_embedding_bytes_are_not_cache_namespace() {
        let extra_keys = [Some(vec![ExtraKeyItem::Bytes(b"prompt-embed".to_vec())])];
        assert_eq!(extra_keys_to_cache_namespace(Some(&extra_keys), None), None);
    }

    #[test]
    fn untagged_strings_are_not_cache_namespaces() {
        let extra_keys = [Some(vec![ExtraKeyItem::Hash("tenant-a".to_string())])];
        assert_eq!(extra_keys_to_cache_namespace(Some(&extra_keys), None), None);
    }

    #[test]
    fn tagged_cache_namespace_is_decoded() {
        let extra_keys = [Some(vec![ExtraKeyItem::Hash(
            "dynamo-cache-salt:tenant-a".to_string(),
        )])];
        assert_eq!(
            extra_keys_to_cache_namespace(Some(&extra_keys), None).as_deref(),
            Some("tenant-a")
        );
    }

    #[test]
    fn cache_namespace_equal_to_lora_name_is_decoded() {
        let extra_keys = [Some(vec![
            ExtraKeyItem::Hash("adapter-a".to_string()),
            ExtraKeyItem::Hash("dynamo-cache-salt:adapter-a".to_string()),
        ])];
        assert_eq!(
            extra_keys_to_cache_namespace(Some(&extra_keys), Some("adapter-a")).as_deref(),
            Some("adapter-a")
        );
    }
}
