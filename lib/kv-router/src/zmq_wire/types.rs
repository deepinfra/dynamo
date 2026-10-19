// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Deserialize;
use serde::Serialize;

use crate::protocols::BlockExtraInfo;

use super::filter::{KvCacheEventMetadata, KvCacheSpecKind};

#[derive(Debug, Serialize)]
pub struct KvEventBatch {
    pub ts: f64,
    pub events: Vec<RawKvEvent>,
    #[serde(alias = "dp_rank")]
    pub data_parallel_rank: Option<i32>,
}

impl<'de> Deserialize<'de> for KvEventBatch {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Deserialize from array format: [timestamp, [events], data_parallel_rank]
        let arr: (f64, Vec<RawKvEvent>, Option<i32>) = Deserialize::deserialize(deserializer)?;
        Ok(KvEventBatch {
            ts: arr.0,
            events: arr.1,
            data_parallel_rank: arr.2,
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(untagged)]
pub enum BlockHashValue {
    Signed(i64),
    Unsigned(u64),
}

impl BlockHashValue {
    pub fn into_u64(self) -> u64 {
        match self {
            BlockHashValue::Signed(v) => v.cast_unsigned(),
            BlockHashValue::Unsigned(v) => v,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum KvTokenIds {
    Single(Vec<u32>),
    Bigram(Vec<(u32, u32)>),
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")] // msgspec encodes variant tag as a string when `tag=True`
pub enum RawKvEvent {
    BlockStored {
        /// Block hashes may be emitted as either signed or unsigned 64-bit values.
        /// We normalize them to `u64` while deserializing to support both producers.
        block_hashes: Vec<BlockHashValue>,
        parent_block_hash: Option<BlockHashValue>,
        token_ids: Vec<u32>,
        block_size: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        medium: Option<String>,
        /// LoRA adapter name for adapter-aware block hashing
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lora_name: Option<String>,
        /// Multimodal extra info for each block (length should match block_hashes)
        #[serde(default, skip_serializing_if = "Option::is_none")]
        block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_eagle: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        group_idx: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        kv_cache_spec_kind: Option<KvCacheSpecKind>,
        #[serde(skip_serializing_if = "Option::is_none")]
        kv_cache_spec_sliding_window: Option<u32>,
    },
    BlockRemoved {
        block_hashes: Vec<BlockHashValue>,
        #[serde(skip_serializing_if = "Option::is_none")]
        medium: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        group_idx: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        kv_cache_spec_kind: Option<KvCacheSpecKind>,
        #[serde(skip_serializing_if = "Option::is_none")]
        kv_cache_spec_sliding_window: Option<u32>,
    },
    AllBlocksCleared,
    Ignored,
}

impl RawKvEvent {
    pub fn event_type_label(&self) -> &'static str {
        match self {
            Self::BlockStored { .. } => "stored",
            Self::BlockRemoved { .. } => "removed",
            Self::AllBlocksCleared => "cleared",
            Self::Ignored => "ignored",
        }
    }

    pub fn is_ignored(&self) -> bool {
        matches!(self, Self::Ignored)
    }

    pub(crate) fn metadata(&self) -> KvCacheEventMetadata {
        match self {
            Self::BlockStored {
                group_idx,
                kv_cache_spec_kind,
                kv_cache_spec_sliding_window,
                ..
            }
            | Self::BlockRemoved {
                group_idx,
                kv_cache_spec_kind,
                kv_cache_spec_sliding_window,
                ..
            } => KvCacheEventMetadata {
                group_idx: *group_idx,
                kv_cache_spec_kind: *kv_cache_spec_kind,
                kv_cache_spec_sliding_window: *kv_cache_spec_sliding_window,
            },
            Self::AllBlocksCleared | Self::Ignored => KvCacheEventMetadata::default(),
        }
    }
}

/// One entry of a vLLM BlockStored block's `extra_keys`.
///
/// Older vLLM sends bare values (`"<mm id>"`, `["<mm id>", offset]`, the salt
/// and LoRA name as bare strings). vLLM with #59335 tags every key:
/// `["mm", id, offset]`, `["cache_salt", salt]`, `["lora", name, path]`,
/// `["prompt_embeds", <bytes>]`. `Other` keeps any further shape from failing
/// the whole event's decode.
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum ExtraKeyItem {
    Hash(String),
    HashWithSignedOffset((String, i64)),
    HashWithUnsignedOffset((String, u64)),
    Bytes(Vec<u8>),
    Signed(i64),
    Unsigned(u64),
    Float(f64),
    Bool(bool),
    Tagged((String, String)),
    TaggedWithSignedOffset((String, String, i64)),
    TaggedWithUnsignedOffset((String, String, u64)),
    TaggedPair((String, String, String)),
    Other(serde::de::IgnoredAny),
}
