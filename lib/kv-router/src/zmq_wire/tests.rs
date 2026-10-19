// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use rmp_serde::{from_slice, to_vec};

use crate::protocols::{
    BlockExtraInfo, BlockHashOptions, BlockMmObjectInfo, ExternalSequenceBlockHash,
    KvCacheEventData, StorageTier, WorkerWithDpRank, compute_block_hash_for_seq,
};

use super::filter::KvCacheSpecKind;
use super::*;

#[derive(Clone, Copy, Debug)]
enum TestEventKind {
    BlockStored,
    BlockRemoved,
}

#[test]
fn test_deserialize_bigram_block_stored_sequence() {
    let raw_event = (
        "BlockStored",
        vec![BlockHashValue::Unsigned(11), BlockHashValue::Unsigned(12)],
        Option::<BlockHashValue>::None,
        vec![(10u32, 11u32), (11, 12), (12, 13), (13, 14)],
        2usize,
        Option::<u64>::None,
        Option::<String>::None,
        Option::<String>::None,
    );
    let encoded = to_vec(&raw_event).unwrap();
    let event: RawKvEvent = from_slice(&encoded).unwrap();

    match event {
        RawKvEvent::BlockStored {
            token_ids,
            block_size,
            is_eagle,
            ..
        } => {
            assert_eq!(token_ids, vec![10, 11, 12, 13, 14]);
            assert_eq!(block_size, 2);
            assert_eq!(is_eagle, Some(true));
        }
        other => panic!("expected BlockStored, got {other:?}"),
    }
}

fn block_stored_sequence(
    group_idx: Option<u32>,
    kv_cache_spec_kind: Option<&'static str>,
) -> Vec<u8> {
    match (group_idx, kv_cache_spec_kind) {
        (Some(group_idx), Some(kv_cache_spec_kind)) => to_vec(&(
            "BlockStored",
            vec![BlockHashValue::Unsigned(11)],
            Option::<BlockHashValue>::None,
            vec![10u32, 11],
            2usize,
            Option::<u64>::None,
            Option::<String>::None,
            Option::<String>::None,
            Option::<u8>::None,
            group_idx,
            kv_cache_spec_kind,
        ))
        .unwrap(),
        (Some(group_idx), None) => to_vec(&(
            "BlockStored",
            vec![BlockHashValue::Unsigned(11)],
            Option::<BlockHashValue>::None,
            vec![10u32, 11],
            2usize,
            Option::<u64>::None,
            Option::<String>::None,
            Option::<String>::None,
            Option::<u8>::None,
            group_idx,
        ))
        .unwrap(),
        (None, Some(kv_cache_spec_kind)) => to_vec(&(
            "BlockStored",
            vec![BlockHashValue::Unsigned(11)],
            Option::<BlockHashValue>::None,
            vec![10u32, 11],
            2usize,
            Option::<u64>::None,
            Option::<String>::None,
            Option::<String>::None,
            Option::<u8>::None,
            Option::<u32>::None,
            kv_cache_spec_kind,
        ))
        .unwrap(),
        (None, None) => to_vec(&(
            "BlockStored",
            vec![BlockHashValue::Unsigned(11)],
            Option::<BlockHashValue>::None,
            vec![10u32, 11],
            2usize,
            Option::<u64>::None,
            Option::<String>::None,
            Option::<String>::None,
        ))
        .unwrap(),
    }
}

fn block_removed_sequence(
    group_idx: Option<u32>,
    kv_cache_spec_kind: Option<&'static str>,
) -> Vec<u8> {
    match (group_idx, kv_cache_spec_kind) {
        (Some(group_idx), Some(kv_cache_spec_kind)) => to_vec(&(
            "BlockRemoved",
            vec![BlockHashValue::Unsigned(11)],
            Option::<String>::None,
            group_idx,
            kv_cache_spec_kind,
        ))
        .unwrap(),
        (Some(group_idx), None) => to_vec(&(
            "BlockRemoved",
            vec![BlockHashValue::Unsigned(11)],
            Option::<String>::None,
            group_idx,
        ))
        .unwrap(),
        (None, Some(kv_cache_spec_kind)) => to_vec(&(
            "BlockRemoved",
            vec![BlockHashValue::Unsigned(11)],
            Option::<String>::None,
            Option::<u32>::None,
            kv_cache_spec_kind,
        ))
        .unwrap(),
        (None, None) => to_vec(&(
            "BlockRemoved",
            vec![BlockHashValue::Unsigned(11)],
            Option::<String>::None,
        ))
        .unwrap(),
    }
}

fn sequence_with_group_idx(event_kind: TestEventKind, group_idx: Option<u32>) -> Vec<u8> {
    match event_kind {
        TestEventKind::BlockStored => block_stored_sequence(group_idx, None),
        TestEventKind::BlockRemoved => block_removed_sequence(group_idx, None),
    }
}

fn sequence_with_cache_spec_kind(
    event_kind: TestEventKind,
    group_idx: Option<u32>,
    kv_cache_spec_kind: &'static str,
) -> Vec<u8> {
    match event_kind {
        TestEventKind::BlockStored => block_stored_sequence(group_idx, Some(kv_cache_spec_kind)),
        TestEventKind::BlockRemoved => block_removed_sequence(group_idx, Some(kv_cache_spec_kind)),
    }
}

fn sequence_with_cache_spec_kind_without_group_idx_slot(
    event_kind: TestEventKind,
    kv_cache_spec_kind: &'static str,
) -> Vec<u8> {
    match event_kind {
        TestEventKind::BlockStored => to_vec(&(
            "BlockStored",
            vec![BlockHashValue::Unsigned(11)],
            Option::<BlockHashValue>::None,
            vec![10u32, 11],
            2usize,
            Option::<u64>::None,
            Option::<String>::None,
            Option::<String>::None,
            Option::<u8>::None,
            kv_cache_spec_kind,
        ))
        .unwrap(),
        TestEventKind::BlockRemoved => to_vec(&(
            "BlockRemoved",
            vec![BlockHashValue::Unsigned(11)],
            Option::<String>::None,
            kv_cache_spec_kind,
        ))
        .unwrap(),
    }
}

fn assert_parsed_event_kind(event: RawKvEvent, expected_kind: TestEventKind) {
    match (event, expected_kind) {
        (RawKvEvent::BlockStored { .. }, TestEventKind::BlockStored)
        | (RawKvEvent::BlockRemoved { .. }, TestEventKind::BlockRemoved) => {}
        (event, expected_kind) => {
            panic!("expected {expected_kind:?}, got {event:?}");
        }
    }
}

fn assert_event_metadata(
    event: &RawKvEvent,
    expected_group_idx: Option<u32>,
    expected_kind: Option<KvCacheSpecKind>,
    expected_sliding_window: Option<u32>,
) {
    let metadata = event.metadata();
    assert_eq!(metadata.group_idx, expected_group_idx);
    assert_eq!(metadata.kv_cache_spec_kind, expected_kind);
    assert_eq!(
        metadata.kv_cache_spec_sliding_window,
        expected_sliding_window
    );
}

#[test]
fn test_deserialize_sequence_accepts_main_group_idx() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent = from_slice(&sequence_with_group_idx(event_kind, Some(0))).unwrap();

        assert_event_metadata(&event, Some(0), None, None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_deserialize_sequence_preserves_non_main_group_idx() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent = from_slice(&sequence_with_group_idx(event_kind, Some(1))).unwrap();

        assert_event_metadata(&event, Some(1), None, None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_deserialize_sequence_accepts_missing_group_idx() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent = from_slice(&sequence_with_group_idx(event_kind, None)).unwrap();

        assert_event_metadata(&event, None, None, None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_deserialize_sequence_accepts_main_attention_kind_with_nonzero_group_idx() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent = from_slice(&sequence_with_cache_spec_kind(
            event_kind,
            Some(3),
            "full_attention",
        ))
        .unwrap();

        assert_event_metadata(&event, Some(3), Some(KvCacheSpecKind::FullAttention), None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_deserialize_sequence_accepts_main_attention_kind_without_group_idx_slot() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent = from_slice(&sequence_with_cache_spec_kind_without_group_idx_slot(
            event_kind,
            "full_attention",
        ))
        .unwrap();

        assert_event_metadata(&event, None, Some(KvCacheSpecKind::FullAttention), None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_deserialize_block_stored_sequence_preserves_block_mm_infos_and_metadata() {
    let block_mm_infos = vec![Some(BlockExtraInfo {
        mm_objects: vec![BlockMmObjectInfo {
            mm_hash: 99,
            offsets: vec![(0, 1)],
        }],
    })];
    let raw_event = (
        "BlockStored",
        vec![BlockHashValue::Unsigned(11)],
        Option::<BlockHashValue>::None,
        vec![10u32, 11],
        2usize,
        Option::<u64>::None,
        Option::<String>::None,
        Option::<String>::None,
        Option::<u8>::None,
        block_mm_infos.clone(),
        3u32,
        "full_attention",
    );
    let encoded = to_vec(&raw_event).unwrap();
    let event: RawKvEvent = from_slice(&encoded).unwrap();

    match &event {
        RawKvEvent::BlockStored {
            block_mm_infos: Some(parsed),
            ..
        } => assert_eq!(parsed, &block_mm_infos),
        other => panic!("expected BlockStored with block_mm_infos, got {other:?}"),
    }
    assert_event_metadata(&event, Some(3), Some(KvCacheSpecKind::FullAttention), None);

    let remove: RawKvEvent =
        from_slice(&block_removed_sequence(Some(3), None)).expect("valid remove event");
    let mut normalizer = ZmqEventNormalizer::new(2);
    let worker = WorkerWithDpRank::new(7, 0);

    assert!(normalizer.preprocess(event, worker).is_some());
    assert!(normalizer.preprocess(remove, worker).is_some());
}

#[test]
fn test_deserialize_sequence_preserves_non_main_attention_kind_with_group_idx_zero() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent =
            from_slice(&sequence_with_cache_spec_kind(event_kind, Some(0), "mamba")).unwrap();

        assert_event_metadata(&event, Some(0), Some(KvCacheSpecKind::Mamba), None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_normalizer_ignores_non_main_group_idx_without_metadata() {
    let raw_event: RawKvEvent =
        from_slice(&block_removed_sequence(Some(1), None)).expect("valid raw event");
    let mut normalizer = ZmqEventNormalizer::new(2);

    assert_eq!(
        normalizer
            .preprocess_with_reason(raw_event, WorkerWithDpRank::new(3, 0))
            .unwrap_err(),
        ZmqEventFilterReason::UnlearnedGroupIdx
    );
}

#[test]
fn test_normalizer_ignores_map_serialized_non_main_attention_kind() {
    #[derive(serde::Serialize)]
    struct MapBlockStoredEvent {
        #[serde(rename = "type")]
        event_type: &'static str,
        block_hashes: Vec<u64>,
        parent_block_hash: Option<u64>,
        token_ids: Vec<u32>,
        block_size: usize,
        group_idx: Option<u32>,
        kv_cache_spec_kind: Option<&'static str>,
    }

    let event = MapBlockStoredEvent {
        event_type: "BlockStored",
        block_hashes: vec![11],
        parent_block_hash: None,
        token_ids: vec![10, 11],
        block_size: 2,
        group_idx: Some(1),
        kv_cache_spec_kind: Some("mamba"),
    };
    let encoded = rmp_serde::to_vec_named(&(0.0, vec![event], Some(0_i32)))
        .expect("serialize raw event batch");
    let mut batch = decode_event_batch(&encoded).expect("deserialize raw event batch");
    let decoded = batch.events.pop().expect("batch should contain event");
    let mut normalizer = ZmqEventNormalizer::new(2);

    assert_event_metadata(&decoded, Some(1), Some(KvCacheSpecKind::Mamba), None);
    assert_eq!(
        normalizer
            .preprocess_with_reason(decoded, WorkerWithDpRank::new(3, 0))
            .unwrap_err(),
        ZmqEventFilterReason::NonMainAttentionKind
    );
}

#[test]
fn test_normalizer_learns_main_attention_metadata_for_remove() {
    let store: RawKvEvent = from_slice(&sequence_with_cache_spec_kind(
        TestEventKind::BlockStored,
        Some(3),
        "full_attention",
    ))
    .expect("valid store event");
    let remove: RawKvEvent =
        from_slice(&block_removed_sequence(Some(3), None)).expect("valid remove event");
    let mut normalizer = ZmqEventNormalizer::new(2);
    let worker = WorkerWithDpRank::new(7, 0);

    assert!(normalizer.preprocess(store, worker).is_some());
    assert!(normalizer.preprocess(remove, worker).is_some());
}

#[test]
fn test_normalizer_metadata_is_dp_rank_scoped() {
    let store: RawKvEvent = from_slice(&sequence_with_cache_spec_kind(
        TestEventKind::BlockStored,
        Some(3),
        "full_attention",
    ))
    .expect("valid store event");
    let same_rank_remove: RawKvEvent =
        from_slice(&block_removed_sequence(Some(3), None)).expect("valid same-rank remove event");
    let different_rank_remove: RawKvEvent = from_slice(&block_removed_sequence(Some(3), None))
        .expect("valid different-rank remove event");
    let mut normalizer = ZmqEventNormalizer::new(2);

    assert!(
        normalizer
            .preprocess(store, WorkerWithDpRank::new(7, 0))
            .is_some()
    );
    assert!(
        normalizer
            .preprocess(same_rank_remove, WorkerWithDpRank::new(7, 0))
            .is_some()
    );
    assert!(
        normalizer
            .preprocess(different_rank_remove, WorkerWithDpRank::new(7, 1))
            .is_none()
    );
}

#[test]
fn test_normalizer_does_not_learn_metadata_from_remove_events() {
    let metadata_remove: RawKvEvent = from_slice(&sequence_with_cache_spec_kind(
        TestEventKind::BlockRemoved,
        Some(3),
        "full_attention",
    ))
    .expect("valid metadata remove event");
    let bare_remove: RawKvEvent =
        from_slice(&block_removed_sequence(Some(3), None)).expect("valid bare remove event");
    let mut normalizer = ZmqEventNormalizer::new(2);
    let worker = WorkerWithDpRank::new(7, 0);

    assert!(normalizer.preprocess(metadata_remove, worker).is_some());
    assert!(normalizer.preprocess(bare_remove, worker).is_none());
}

#[test]
fn test_normalizer_ignores_non_main_attention_kind_with_group_idx_zero() {
    let raw_event: RawKvEvent = from_slice(&sequence_with_cache_spec_kind(
        TestEventKind::BlockStored,
        Some(0),
        "mamba",
    ))
    .expect("valid raw event");
    let remove: RawKvEvent =
        from_slice(&block_removed_sequence(Some(0), None)).expect("valid remove event");
    let mut normalizer = ZmqEventNormalizer::new(2);
    let worker = WorkerWithDpRank::new(3, 0);

    assert!(normalizer.preprocess(raw_event, worker).is_none());
    assert!(normalizer.preprocess(remove, worker).is_none());
}

#[test]
fn test_convert_event_bigram_emits_eagle_windows() {
    let raw_event = RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(21), BlockHashValue::Unsigned(22)],
        parent_block_hash: None,
        token_ids: vec![10, 11, 12, 13, 14],
        block_size: 2,
        medium: None,
        lora_name: None,
        block_mm_infos: None,
        is_eagle: Some(true),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    };
    let warning_count = Arc::new(AtomicU32::new(0));
    let placement_event =
        convert_event(raw_event, 7, 2, WorkerWithDpRank::new(3, 0), &warning_count)
            .expect("block size matches");

    match placement_event.unwrap().event.data {
        KvCacheEventData::Stored(store_data) => {
            assert_eq!(store_data.blocks.len(), 2);
            assert_eq!(
                store_data.blocks[0].block_hash,
                ExternalSequenceBlockHash(21)
            );
            assert_eq!(
                store_data.blocks[1].block_hash,
                ExternalSequenceBlockHash(22)
            );

            let expected_first = compute_block_hash_for_seq(
                &[10, 11, 12],
                2,
                BlockHashOptions {
                    is_eagle: Some(true),
                    ..Default::default()
                },
            );
            let expected_second = compute_block_hash_for_seq(
                &[12, 13, 14],
                2,
                BlockHashOptions {
                    is_eagle: Some(true),
                    ..Default::default()
                },
            );

            assert_eq!(store_data.blocks[0].tokens_hash, expected_first[0]);
            assert_eq!(store_data.blocks[1].tokens_hash, expected_second[0]);
        }
        other => panic!("expected Stored event, got {other:?}"),
    }
}

#[test]
fn test_convert_event_block_size_mismatch_is_fatal() {
    let raw_event = RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(21)],
        parent_block_hash: None,
        token_ids: vec![10; 256],
        block_size: 256,
        medium: None,
        lora_name: None,
        block_mm_infos: None,
        is_eagle: None,
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    };
    let warning_count = Arc::new(AtomicU32::new(0));
    let result = convert_event(raw_event, 7, 128, WorkerWithDpRank::new(3, 0), &warning_count);

    assert_eq!(
        result.unwrap_err(),
        ConvertError::BlockSizeMismatch {
            event_block_size: 256,
            configured_block_size: 128,
        }
    );
}

#[test]
fn test_convert_event_partial_prefix_entry_is_skipped() {
    // vLLM with prefix_match_unit (hash_block_size) < block_size publishes the
    // prompt tail that ends inside a cache block as a BlockStored whose
    // block_size is the sub-block length (32/64/96 for a 128-token block).
    // That is not a --block-size misconfiguration: drop it, don't exit.
    for partial in [32usize, 64, 96] {
        let raw_event = RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(21)],
            parent_block_hash: Some(BlockHashValue::Unsigned(9)),
            token_ids: vec![10; partial],
            block_size: partial,
            medium: None,
            lora_name: None,
            block_mm_infos: None,
            is_eagle: None,
            group_idx: Some(4),
            kv_cache_spec_kind: Some(KvCacheSpecKind::MlaAttention),
            kv_cache_spec_sliding_window: None,
        };
        let warning_count = Arc::new(AtomicU32::new(0));
        let result =
            convert_event(raw_event, 7, 128, WorkerWithDpRank::new(3, 0), &warning_count)
                .expect("partial-prefix entry is not a config error");
        assert!(result.is_none(), "partial entry of {partial} tokens must be dropped");
    }
    // Anything smaller is a partial entry; equal/larger is not.
    assert!(is_partial_prefix_entry(96, 128));
    assert!(!is_partial_prefix_entry(256, 128));
    assert!(!is_partial_prefix_entry(128, 128));
    assert!(is_partial_prefix_entry(32, 128));
}

#[test]
fn test_convert_event_empty_store_is_not_fatal() {
    // No blocks to publish -> nothing can mismatch; must not error.
    let raw_event = RawKvEvent::BlockStored {
        block_hashes: vec![],
        parent_block_hash: Some(BlockHashValue::Unsigned(9)),
        token_ids: vec![],
        block_size: 256,
        medium: None,
        lora_name: None,
        block_mm_infos: None,
        is_eagle: None,
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    };
    let warning_count = Arc::new(AtomicU32::new(0));
    let placement_event =
        convert_event(raw_event, 7, 128, WorkerWithDpRank::new(3, 0), &warning_count)
            .expect("empty store is not a config error");

    match placement_event.unwrap().event.data {
        KvCacheEventData::Stored(store_data) => assert!(store_data.blocks.is_empty()),
        other => panic!("expected Stored event, got {other:?}"),
    }
}

#[test]
fn test_convert_event_short_token_ids_keeps_parsed_blocks() {
    // Two hashes but token_ids only covers the first block: the second is
    // dropped (loudly), the first is still published.
    let raw_event = RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(21), BlockHashValue::Unsigned(22)],
        parent_block_hash: None,
        token_ids: vec![10, 11],
        block_size: 2,
        medium: None,
        lora_name: None,
        block_mm_infos: None,
        is_eagle: None,
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    };
    let warning_count = Arc::new(AtomicU32::new(0));
    let placement_event =
        convert_event(raw_event, 7, 2, WorkerWithDpRank::new(3, 0), &warning_count)
            .expect("short token_ids is not a config error");

    match placement_event.unwrap().event.data {
        KvCacheEventData::Stored(store_data) => {
            assert_eq!(store_data.blocks.len(), 1);
            assert_eq!(
                store_data.blocks[0].block_hash,
                ExternalSequenceBlockHash(21)
            );
        }
        other => panic!("expected Stored event, got {other:?}"),
    }
}

#[test]
fn test_convert_event_cpu_medium_lands_on_host_tier() {
    // vLLM's SimpleCPUOffloadConnector tags its events medium="CPU". Before
    // #10368 this string was unknown and fell back to the Device tier, so a
    // CPU-tier eviction removed a block the GPU still held.
    for (medium, expected) in [
        (Some("CPU"), StorageTier::HostPinned),
        (Some("CPU_PINNED"), StorageTier::HostPinned),
        (Some("GPU"), StorageTier::Device),
        (None, StorageTier::Device),
    ] {
        let raw_event = RawKvEvent::BlockRemoved {
            block_hashes: vec![BlockHashValue::Unsigned(201)],
            medium: medium.map(str::to_string),
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
        };
        let warning_count = Arc::new(AtomicU32::new(0));
        let placement = convert_event(raw_event, 7, 4, WorkerWithDpRank::new(3, 0), &warning_count)
            .expect("remove event converts")
            .expect("remove event is kept");
        assert_eq!(placement.placement.tier, expected, "medium {medium:?}");
    }
}

#[derive(serde::Serialize)]
#[serde(untagged)]
enum WireExtraKey {
    Bare(String),
    WithOffset((String, i64)),
    Tagged((String, String)),
    TaggedWithOffset((String, String, i64)),
    TaggedPair((String, String, String)),
}

fn encode_stored(
    parent: Option<u64>,
    lora_name: Option<&str>,
    extra_keys: Vec<Option<Vec<WireExtraKey>>>,
) -> Vec<u8> {
    to_vec(&(
        "BlockStored",
        vec![BlockHashValue::Unsigned(21), BlockHashValue::Unsigned(22)],
        parent.map(BlockHashValue::Unsigned),
        vec![10u32, 11, 12, 13],
        2usize,
        Option::<u64>::None,
        Option::<String>::None,
        lora_name.map(str::to_string),
        Some(extra_keys),
    ))
    .unwrap()
}

fn decoded_tokens_hashes(encoded: &[u8]) -> Vec<u64> {
    let raw_event: RawKvEvent = from_slice(encoded).expect("event decodes");
    let warning_count = Arc::new(AtomicU32::new(0));
    let placement = convert_event(raw_event, 7, 2, WorkerWithDpRank::new(3, 0), &warning_count)
        .expect("block size matches")
        .expect("store is kept");
    match placement.event.data {
        KvCacheEventData::Stored(store_data) => {
            store_data.blocks.iter().map(|b| b.tokens_hash.0).collect()
        }
        other => panic!("expected Stored event, got {other:?}"),
    }
}

#[test]
fn test_tagged_cache_salt_and_mm_keys() {
    let salted = encode_stored(
        None,
        None,
        vec![Some(vec![WireExtraKey::Tagged(("cache_salt".into(), "tenant-a".into()))]), None],
    );
    assert_eq!(decoded_tokens_hashes(&salted), vec![SALTED_10_11, PLAIN_12_13]);

    let salted_image = encode_stored(
        None,
        None,
        vec![
            Some(vec![
                WireExtraKey::TaggedWithOffset(("mm".into(), "ab".repeat(32), 0)),
                WireExtraKey::Tagged(("cache_salt".into(), "tenant-a".into())),
            ]),
            None,
        ],
    );
    assert_eq!(decoded_tokens_hashes(&salted_image), vec![SALTED_IMAGE_10_11, PLAIN_12_13]);
}

#[test]
fn test_tagged_lora_key_hashes_through_lora_name_only() {
    let lora_key = || {
        Some(vec![WireExtraKey::TaggedPair((
            "lora".into(),
            "adapter".into(),
            "/adapters/a".into(),
        ))])
    };
    let encoded = encode_stored(None, Some("adapter"), vec![lora_key(), lora_key()]);
    let lora_seed_only = compute_block_hash_for_seq(
        &[10, 11, 12, 13],
        2,
        BlockHashOptions {
            lora_name: Some("adapter"),
            ..Default::default()
        },
    );
    assert_eq!(
        decoded_tokens_hashes(&encoded),
        lora_seed_only.iter().map(|h| h.0).collect::<Vec<_>>()
    );
}

#[derive(serde::Serialize)]
#[serde(untagged)]
enum OddWireKey {
    Embeds((String, Vec<u8>)),
    Quad((String, i64, i64, i64)),
}

#[test]
fn test_unknown_extra_key_shapes_do_not_drop_the_event() {
    let encoded = to_vec(&(
        "BlockStored",
        vec![BlockHashValue::Unsigned(21), BlockHashValue::Unsigned(22)],
        Option::<BlockHashValue>::None,
        vec![10u32, 11, 12, 13],
        2usize,
        Option::<u64>::None,
        Option::<String>::None,
        Option::<String>::None,
        Some(vec![
            Some(vec![OddWireKey::Embeds(("prompt_embeds".into(), vec![1, 2, 3]))]),
            Some(vec![OddWireKey::Quad(("future".into(), 1, 2, 3))]),
        ]),
    ))
    .unwrap();
    assert_eq!(decoded_tokens_hashes(&encoded), vec![PLAIN_10_11, PLAIN_12_13]);
}

fn stored_tokens_hashes(
    parent: Option<u64>,
    lora_name: Option<&str>,
    extra_keys: Vec<Option<Vec<(String, i64)>>>,
    bare_keys: Vec<Option<Vec<String>>>,
) -> Vec<u64> {
    let extra_keys: Vec<Option<Vec<WireExtraKey>>> = extra_keys
        .into_iter()
        .zip(bare_keys)
        .map(|(tuples, bare)| {
            let mut items: Vec<WireExtraKey> = bare
                .unwrap_or_default()
                .into_iter()
                .map(WireExtraKey::Bare)
                .collect();
            items.extend(tuples.unwrap_or_default().into_iter().map(WireExtraKey::WithOffset));
            (!items.is_empty()).then_some(items)
        })
        .collect();
    let encoded = to_vec(&(
        "BlockStored",
        vec![BlockHashValue::Unsigned(21), BlockHashValue::Unsigned(22)],
        parent.map(BlockHashValue::Unsigned),
        vec![10u32, 11, 12, 13],
        2usize,
        Option::<u64>::None,
        Option::<String>::None,
        lora_name.map(str::to_string),
        Some(extra_keys),
    ))
    .unwrap();
    let raw_event: RawKvEvent = from_slice(&encoded).unwrap();
    let warning_count = Arc::new(AtomicU32::new(0));
    let placement = convert_event(raw_event, 7, 2, WorkerWithDpRank::new(3, 0), &warning_count)
        .expect("block size matches")
        .expect("store is kept");
    match placement.event.data {
        KvCacheEventData::Stored(store_data) => {
            store_data.blocks.iter().map(|b| b.tokens_hash.0).collect()
        }
        other => panic!("expected Stored event, got {other:?}"),
    }
}

// Shared with deepapi's probe tests (tests/test_kv_probe_cache_salt.py).
const PLAIN_10_11: u64 = 7369860969341005925;
const SALTED_10_11: u64 = 10774232232830598228;
const PLAIN_12_13: u64 = 2046086924859723728;
const SALTED_IMAGE_10_11: u64 = 1881935480998609139;

#[test]
fn test_cache_salt_extra_hash() {
    assert_eq!(cache_salt_extra_hash("tenant-a"), 7159698576361518977);
    assert_eq!(cache_salt_extra_hash(&"ab".repeat(32)), 0xabababababababab);
}

#[test]
fn test_cache_salt_hashes_into_first_block_only() {
    let salted = stored_tokens_hashes(
        None,
        None,
        vec![None, None],
        vec![Some(vec!["tenant-a".into()]), None],
    );
    assert_eq!(salted, vec![SALTED_10_11, PLAIN_12_13]);

    let plain = stored_tokens_hashes(None, None, vec![None, None], vec![None, None]);
    assert_eq!(plain, vec![PLAIN_10_11, PLAIN_12_13]);
}

#[test]
fn test_cache_salt_ignored_when_event_continues_a_sequence() {
    let hashes = stored_tokens_hashes(
        Some(9),
        None,
        vec![None, None],
        vec![Some(vec!["tenant-a".into()]), None],
    );
    assert_eq!(hashes, vec![PLAIN_10_11, PLAIN_12_13]);
}

#[test]
fn test_lora_name_is_not_a_cache_salt() {
    let lora_only = stored_tokens_hashes(
        None,
        Some("adapter"),
        vec![None, None],
        vec![Some(vec!["adapter".into()]), Some(vec!["adapter".into()])],
    );
    let lora_seed_only = compute_block_hash_for_seq(
        &[10, 11, 12, 13],
        2,
        BlockHashOptions {
            lora_name: Some("adapter"),
            ..Default::default()
        },
    );
    assert_eq!(
        lora_only,
        lora_seed_only.iter().map(|h| h.0).collect::<Vec<_>>()
    );
}

#[test]
fn test_cache_salt_combines_with_image_in_first_block() {
    let hashes = stored_tokens_hashes(
        None,
        None,
        vec![Some(vec![("ab".repeat(32), 0)]), None],
        vec![Some(vec!["tenant-a".into()]), None],
    );
    assert_eq!(hashes, vec![SALTED_IMAGE_10_11, PLAIN_12_13]);
}
