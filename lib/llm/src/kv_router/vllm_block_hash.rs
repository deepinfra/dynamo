// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM's block hashes, computed from token ids the way a worker computes them,
//! so the router can name blocks as the workers' KV events and stores do.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The workers' hash scheme, for routers whose workers announce none: `{algo}:{NONE_HASH hex}[:{hash block size}]`,
/// e.g. `sha256:1973e2…` for vLLM's default algo under `PYTHONHASHSEED=0`.
pub const BLOCK_HASH_ENV: &str = "DYN_MOONCAKE_STORE_BLOCK_HASH";

/// How a worker's vLLM names blocks: `--prefix-caching-hash-algo` and the hash
/// its first block chains from.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockHashScheme {
    pub algo: String,
    /// vLLM's `NONE_HASH`, in hex.
    pub none_hash: String,
    /// Tokens per block hash (`prefix_match_unit`) when finer than the router's block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash_block_size: Option<u32>,
}

impl BlockHashScheme {
    pub(crate) fn from_env(value: &str) -> Option<Self> {
        let mut parts = value.split(':');
        let algo = parts.next()?.to_string();
        let none_hash = parts.next()?.to_string();
        let hash_block_size = match parts.next() {
            Some(size) => Some(size.parse().ok()?),
            None => None,
        };
        parts.next().is_none().then_some(Self {
            algo,
            none_hash,
            hash_block_size,
        })
    }

    /// Only vLLM's default "sha256" is computed; other algos keep learned links.
    pub(crate) fn hasher(&self) -> Option<VllmBlockHasher> {
        if self.algo != "sha256" || self.none_hash.len() != 64 {
            return None;
        }
        let mut none_hash = [0_u8; 32];
        for (byte, pair) in none_hash
            .iter_mut()
            .zip(self.none_hash.as_bytes().chunks(2))
        {
            *byte = u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?;
        }
        Some(VllmBlockHasher {
            none_hash,
            hash_block_size: self.hash_block_size.map(|size| size as usize),
        })
    }
}

/// vLLM's block hashes for the "sha256" algo (`hash_block_tokens`): SHA-256 of
/// `pickle.dumps((parent hash, tuple(token ids), None), protocol=5)`, chained
/// from `NONE_HASH`. Prompts with extra keys (LoRA, multimodal, cache salt)
/// hash differently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VllmBlockHasher {
    none_hash: [u8; 32],
    hash_block_size: Option<usize>,
}

impl VllmBlockHasher {
    /// The chunk hash (low 64 bits, as vLLM's int block hashes) of each of the
    /// first `blocks` router blocks of `tokens`.
    pub(crate) fn chunk_hashes(
        &self,
        tokens: &[u32],
        block_size: usize,
        blocks: usize,
    ) -> Option<Vec<u64>> {
        let hash_block_size = self.hash_block_size.unwrap_or(block_size);
        if hash_block_size == 0 || !block_size.is_multiple_of(hash_block_size) {
            return None;
        }
        let mut parent = self.none_hash;
        let mut pickled = Vec::with_capacity(48 + 5 * hash_block_size);
        let mut hashes = Vec::with_capacity(blocks);
        for block in tokens.chunks_exact(block_size).take(blocks) {
            for part in block.chunks_exact(hash_block_size) {
                if !pickle_block(&parent, part, &mut pickled) {
                    return (!hashes.is_empty()).then_some(hashes);
                }
                parent = Sha256::digest(&pickled).into();
            }
            hashes.push(u64::from_be_bytes(parent[24..].try_into().unwrap()));
        }
        Some(hashes)
    }
}

/// `pickle.dumps((parent, tuple(tokens), None), protocol=5)` byte for byte as
/// CPython writes it. False for a token pickle would not write as a 4-byte int.
fn pickle_block(parent: &[u8; 32], tokens: &[u32], out: &mut Vec<u8>) -> bool {
    const MEMOIZE: u8 = 0x94;
    out.clear();
    // PROTO 5, then one FRAME whose length is filled in last.
    out.extend_from_slice(&[0x80, 5, 0x95, 0, 0, 0, 0, 0, 0, 0, 0]);
    // SHORT_BINBYTES
    out.extend_from_slice(&[b'C', 32]);
    out.extend_from_slice(parent);
    out.push(MEMOIZE);
    if tokens.is_empty() {
        // EMPTY_TUPLE, which is not memoized.
        out.push(b')');
    } else {
        if tokens.len() > 3 {
            out.push(b'(');
        }
        for &token in tokens {
            match token {
                0..=0xff => out.extend_from_slice(&[b'K', token as u8]),
                0x100..=0xffff => {
                    out.push(b'M');
                    out.extend_from_slice(&(token as u16).to_le_bytes());
                }
                0x1_0000..=0x7fff_ffff => {
                    out.push(b'J');
                    out.extend_from_slice(&token.to_le_bytes());
                }
                _ => return false,
            }
        }
        // TUPLE, or TUPLE1-3 for short tuples.
        out.push(match tokens.len() {
            1 => 0x85,
            2 => 0x86,
            3 => 0x87,
            _ => b't',
        });
        out.push(MEMOIZE);
    }
    // NONE, TUPLE3, STOP.
    out.extend_from_slice(&[b'N', 0x87, MEMOIZE, b'.']);
    let frame = (out.len() - 11) as u64;
    out[3..11].copy_from_slice(&frame.to_le_bytes());
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // CPython 3.12: `NONE_HASH` under PYTHONHASHSEED=0, i.e. sha256(pickle.dumps("0", protocol=5)).
    const NONE_HASH: &str = "1973e23848344dc43a988a9b478663803cfffe1243480253f9a3cf004b14aa7c";

    fn none_hash() -> [u8; 32] {
        BlockHashScheme::from_env(&format!("sha256:{NONE_HASH}"))
            .unwrap()
            .hasher()
            .unwrap()
            .none_hash
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// Token ids covering every int opcode width, as CPython's tests used them.
    fn vector_tokens() -> Vec<u32> {
        vec![
            1, 1, 2147483647, 129279, 256, 2147483647, 129279, 0, 2147483647, 256, 1, 65536, 0,
            3335, 129279, 256, 129279, 0, 256, 2147483647, 2147483647, 256, 256, 256, 2147483647,
            65535, 0, 54549, 1, 65535, 65536, 117384, 129279, 256, 65535, 2147483647, 110912, 0,
            256, 129279, 255, 48119, 65536, 2147483647, 87000, 255, 129279, 2147483647, 0, 0,
            129279, 255, 22097, 0, 256, 70728, 129279, 65536, 65536, 65535, 86404, 0, 50290, 255,
            67984, 129279, 0, 65536, 74710, 26193, 2147483647, 65536, 65536, 207, 65536, 0, 256,
            255, 255, 1, 104652, 65535, 1, 0, 0, 65535, 65535, 255, 65535, 255, 65535, 255, 65535,
            65535, 65536, 2147483647, 0, 129279, 129279, 256, 1, 33221, 256, 129279, 0, 0, 255,
            255, 58414, 129279, 256, 128054, 256, 0, 65536, 129279, 65535, 256, 0, 1, 1, 65535,
            255, 65535, 0, 0, 256, 2147483647, 22481, 129279, 65536, 256, 129279, 256, 1, 129279,
            38806, 0, 129279, 65535, 255, 65536, 255, 129279, 65535, 1, 129279, 122183, 45069,
            63504, 1, 0, 255, 255, 119404, 65535, 65536, 78670, 65535, 65536, 1, 256, 2147483647,
            17740, 1, 0, 1, 255, 255, 1, 129279, 10046, 1, 65535, 65535, 73983, 1, 65535, 0, 65535,
            0, 129279, 0, 256, 129279, 1, 255, 256, 1, 129279, 105707, 65535,
        ]
    }

    #[test]
    fn blocks_pickle_as_cpython_does() {
        let mut out = Vec::new();
        // Expected bytes from `pickle.dumps((NONE_HASH, tuple(tokens), None), protocol=5)`.
        let parent = "43201973e23848344dc43a988a9b478663803cfffe1243480253f9a3cf004b14aa7c94";
        for (tokens, expected) in [
            (&[][..], format!("8005952800000000000000{parent}294e87942e")),
            (
                &[1][..],
                format!("8005952b00000000000000{parent}4b0185944e87942e"),
            ),
            (
                &[1, 1][..],
                format!("8005952d00000000000000{parent}4b014b0186944e87942e"),
            ),
            (
                &[1, 1, 2147483647][..],
                format!("8005953200000000000000{parent}4b014b014affffff7f87944e87942e"),
            ),
        ] {
            assert!(pickle_block(&none_hash(), tokens, &mut out));
            assert_eq!(hex(&out), expected);
        }
        // Each int width: K (1 byte), M (2), J (4).
        assert!(pickle_block(
            &Sha256::digest(b"x").into(),
            &[
                [0, 1, 255, 256, 65535, 65536, 2147483647].as_slice(),
                &[7; 57]
            ]
            .concat(),
            &mut out
        ));
        assert_eq!(
            hex(&out),
            "800595b20000000000000043202d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a488194284b004b014bff4d00014dffff4a000001004affffff7f"
                .to_string()
                + &"4b07".repeat(57)
                + "74944e87942e"
        );
        // A token CPython would write as a long.
        assert!(!pickle_block(&none_hash(), &[1 << 31], &mut out));
    }

    #[test]
    fn computed_chunk_hashes_match_vllm() {
        let tokens = vector_tokens();
        let scheme = |hash_block_size: Option<u32>| BlockHashScheme {
            algo: "sha256".to_string(),
            none_hash: NONE_HASH.to_string(),
            hash_block_size,
        };
        // vLLM's `hash_block_tokens` chain, low 64 bits, from CPython.
        assert_eq!(
            scheme(None).hasher().unwrap().chunk_hashes(&tokens, 64, 3),
            Some(vec![
                0xed1a61cabd2a4392,
                0xf351ca48d7e0860f,
                0x3f98e2f0c81b2e6c
            ])
        );
        // Hashes every 32 tokens: a chunk is named by its last one.
        assert_eq!(
            scheme(Some(32))
                .hasher()
                .unwrap()
                .chunk_hashes(&tokens, 64, 3),
            Some(vec![
                0x6d12db8d48a39220,
                0xfce9ac6a40838904,
                0x5d458272d9d0fe84
            ])
        );
        // A partial last block and a shorter block list.
        assert_eq!(
            scheme(None)
                .hasher()
                .unwrap()
                .chunk_hashes(&tokens[..150], 64, 3),
            Some(vec![0xed1a61cabd2a4392, 0xf351ca48d7e0860f])
        );
        assert_eq!(
            scheme(Some(48))
                .hasher()
                .unwrap()
                .chunk_hashes(&tokens, 64, 3),
            None
        );
        assert_eq!(
            BlockHashScheme {
                algo: "xxhash".to_string(),
                ..scheme(None)
            }
            .hasher(),
            None
        );
    }

    #[test]
    fn schemes_parse() {
        assert_eq!(
            BlockHashScheme::from_env(&format!("sha256:{NONE_HASH}:32")),
            Some(BlockHashScheme {
                algo: "sha256".to_string(),
                none_hash: NONE_HASH.to_string(),
                hash_block_size: Some(32),
            })
        );
        assert_eq!(BlockHashScheme::from_env("sha256"), None);
        assert_eq!(BlockHashScheme::from_env("sha256:ab:x"), None);
        assert_eq!(BlockHashScheme::from_env("sha256:ab:32:1"), None);
        assert_eq!(
            BlockHashScheme::from_env("sha256:ab").unwrap().hasher(),
            None
        );
    }
}
