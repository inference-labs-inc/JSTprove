// Copyright 2025 Irreducible Inc.
use anyhow::Result;
use binius_circuits::keccak::{Keccak256, N_WORDS_PER_DIGEST};
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use clap::Args;
use sha3::Digest;

use super::utils;
use crate::ExampleCircuit;

/// Keccak-256 hash circuit example
pub struct KeccakExample {
    keccak_hash: Keccak256,
    max_len_bytes: usize,
}

#[derive(Args, Debug, Clone)]
pub struct Params {
    /// Maximum message length in bytes that the circuit can handle
    #[arg(long)]
    pub max_len_bytes: Option<usize>,
}

#[derive(Args, Debug, Clone)]
#[group(multiple = false)]
pub struct Instance {
    /// Length of the randomly generated message, in bytes (defaults to 1024)
    #[arg(long)]
    pub message_len: Option<usize>,

    /// UTF-8 string to hash (if not provided, random bytes are generated)
    #[arg(long)]
    pub message_string: Option<String>,
}

impl ExampleCircuit for KeccakExample {
    type Params = Params;
    type Instance = Instance;

    fn build(params: Params, builder: &mut CircuitBuilder) -> Result<Self> {
        let max_len_bytes = utils::determine_hash_max_bytes_from_args(params.max_len_bytes)?;

        let len_bytes = builder.add_witness();
        let digest: [Wire; N_WORDS_PER_DIGEST] = std::array::from_fn(|_| builder.add_inout());

        let n_words = max_len_bytes.div_ceil(8);
        let message = (0..n_words).map(|_| builder.add_inout()).collect();

        let keccak = Keccak256::new(builder, len_bytes, digest, message);

        Ok(Self {
            keccak_hash: keccak,
            max_len_bytes,
        })
    }

    fn populate_witness(&self, instance: Instance, w: &mut WitnessFiller) -> Result<()> {
        // Step 1: Get raw message bytes
        let raw_message =
            utils::generate_message_bytes(instance.message_string, instance.message_len);

        // Step 2: Zero-pad to maximum length
        let padded_message = utils::zero_pad_message(raw_message, self.max_len_bytes)?;

        // Step 3: Compute digest using reference implementation
        let mut hasher = sha3::Keccak256::new();
        hasher.update(&padded_message);
        let digest: [u8; 32] = hasher.finalize().into();

        // Step 4: Populate witness values
        self.keccak_hash.populate_len_bytes(w, padded_message.len());
        self.keccak_hash.populate_message(w, &padded_message);
        self.keccak_hash.populate_digest(w, digest);

        Ok(())
    }

    fn param_summary(params: &Self::Params) -> Option<String> {
        Some(format!(
            "{}b",
            params
                .max_len_bytes
                .unwrap_or(utils::DEFAULT_HASH_MESSAGE_BYTES)
        ))
    }
}
