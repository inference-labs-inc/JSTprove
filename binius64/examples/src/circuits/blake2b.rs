// Copyright 2025 Irreducible Inc.

use anyhow::Result;
use binius_circuits::blake2b::{Blake2bCircuit, blake2b};
use binius_frontend::{CircuitBuilder, WitnessFiller};
use clap::Args;

use super::utils;
use crate::ExampleCircuit;

/// Blake2b circuit example demonstrating the Blake2b hash function implementation
pub struct Blake2bExample {
    blake2b_circuit: Blake2bCircuit,
    max_msg_len_bytes: usize,
}

/// Circuit parameters that affect structure (compile-time configuration)
#[derive(Args, Debug, Clone)]
pub struct Params {
    /// Maximum message length in bytes that the circuit can handle.
    #[arg(long)]
    pub max_msg_len_bytes: Option<usize>,
}

/// Instance data for witness population (runtime values)
#[derive(Args, Debug, Clone)]
#[group(multiple = false)]
pub struct Instance {
    /// Length of the randomly generated message, in bytes (defaults to 1024).
    #[arg(long)]
    pub message_len: Option<usize>,

    /// UTF-8 string to hash (if not provided, random bytes are generated)
    #[arg(long)]
    pub message_string: Option<String>,
}

impl ExampleCircuit for Blake2bExample {
    type Params = Params;
    type Instance = Instance;

    fn build(params: Params, builder: &mut CircuitBuilder) -> Result<Self> {
        let max_msg_len_bytes =
            utils::determine_hash_max_bytes_from_args(params.max_msg_len_bytes)?;

        let blake2b_circuit = Blake2bCircuit::new_with_length(builder, max_msg_len_bytes);

        Ok(Self {
            blake2b_circuit,
            max_msg_len_bytes,
        })
    }

    fn populate_witness(&self, instance: Instance, w: &mut WitnessFiller) -> Result<()> {
        // Step 1: Get raw message bytes
        let raw_message =
            utils::generate_message_bytes(instance.message_string, instance.message_len);

        // Step 2: Zero-pad to maximum length
        let padded_message = utils::zero_pad_message(raw_message, self.max_msg_len_bytes)?;

        // Step 3: Compute digest using reference implementation
        let expected_digest_vec = blake2b(&padded_message, 64);
        let mut expected_digest = [0u8; 64];
        expected_digest.copy_from_slice(&expected_digest_vec);

        // Step 4: Populate witness values (Blake2b doesn't use len_bytes)
        self.blake2b_circuit.populate_message(w, &padded_message);
        self.blake2b_circuit.populate_digest(w, &expected_digest);

        Ok(())
    }

    fn param_summary(params: &Self::Params) -> Option<String> {
        Some(format!(
            "{}b",
            params
                .max_msg_len_bytes
                .unwrap_or(utils::DEFAULT_HASH_MESSAGE_BYTES)
        ))
    }
}
