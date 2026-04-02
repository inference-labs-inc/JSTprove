// Copyright 2025 Irreducible Inc.
use std::array;

use anyhow::Result;
use binius_circuits::sha256::Sha256;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use clap::Args;
use sha2::Digest;

use super::utils;
use crate::ExampleCircuit;

pub struct Sha256Example {
    sha256_gadget: Sha256,
}

#[derive(Args, Debug, Clone)]
pub struct Params {
    /// Maximum message length in bytes that the circuit can handle.
    #[arg(long)]
    pub max_len_bytes: Option<usize>,

    /// Build circuit for exact message length (makes length a compile-time constant instead of
    /// runtime witness).
    #[arg(long, default_value_t = false)]
    pub exact_len: bool,
}

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

impl ExampleCircuit for Sha256Example {
    type Params = Params;
    type Instance = Instance;

    fn build(params: Params, builder: &mut CircuitBuilder) -> Result<Self> {
        let max_len_bytes = utils::determine_hash_max_bytes_from_args(params.max_len_bytes)?;
        let max_len = max_len_bytes.div_ceil(8);
        let len_bytes = if params.exact_len {
            builder.add_constant_64(max_len_bytes as u64)
        } else {
            builder.add_witness()
        };
        let sha256_gadget = mk_circuit(builder, max_len, len_bytes);

        Ok(Self { sha256_gadget })
    }

    fn populate_witness(&self, instance: Instance, w: &mut WitnessFiller) -> Result<()> {
        // Step 1: Get raw message bytes
        let raw_message =
            utils::generate_message_bytes(instance.message_string, instance.message_len);

        // Step 2: Zero-pad to maximum length
        let padded_message =
            utils::zero_pad_message(raw_message, self.sha256_gadget.max_len_bytes())?;

        // Step 3: Compute digest using reference implementation
        let digest = sha2::Sha256::digest(&padded_message);

        // Step 4: Populate witness values
        self.sha256_gadget
            .populate_len_bytes(w, padded_message.len());
        self.sha256_gadget.populate_message(w, &padded_message);
        self.sha256_gadget.populate_digest(w, digest.into());

        Ok(())
    }

    fn param_summary(params: &Self::Params) -> Option<String> {
        let base = format!(
            "{}b",
            params
                .max_len_bytes
                .unwrap_or(utils::DEFAULT_HASH_MESSAGE_BYTES)
        );
        if params.exact_len {
            Some(format!("{}-exact", base))
        } else {
            Some(base)
        }
    }
}

fn mk_circuit(b: &mut CircuitBuilder, max_len: usize, len_bytes: Wire) -> Sha256 {
    let digest: [Wire; 4] = array::from_fn(|_| b.add_inout());
    let message = (0..max_len).map(|_| b.add_inout()).collect();
    Sha256::new(b, len_bytes, digest, message)
}
