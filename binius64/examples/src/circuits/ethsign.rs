// Copyright 2025 Irreducible Inc.
use std::{array, iter};

use anyhow::Result;
use binius_circuits::{
    bignum::BigUint, ecdsa::ecrecover, fixed_byte_vec::ByteVec, keccak::Keccak256,
};
use binius_core::word::Word;
use binius_frontend::{
    CircuitBuilder, Wire, WitnessFiller,
    util::{byteswap, pack_bytes_into_wires_le},
};
use clap::Args;
use ethsign::SecretKey;
use rand::{Rng, SeedableRng, rngs::StdRng};
use tiny_keccak::{Hasher, Keccak as KeccakHasher};

use crate::ExampleCircuit;

struct Signature {
    r: [Wire; 4],
    s: [Wire; 4],
    recid_odd: Wire,
    address: [Wire; 3],
    msg_keccak: Keccak256,
    address_keccak: Keccak256,
}

/// Example circuit that proves validity of Ethereum-style ECDSA signatures.
pub struct EthSignExample {
    max_msg_len_bytes: usize,
    signatures: Vec<Signature>,
}

#[derive(Args, Debug, Clone)]
pub struct Params {
    /// Number of Ethereum-style signatures to validate
    #[arg(short = 'n', long, default_value_t = 1)]
    pub n_signatures: usize,
    /// Maximum message length
    #[arg(short = 'm', long, default_value_t = 128, value_parser = clap::value_parser!(u16).range(1..))]
    pub max_msg_len_bytes: u16,
}

#[derive(Args, Debug, Clone)]
pub struct Instance {}

impl ExampleCircuit for EthSignExample {
    type Params = Params;
    type Instance = Instance;

    fn build(params: Params, builder: &mut CircuitBuilder) -> Result<Self> {
        let max_msg_len_bytes = params.max_msg_len_bytes as usize;
        let signatures = (0..params.n_signatures)
            .map(|_| {
                let msg_len = builder.add_inout();
                let message = (0..params.max_msg_len_bytes.div_ceil(8))
                    .map(|_| builder.add_inout())
                    .collect::<Vec<_>>();
                let r = array::from_fn(|_| builder.add_inout());
                let s = array::from_fn(|_| builder.add_inout());
                let recid_odd = builder.add_inout();
                let address = array::from_fn(|_| builder.add_inout());

                let msg_final_state = array::from_fn(|_| builder.add_witness());
                let msg_keccak = Keccak256::new(builder, msg_len, msg_final_state, message.clone());

                // The Keccak digest is little endian encoded into 4 words, while Ethereum expects
                // big endian
                let z = BigUint {
                    limbs: msg_final_state[..4]
                        .iter()
                        .rev()
                        .map(|&word| byteswap(builder, word))
                        .collect(),
                };

                // Encoding r & s in little endian
                let public_key = ecrecover(
                    builder,
                    &z,
                    &BigUint { limbs: r.to_vec() },
                    &BigUint { limbs: s.to_vec() },
                    recid_odd,
                );

                // Check that public key is not a point-at-infinity
                builder.assert_false("recovered_pk_not_pai", public_key.is_point_at_infinity);

                // Concatenate x & y in _big_ endian, hash the result to obtain the address
                let mut public_key_limbs = Vec::with_capacity(8);
                public_key_limbs.extend(&public_key.y.limbs);
                public_key_limbs.extend(&public_key.x.limbs);
                public_key_limbs.reverse();

                let public_key_message = public_key_limbs
                    .into_iter()
                    .map(|word| byteswap(builder, word))
                    .collect::<Vec<_>>();

                let address_final_state = array::from_fn(|_| builder.add_witness());
                let address_keccak = Keccak256::new(
                    builder,
                    builder.add_constant_64(64),
                    address_final_state,
                    public_key_message,
                );

                // Assert that the provided address equals digest bytes 12..32
                assert_address_eq(builder, &address_keccak.digest, &address);

                Signature {
                    r,
                    s,
                    recid_odd,
                    address,

                    msg_keccak,
                    address_keccak,
                }
            })
            .collect();

        Ok(Self {
            signatures,
            max_msg_len_bytes,
        })
    }

    fn populate_witness(&self, _instance: Instance, w: &mut WitnessFiller) -> Result<()> {
        // Generate random initial state with fixed seed for reproducibility
        let mut rng = StdRng::seed_from_u64(42);

        for Signature {
            r,
            s,
            recid_odd,
            address,
            msg_keccak,
            address_keccak,
        } in &self.signatures
        {
            // Random private key
            let sk_bytes: [u8; 32] = rng.random();
            let secret_key = SecretKey::from_raw(&sk_bytes)?;

            // Random message
            let msg_len = rng.random_range(1..=self.max_msg_len_bytes);
            let msg_bytes = (0..msg_len).map(|_| rng.random()).collect::<Vec<u8>>();
            let msg_hash = keccak256(&msg_bytes);

            // Sign the message with ECDSA
            let mut signature = secret_key.sign(&msg_hash)?;
            let public = secret_key.public();

            // Hash the message with Keccak
            msg_keccak.populate_len_bytes(w, msg_len);
            msg_keccak.populate_message(w, &msg_bytes);
            msg_keccak.populate_digest(w, msg_hash);

            // ethsign crate returns 0/1 recid, convert to `recid_odd` boolean
            w[*recid_odd] = if signature.v != 0 {
                Word::ALL_ONE
            } else {
                Word::ZERO
            };

            // ethsign crate returns r & s big endian, byteswap
            signature.r.reverse();
            pack_bytes_into_wires_le(w, r, &signature.r);

            signature.s.reverse();
            pack_bytes_into_wires_le(w, s, &signature.s);

            // Hash the (big endian) public key
            let pk_bytes = public.bytes();
            let pk_hash = keccak256(pk_bytes);

            address_keccak.populate_len_bytes(w, 64);
            address_keccak.populate_message(w, pk_bytes);
            address_keccak.populate_digest(w, pk_hash);

            pack_bytes_into_wires_le(w, address, public.address());
        }

        Ok(())
    }

    fn param_summary(params: &Self::Params) -> Option<String> {
        Some(format!(
            "{}s-{}b",
            params.n_signatures, params.max_msg_len_bytes
        ))
    }
}

fn assert_address_eq(b: &CircuitBuilder, digest: &[Wire], address: &[Wire]) {
    assert_eq!(digest.len(), 4);
    assert_eq!(address.len(), 3);

    let digest_len = b.add_constant_64(32);
    let digest_byte_vec = ByteVec::new(digest.to_vec(), digest_len);
    let digest_sliced = digest_byte_vec.slice_const_range(b, 12..32);

    for (i, (&lhs_i, rhs_i)) in iter::zip(address, digest_sliced.data).enumerate() {
        b.assert_eq(format!("address_word_{i}"), lhs_i, rhs_i);
    }
}

fn keccak256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = KeccakHasher::v256();
    hasher.update(bytes);

    let mut digest = [0u8; 32];
    hasher.finalize(&mut digest);

    digest
}
