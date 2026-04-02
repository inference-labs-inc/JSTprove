// Copyright 2025 Irreducible Inc.

//! Verifying a Bitcoin header chain.

use binius_core::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};

use crate::{
    bignum::{self, BigUint},
    bitcoin::double_sha256::DoubleSha256,
};

/// Stores some intermediate wires of the circuit, so that they can later be populated with
/// [`Self::populate_inner`].
pub struct HeaderChain {
    header_digests: Vec<DoubleSha256>,
}

impl HeaderChain {
    /// Constructs a circuit that asserts the following things:
    ///
    /// - `latest_digest == hash(headers[0])` (head)
    /// - `previous_block_hash(headers[i]) = hash(headers[i+1])` (hash chain)
    /// - `hash(headers[i]) < target(headers[i])` (proof of work)
    ///
    /// **IMPORTANT**: This does currently NOT assert that `target(headers[i])` is in any way
    /// related to `target(headers[i+1])`, like it must be in the Bitcoin protocol. In particular,
    /// this means that one can easily satisfy this circuit with a self-mined sequence of blocks
    /// which have low difficulty.
    ///
    /// **Note**: It also doesn't check many other things about the block header, like that the
    /// constraints on the timestamps.
    pub fn construct_circuit(
        builder: &CircuitBuilder,
        // latest block comes first
        headers: &[[Wire; 10]],
        latest_digest: [Wire; 4],
    ) -> Self {
        let mut header_digests = Vec::new();

        // latest header
        {
            // hash equals `latest_digest`
            header_digests.push(DoubleSha256::construct_circuit(
                builder,
                headers[0].to_vec(),
                latest_digest,
            ));

            // hash is smaller than target (proof of work)
            let target = target(builder, headers[0][9]);
            let digest_as_uint = BigUint {
                limbs: latest_digest.to_vec(),
            };
            let fulfills_target = bignum::biguint_lt(builder, &digest_as_uint, &target);
            builder.assert_true("PoW fulfills target", fulfills_target);
        }

        // header chain
        for i in 1..headers.len() {
            // hash equals the "previous block hash" in the newer block
            let digest = previous_block_hash(builder, &headers[i - 1]);
            header_digests.push(DoubleSha256::construct_circuit(
                builder,
                headers[i].to_vec(),
                digest,
            ));

            // hash is smaller than target (proof of work)
            // NOTE: One could save constraints by noting that the target only changes every 2016
            // blocks.
            let target = target(builder, headers[i][9]);
            let digest_as_uint = BigUint {
                limbs: digest.to_vec(),
            };
            let fulfills_target = bignum::biguint_lt(builder, &digest_as_uint, &target);
            builder.assert_true("PoW fulfills target", fulfills_target);

            // bits of newer block is correctly computed from bits of older block
            // FIXME TODO
        }

        Self { header_digests }
    }

    pub fn populate_inner(&self, filler: &mut WitnessFiller, headers: &[&[u8]]) {
        for (header_digest, header) in self.header_digests.iter().zip(headers) {
            header_digest.populate_inner(filler, header);
        }
    }
}

/// Extracts the previous block hash from a block header.
fn previous_block_hash(builder: &CircuitBuilder, header: &[Wire; 10]) -> [Wire; 4] {
    [
        join(builder, header[1], header[0]),
        join(builder, header[2], header[1]),
        join(builder, header[3], header[2]),
        join(builder, header[4], header[3]),
    ]
}

fn join(builder: &CircuitBuilder, b0: Wire, b1: Wire) -> Wire {
    let c0 = builder.shl(b0, 32);
    let c1 = builder.shr(b1, 32);
    builder.bxor(c0, c1)
}

/// Computes the 32-byte target from the 4-byte compact bits field.
fn target(builder: &CircuitBuilder, bits: Wire) -> BigUint {
    let mantissa = builder.band(builder.add_constant_64(0x0000000000ffffff), bits);
    let exponent = builder.band(
        builder.add_constant_64(0x00000000000000ff),
        builder.shr(bits, 24),
    );

    // compute how many bytes we need to shift the mantissa to the left
    // NOTE: Check underflow? And check that exponent is not too big?
    let (shift_val, _) = builder.isub_bin_bout(
        exponent,
        builder.add_constant_64(3),
        builder.add_constant(Word::ZERO),
    );

    // optionally shift by 1 byte
    let cond_1 = builder.shl(shift_val, 63);
    let v0_1 = builder.select(cond_1, builder.shl(mantissa, 8), mantissa);

    // optionally shift by 2 bytes
    let cond_2 = builder.shl(shift_val, 62);
    let v0_2 = builder.select(cond_2, builder.shl(v0_1, 2 * 8), v0_1);

    // optionally shift by 4 bytes
    let cond_4 = builder.shl(shift_val, 61);
    let v0_4 = builder.select(cond_4, builder.shl(v0_2, 4 * 8), v0_2);
    let v1_4 = builder.select(
        cond_4,
        builder.shr(v0_2, 4 * 8),
        builder.add_constant(Word::ZERO),
    );

    // optionally shift by 8 bytes
    let cond_8 = builder.shl(shift_val, 60);
    let v0_8 = builder.select(cond_8, builder.add_constant(Word::ZERO), v0_4);
    let v1_8 = builder.select(cond_8, v0_4, v1_4);
    let v2_8 = builder.select(cond_8, v1_4, builder.add_constant(Word::ZERO));

    // optionally shift by 16 bytes
    let cond_16 = builder.shl(shift_val, 59);
    let v0_16 = builder.select(cond_16, builder.add_constant(Word::ZERO), v0_8);
    let v1_16 = builder.select(cond_16, builder.add_constant(Word::ZERO), v1_8);
    let v2_16 = builder.select(cond_16, v0_8, v2_8);
    let v3_16 = builder.select(cond_16, v1_8, builder.add_constant(Word::ZERO));

    BigUint {
        limbs: vec![v0_16, v1_16, v2_16, v3_16],
    }
}

#[cfg(test)]
mod tests {
    use binius_core::verify::verify_constraints;
    use binius_frontend::util::pack_bytes_into_wires_le;
    use hex_literal::hex;

    use super::*;
    use crate::bignum;

    /// Tests that the `target` circuit is correct.
    fn test_target_fixed_exponent(exponent: usize) {
        // build circuit
        let builder = CircuitBuilder::new();
        let bits = builder.add_witness();
        let asserted_target = BigUint::new_witness(&builder, 4);
        let target = target(&builder, bits);
        bignum::assert_eq(
            &builder,
            "target matches asserted target",
            &target,
            &asserted_target,
        );
        let circuit = builder.build();

        // populate witness
        let mut filler = circuit.new_witness_filler();
        let bits_value: u64 = 0x4444444400aabbcc | ((exponent as u64) << 24);
        filler[bits] = Word(bits_value);
        let shift_val = exponent - 3;
        let asserted_target_value = num_bigint::BigUint::from_bytes_be(&hex!(
            "0000000000000000000000000000000000000000000000000000000000aabbcc"
        )) << (shift_val * 8);
        let mut asserted_target_digits = asserted_target_value.to_u64_digits();
        asserted_target_digits.truncate(4);
        while asserted_target_digits.len() < 4 {
            asserted_target_digits.push(0);
        }
        asserted_target.populate_limbs(&mut filler, &asserted_target_digits);
        circuit.populate_wire_witness(&mut filler).unwrap();

        // check
        let constraint_system = circuit.constraint_system();
        verify_constraints(constraint_system, &filler.into_value_vec()).unwrap();
    }

    #[test]
    fn test_target() {
        // `exponent - 3` runs in 0..32
        for exponent in 3..35 {
            test_target_fixed_exponent(exponent);
        }
    }

    #[test]
    fn test_valid() {
        // build circuit
        let builder = CircuitBuilder::new();
        let headers: Vec<[Wire; 10]> =
            std::iter::repeat_with(|| std::array::from_fn(|_| builder.add_witness()))
                .take(3)
                .collect();
        let latest_digest: [Wire; 4] = std::array::from_fn(|_| builder.add_witness());
        let header_chain = HeaderChain::construct_circuit(&builder, &headers, latest_digest);
        let circuit = builder.build();

        // populate witness
        let mut filler = circuit.new_witness_filler();
        let headers_value = vec![
            hex!(
                "000000264a14e21adad047d981c06a26446e345eda3d8beb807401000000000000000000fc01df2139954b36cebc3fa6fbf6a7160a67d34b67e5c4aa2a7ce46f5bb42a83642ea468b32c0217d14ba4d1"
            ),
            hex!(
                "00606a3190af271dec0197c6b87f218070c5611cf0c506f6671c0200000000000000000021f21732014796558e6736b15de9b132ca65318b42fe5759b153c63dd4306e38842da468b32c021737421730"
            ),
            hex!(
                "00800020e77cf8eb3114116cc2e6d4aca9b27d35bb5402a5054a010000000000000000005b41d83c40c8226c48807401b0b08aa8e4e2eb053bf91d580f62cd49f5c2a99f802ba468b32c0217cea24c34"
            ),
        ];
        let latest_digest_value =
            hex!("228561b085b7524957e515605725901238299ff2793300000000000000000000");
        for (header, header_value) in headers.iter().zip(&headers_value) {
            pack_bytes_into_wires_le(&mut filler, header, header_value);
        }
        pack_bytes_into_wires_le(&mut filler, &latest_digest, &latest_digest_value);
        let headers_value_ref: Vec<&[u8]> = headers_value.iter().map(AsRef::as_ref).collect();
        header_chain.populate_inner(&mut filler, &headers_value_ref);
        circuit.populate_wire_witness(&mut filler).unwrap();

        // check
        let constraint_system = circuit.constraint_system();
        verify_constraints(constraint_system, &filler.into_value_vec()).unwrap();
    }
}
