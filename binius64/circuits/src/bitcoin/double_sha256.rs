// Copyright 2025 Irreducible Inc.
//! The Bitcoin double-SHA256 hash function.

use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use sha2::Digest;

use crate::sha256::Sha256;

/// Stores some intermediate wires of a double-SHA256 computation, so that they can later be
/// populated with [`Self::populate_inner`].
///
/// (Hopefully this struct is not needed in the future, and only the [`Self::construct_circuit`]
/// method is needed without any return value.)
pub struct DoubleSha256 {
    sha256_0: Sha256,
    sha256_1: Sha256,
}

impl DoubleSha256 {
    /// Constructs a circuit that asserts that `digest = SHA256(SHA256(message))`.
    /// The message length in bytes is fixed at compile time to be `message.len() * 8`.
    ///
    /// # Preconditions
    ///
    /// - `message.len() * 8 == message_len`
    pub fn construct_circuit(
        builder: &CircuitBuilder,
        message: Vec<Wire>,
        digest: [Wire; 4],
    ) -> Self {
        // first SHA256 circuit
        let message_0: Vec<Wire> = std::iter::repeat_with(|| builder.add_witness())
            .take(message.len())
            .collect();
        let digest_0: [Wire; 4] = std::array::from_fn(|_| builder.add_witness());
        let sha256_0 = Sha256::new(
            builder,
            builder.add_constant_64(message.len() as u64 * 8),
            digest_0,
            message_0,
        );

        // second SHA256 circuit
        let message_1: Vec<Wire> = std::iter::repeat_with(|| builder.add_witness())
            .take(4)
            .collect();
        let digest_1: [Wire; 4] = std::array::from_fn(|_| builder.add_witness());
        let sha256_1 = Sha256::new(builder, builder.add_constant_64(32), digest_1, message_1);

        // input of first SHA256 circuit is `message`
        let message_0_le = sha256_0.message_to_le_wires(builder);
        for i in 0..message.len() {
            builder.assert_eq("agreement input", message[i], message_0_le[i]);
        }

        // output of first SHA256 circuit is input of second SHA256 circuit
        let digest_0_le = sha256_0.digest_to_le_wires(builder);
        let message_1_le = sha256_1.message_to_le_wires(builder);
        for i in 0..4 {
            builder.assert_eq("agreement intermediate", digest_0_le[i], message_1_le[i]);
        }

        // output of second SHA256 circuit is `digest`
        let digest_1_le = sha256_1.digest_to_le_wires(builder);
        for i in 0..4 {
            builder.assert_eq("agreement output", digest_1_le[i], digest[i]);
        }

        Self { sha256_0, sha256_1 }
    }

    /// You need to call this with the `message` bytes that you will put in the message wires from
    /// the [`Self::construct_circuit`].
    ///
    /// **Note:** This does NOT populate the message wires or digest wires that you passed to
    /// [`Self::construct_circuit`]. You are responsible yourself for populating those wires.
    /// This method only fills some internal wires needed for the SHA256 computations.
    ///
    /// Return the digest.
    ///
    /// (Hopefully this method is not needed in the future.)
    pub fn populate_inner(&self, filler: &mut WitnessFiller, message: &[u8]) -> [u8; 32] {
        // don't need to `populate_len(...)` because we passed a constant to the length wires
        self.sha256_0.populate_message(filler, message);
        let digest_0: [u8; 32] = sha2::Sha256::digest(message).into();
        self.sha256_0.populate_digest(filler, digest_0);

        // don't need to `populate_len(...)` because we passed a constant to the length wires
        self.sha256_1.populate_message(filler, &digest_0);
        let digest_1: [u8; 32] = sha2::Sha256::digest(digest_0).into();
        self.sha256_1.populate_digest(filler, digest_1);

        digest_1
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use binius_core::verify::verify_constraints;
    use binius_frontend::util::pack_bytes_into_wires_le;
    use hex_literal::hex;

    use super::*;

    #[test]
    fn test_valid() {
        // construct circuit
        let builder = CircuitBuilder::new();
        let block_header: [Wire; 10] = array::from_fn(|_| builder.add_witness());
        let block_hash: [Wire; 4] = array::from_fn(|_| builder.add_witness());
        let double_sha_256 =
            DoubleSha256::construct_circuit(&builder, block_header.to_vec(), block_hash);
        let circuit = builder.build();

        // populate_witness
        let mut filler = circuit.new_witness_filler();
        let block_header_value = hex!(
            "000000264a14e21adad047d981c06a26446e345eda3d8beb807401000000000000000000fc01df2139954b36cebc3fa6fbf6a7160a67d34b67e5c4aa2a7ce46f5bb42a83642ea468b32c0217d14ba4d1"
        );
        let block_hash_value =
            hex!("228561b085b7524957e515605725901238299ff2793300000000000000000000");
        pack_bytes_into_wires_le(&mut filler, &block_header, &block_header_value);
        pack_bytes_into_wires_le(&mut filler, &block_hash, &block_hash_value);
        double_sha_256.populate_inner(&mut filler, &block_header_value);
        circuit.populate_wire_witness(&mut filler).unwrap();

        // check
        let constraint_system = circuit.constraint_system();
        verify_constraints(constraint_system, &filler.into_value_vec()).unwrap();
    }

    #[test]
    fn test_invalid() {
        // construct circuit
        let builder = CircuitBuilder::new();
        let block_header: [Wire; 10] = array::from_fn(|_| builder.add_witness());
        let block_hash: [Wire; 4] = array::from_fn(|_| builder.add_witness());
        let double_sha_256 =
            DoubleSha256::construct_circuit(&builder, block_header.to_vec(), block_hash);
        let circuit = builder.build();

        // populate_witness
        let mut filler = circuit.new_witness_filler();
        let block_header_value = hex!(
            "000000264a14e21adad047d981c06a26446e345eda3d8beb807401000000000000000000fc01df2139954b36cebc3fa6fbf6a7160a67d34b67e5c4aa2a7ce46f5bb42a83642ea468b32c0217d14ba4d1"
        );
        let block_hash_value =
            hex!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        pack_bytes_into_wires_le(&mut filler, &block_header, &block_header_value);
        pack_bytes_into_wires_le(&mut filler, &block_hash, &block_hash_value);
        double_sha_256.populate_inner(&mut filler, &block_header_value);
        // should fail because the hash is wrong
        circuit.populate_wire_witness(&mut filler).unwrap_err();
    }
}
