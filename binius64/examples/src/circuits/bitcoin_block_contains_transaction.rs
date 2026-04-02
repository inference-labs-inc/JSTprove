// Copyright 2025 Irreducible Inc.
//! Proof that a Bitcoin block contains a certain transaction.

use binius_circuits::bitcoin::{
    double_sha256::DoubleSha256,
    merkle_path::{MerklePath, SiblingSide},
};
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};

/// Stores some intermediate wires of the circuit, so that they can later be populated with
/// [`Self::populate_inner`].
pub struct BlockContainsTransaction {
    merkle_path: MerklePath,
    block_header: DoubleSha256,
}

impl BlockContainsTransaction {
    /// Constructs a circuit that asserts that `transaction_hash` has a valid `merkle_path` to the
    /// merkle root in `block_header`, and that `block_header` hashes to `block_hash`.
    ///
    /// **Note:** This does NOT assert that `transaction_hash` is actually the hash of a well formed
    /// transaction. In particular, `transaction_hash` might just be an internal (non-leaf) node in
    /// the transaction merkle tree, yet this circuit would pass.
    pub fn construct_circuit(
        builder: &CircuitBuilder,
        transaction_hash: [Wire; 4],
        merkle_path: Vec<([Wire; 4], Wire)>,
        merkle_path_len: Wire,
        block_header: [Wire; 10],
        block_hash: [Wire; 4],
    ) -> Self {
        // extract merkle root from block header
        let merkle_root = [
            join(builder, block_header[5], block_header[4]),
            join(builder, block_header[6], block_header[5]),
            join(builder, block_header[7], block_header[6]),
            join(builder, block_header[8], block_header[7]),
        ];

        // validity of merkle path
        let merkle_path = MerklePath::construct_circuit(
            builder,
            transaction_hash,
            merkle_path,
            merkle_root,
            merkle_path_len,
        );

        // `block_header` hashes to `block_hash`
        let block_header =
            DoubleSha256::construct_circuit(builder, block_header.to_vec(), block_hash);

        Self {
            merkle_path,
            block_header,
        }
    }

    pub fn populate_inner(
        &self,
        filler: &mut WitnessFiller,
        transaction_hash: [u8; 32],
        merkle_path: &[([u8; 32], SiblingSide)],
        block_header: &[u8],
    ) {
        self.merkle_path
            .populate_inner(filler, transaction_hash, merkle_path);
        self.block_header.populate_inner(filler, block_header);
    }
}

fn join(builder: &CircuitBuilder, b0: Wire, b1: Wire) -> Wire {
    let c0 = builder.shl(b0, 32);
    let c1 = builder.shr(b1, 32);
    builder.bxor(c0, c1)
}

#[cfg(test)]
mod tests {
    use binius_core::{Word, verify::verify_constraints};
    use binius_frontend::util::pack_bytes_into_wires_le;
    use hex_literal::hex;

    use super::*;

    #[test]
    fn test_valid() {
        // build circuit
        let builder = CircuitBuilder::new();
        let transaction_hash: [Wire; 4] = std::array::from_fn(|_| builder.add_witness());
        let merkle_path: Vec<([Wire; 4], Wire)> = std::iter::repeat_with(|| {
            (
                std::array::from_fn(|_| builder.add_witness()),
                builder.add_witness(),
            )
        })
        .take(30)
        .collect();
        let merkle_path_len = builder.add_witness();
        let block_header: [Wire; 10] = std::array::from_fn(|_| builder.add_witness());
        let block_hash: [Wire; 4] = std::array::from_fn(|_| builder.add_witness());
        let block_contains_transaction = BlockContainsTransaction::construct_circuit(
            &builder,
            transaction_hash,
            merkle_path.clone(),
            merkle_path_len,
            block_header,
            block_hash,
        );
        let circuit = builder.build();

        // populate witness
        let mut filler = circuit.new_witness_filler();
        let block_header_value = hex!(
            "000000264a14e21adad047d981c06a26446e345eda3d8beb807401000000000000000000fc01df2139954b36cebc3fa6fbf6a7160a67d34b67e5c4aa2a7ce46f5bb42a83642ea468b32c0217d14ba4d1"
        );
        let block_hash_value =
            hex!("228561b085b7524957e515605725901238299ff2793300000000000000000000");
        let transaction_hash_value =
            hex!("6f2f044a225e8b293c6e54cf2771bf4d17ba8904b1f61cf9c392965dcbda0b83");
        let merkle_path_value = vec![
            (
                hex!("783089645b0bc42d44e9d6a7ea62adf7a8a2adc6b7f0173d663369217b771b86"),
                SiblingSide::Right,
            ),
            (
                hex!("1eeeeb0cac1753a10ade3b34bd5bf0e005cdec82545abdafa38685c45e5f8ce5"),
                SiblingSide::Right,
            ),
            (
                hex!("e0d7426d603f1a817938cf366c8933d32185625fc821e3b1e964cb5f8e421501"),
                SiblingSide::Right,
            ),
            (
                hex!("7af6e333025422cf892198d216f146d70efe64119071ce0ee96fd195640230df"),
                SiblingSide::Left,
            ),
            (
                hex!("d848bf00d7563a26c9a43ad8cc2fa558f6a299629be20a078a6b197dcf15fc31"),
                SiblingSide::Right,
            ),
            (
                hex!("b643abf3df379ac748494a5eb3025299265fff543571f8b71935e533f672c9e8"),
                SiblingSide::Right,
            ),
            (
                hex!("b07d3ebc129da3ae9d1b9daee64daf74f8504ca5f9194cd006edee48b1bf4d00"),
                SiblingSide::Right,
            ),
            (
                hex!("4cd4173f585e793e48aa479269f38cd986b600c494135e9de33118a8e4ac03ed"),
                SiblingSide::Right,
            ),
            (
                hex!("4b5e59b8d22762cfc2906fa597b29c7eab7cd52d4b0cea9269e84e2aebce4101"),
                SiblingSide::Right,
            ),
            (
                hex!("2321cd016cb8f1a29f1bad981418bed2776bf61b1a729ca86a54f14790ce822b"),
                SiblingSide::Right,
            ),
            (
                hex!("fecdc8a219a271a9a969fdebf38068ffeaf25b7af353ee99e759eb0d05604218"),
                SiblingSide::Right,
            ),
            (
                hex!("1736c19cc6de7296453811916ddedba46c9bbd61a3450ad3dfb8bddb698b6ad0"),
                SiblingSide::Right,
            ),
        ];
        pack_bytes_into_wires_le(&mut filler, &block_header, &block_header_value);
        pack_bytes_into_wires_le(&mut filler, &block_hash, &block_hash_value);
        pack_bytes_into_wires_le(&mut filler, &transaction_hash, &transaction_hash_value);
        for i in 0..merkle_path_value.len() {
            pack_bytes_into_wires_le(&mut filler, &merkle_path[i].0, &merkle_path_value[i].0);
            filler[merkle_path[i].1] = match merkle_path_value[i].1 {
                SiblingSide::Left => Word::ZERO,
                SiblingSide::Right => Word::ALL_ONE,
            };
        }
        filler[merkle_path_len] = Word(12);
        block_contains_transaction.populate_inner(
            &mut filler,
            transaction_hash_value,
            &merkle_path_value,
            &block_header_value,
        );
        circuit.populate_wire_witness(&mut filler).unwrap();

        // check
        let constraint_system = circuit.constraint_system();
        verify_constraints(constraint_system, &filler.into_value_vec()).unwrap();
    }
}
