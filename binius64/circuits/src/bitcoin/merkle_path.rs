// Copyright 2025 Irreducible Inc.
//! The merkle proof for a transaction in a Bitcoin block.

use binius_frontend::{CircuitBuilder, Wire, WitnessFiller, util::pack_bytes_into_wires_le};

use super::double_sha256::DoubleSha256;

#[derive(Debug, Copy, Clone)]
pub enum SiblingSide {
    Left,
    Right,
}

/// Stores some intermediate wires of the circuit, so that they can later be populated with
/// [`Self::populate_inner`].
pub struct MerklePath {
    pairs: Vec<(DoubleSha256, [Wire; 4])>,
}

impl MerklePath {
    /// Constructs a circuit that asserts that if you repeatedly hash the `leaf` together with the
    /// `siblings` using double-SHA256, then you obtain `root`.
    ///
    /// The extra wire for each `sibling` indicated if it's a left or a right sibling. `0` means
    /// it's a left sibling, `1` means it's a right sibling.
    ///
    /// `siblings.len()` determines maximal length of merkle path
    pub fn construct_circuit(
        builder: &CircuitBuilder,
        mut leaf: [Wire; 4],
        siblings: Vec<([Wire; 4], Wire)>,
        root: [Wire; 4],
        length: Wire,
    ) -> Self {
        let mut pairs = Vec::new();
        for (i, sib) in siblings.into_iter().enumerate() {
            let digest = std::array::from_fn(|_| builder.add_witness());
            let message = vec![
                builder.select(sib.1, leaf[0], sib.0[0]),
                builder.select(sib.1, leaf[1], sib.0[1]),
                builder.select(sib.1, leaf[2], sib.0[2]),
                builder.select(sib.1, leaf[3], sib.0[3]),
                builder.select(sib.1, sib.0[0], leaf[0]),
                builder.select(sib.1, sib.0[1], leaf[1]),
                builder.select(sib.1, sib.0[2], leaf[2]),
                builder.select(sib.1, sib.0[3], leaf[3]),
            ];
            pairs.push((
                DoubleSha256::construct_circuit(builder, message, digest),
                digest,
            ));

            let current_length = builder.add_constant_64(i as u64);
            let past_length = builder.icmp_ult(current_length, length);
            let digest_or_passthrough = [
                builder.select(past_length, digest[0], leaf[0]),
                builder.select(past_length, digest[1], leaf[1]),
                builder.select(past_length, digest[2], leaf[2]),
                builder.select(past_length, digest[3], leaf[3]),
            ];
            leaf = digest_or_passthrough;
        }
        builder.assert_eq_v("last digest equals root", root, leaf);

        Self { pairs }
    }

    pub fn populate_inner(
        &self,
        filler: &mut WitnessFiller,
        mut leaf: [u8; 32],
        siblings: &[([u8; 32], SiblingSide)],
    ) {
        assert!(siblings.len() <= self.pairs.len());

        let dummy_sibling: ([u8; 32], SiblingSide) = ([0; 32], SiblingSide::Left);

        for (i, pair) in self.pairs.iter().enumerate() {
            let sibling = siblings.get(i).unwrap_or(&dummy_sibling);

            let message: Vec<u8> = match sibling.1 {
                SiblingSide::Left => [sibling.0, leaf].concat(),
                SiblingSide::Right => [leaf, sibling.0].concat(),
            };
            assert_eq!(message.len(), 64);
            let digest = pair.0.populate_inner(filler, &message);
            pack_bytes_into_wires_le(filler, &pair.1, &digest);
            leaf = if i < siblings.len() { digest } else { leaf };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use binius_core::{Word, verify::verify_constraints};
    use hex_literal::hex;

    use super::*;

    #[test]
    fn test_valid() {
        // construct circuit
        let builder = CircuitBuilder::new();
        let leaf: [Wire; 4] = array::from_fn(|_| builder.add_witness());
        let siblings: Vec<([Wire; 4], Wire)> = std::iter::repeat_with(|| {
            (
                array::from_fn(|_| builder.add_witness()),
                builder.add_witness(),
            )
        })
        .take(2)
        .collect();
        let root: [Wire; 4] = array::from_fn(|_| builder.add_witness());
        let length: Wire = builder.add_witness();
        let merkle_path =
            MerklePath::construct_circuit(&builder, leaf, siblings.clone(), root, length);
        let circuit = builder.build();

        // populate_witness
        let mut filler = circuit.new_witness_filler();
        let leaf_value = hex!("a2b6b171aae6007508e5c8fabec6b662bad3e4594e09405cac7b249e5f1e5155");
        let siblings_value = vec![
            (
                hex!("1346be1a16a09b5fcc5bca52d39c2529396f0fa6a654f3978807ff79eaf91d66"),
                SiblingSide::Right,
            ),
            (
                hex!("557cc3606e7197ff5a7b6cda46e409445b1ab58d8d4ebf1bc3d95764c32ad877"),
                SiblingSide::Left,
            ),
        ];
        let root_value = hex!("5802c63ef536216cf01a0dd0b32c01f5e31536aa773eb6e1d46fd42f66516eba");
        pack_bytes_into_wires_le(&mut filler, &leaf, &leaf_value);
        pack_bytes_into_wires_le(&mut filler, &siblings[0].0, &siblings_value[0].0);
        filler[siblings[0].1] = Word::ALL_ONE;
        pack_bytes_into_wires_le(&mut filler, &siblings[1].0, &siblings_value[1].0);
        filler[siblings[1].1] = Word::ZERO;
        pack_bytes_into_wires_le(&mut filler, &root, &root_value);
        filler[length] = Word(2);
        merkle_path.populate_inner(&mut filler, leaf_value, &siblings_value);
        circuit.populate_wire_witness(&mut filler).unwrap();

        // check
        let constraint_system = circuit.constraint_system();
        verify_constraints(constraint_system, &filler.into_value_vec()).unwrap();
    }

    #[test]
    fn test_invalid_side() {
        // construct circuit
        let builder = CircuitBuilder::new();
        let leaf: [Wire; 4] = array::from_fn(|_| builder.add_witness());
        let siblings: Vec<([Wire; 4], Wire)> = std::iter::repeat_with(|| {
            (
                array::from_fn(|_| builder.add_witness()),
                builder.add_witness(),
            )
        })
        .take(2)
        .collect();
        let root: [Wire; 4] = array::from_fn(|_| builder.add_witness());
        let length: Wire = builder.add_witness();
        let merkle_path =
            MerklePath::construct_circuit(&builder, leaf, siblings.clone(), root, length);
        let circuit = builder.build();

        // populate_witness
        let mut filler = circuit.new_witness_filler();
        let leaf_value = hex!("a2b6b171aae6007508e5c8fabec6b662bad3e4594e09405cac7b249e5f1e5155");
        let siblings_value = vec![
            (
                hex!("1346be1a16a09b5fcc5bca52d39c2529396f0fa6a654f3978807ff79eaf91d66"),
                SiblingSide::Right,
            ),
            (
                hex!("557cc3606e7197ff5a7b6cda46e409445b1ab58d8d4ebf1bc3d95764c32ad877"),
                SiblingSide::Right,
            ),
        ];
        let root_value = hex!("5802c63ef536216cf01a0dd0b32c01f5e31536aa773eb6e1d46fd42f66516eba");
        pack_bytes_into_wires_le(&mut filler, &leaf, &leaf_value);
        pack_bytes_into_wires_le(&mut filler, &siblings[0].0, &siblings_value[0].0);
        filler[siblings[0].1] = Word::ALL_ONE;
        pack_bytes_into_wires_le(&mut filler, &siblings[1].0, &siblings_value[1].0);
        filler[siblings[1].1] = Word::ZERO;
        pack_bytes_into_wires_le(&mut filler, &root, &root_value);
        filler[length] = Word(2);
        merkle_path.populate_inner(&mut filler, leaf_value, &siblings_value);
        circuit.populate_wire_witness(&mut filler).unwrap_err();
    }

    #[test]
    fn test_invalid_path() {
        // construct circuit
        let builder = CircuitBuilder::new();
        let leaf: [Wire; 4] = array::from_fn(|_| builder.add_witness());
        let siblings: Vec<([Wire; 4], Wire)> = std::iter::repeat_with(|| {
            (
                array::from_fn(|_| builder.add_witness()),
                builder.add_witness(),
            )
        })
        .take(2)
        .collect();
        let root: [Wire; 4] = array::from_fn(|_| builder.add_witness());
        let length: Wire = builder.add_witness();
        let merkle_path =
            MerklePath::construct_circuit(&builder, leaf, siblings.clone(), root, length);
        let circuit = builder.build();

        // populate_witness
        let mut filler = circuit.new_witness_filler();
        let leaf_value = hex!("a2b6b171aae6007508e5c8fabec6b662bad3e4594e09405cac7b249e5f1e5155");
        let siblings_value = vec![
            (
                hex!("1346be1a16a09b5fcc5bca52d39c2529396f0fa6a654f3978807ff79eaf91d66"),
                SiblingSide::Right,
            ),
            (
                hex!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
                SiblingSide::Left,
            ),
        ];
        let root_value = hex!("5802c63ef536216cf01a0dd0b32c01f5e31536aa773eb6e1d46fd42f66516eba");
        pack_bytes_into_wires_le(&mut filler, &leaf, &leaf_value);
        pack_bytes_into_wires_le(&mut filler, &siblings[0].0, &siblings_value[0].0);
        filler[siblings[0].1] = Word::ALL_ONE;
        pack_bytes_into_wires_le(&mut filler, &siblings[1].0, &siblings_value[1].0);
        filler[siblings[1].1] = Word::ZERO;
        pack_bytes_into_wires_le(&mut filler, &root, &root_value);
        filler[length] = Word(2);
        merkle_path.populate_inner(&mut filler, leaf_value, &siblings_value);
        circuit.populate_wire_witness(&mut filler).unwrap_err();
    }

    #[test]
    fn test_valid_long() {
        // construct circuit
        let builder = CircuitBuilder::new();
        let leaf: [Wire; 4] = array::from_fn(|_| builder.add_witness());
        let siblings: Vec<([Wire; 4], Wire)> = std::iter::repeat_with(|| {
            (
                array::from_fn(|_| builder.add_witness()),
                builder.add_witness(),
            )
        })
        .take(30)
        .collect();
        let root: [Wire; 4] = array::from_fn(|_| builder.add_witness());
        let length: Wire = builder.add_witness();
        let merkle_path =
            MerklePath::construct_circuit(&builder, leaf, siblings.clone(), root, length);
        let circuit = builder.build();

        // populate_witness
        let mut filler = circuit.new_witness_filler();
        let leaf_value = hex!("6f2f044a225e8b293c6e54cf2771bf4d17ba8904b1f61cf9c392965dcbda0b83");
        let siblings_value = vec![
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
        let root_value = hex!("fc01df2139954b36cebc3fa6fbf6a7160a67d34b67e5c4aa2a7ce46f5bb42a83");
        pack_bytes_into_wires_le(&mut filler, &leaf, &leaf_value);
        for i in 0..siblings_value.len() {
            pack_bytes_into_wires_le(&mut filler, &siblings[i].0, &siblings_value[i].0);
            filler[siblings[i].1] = match siblings_value[i].1 {
                SiblingSide::Left => Word::ZERO,
                SiblingSide::Right => Word::ALL_ONE,
            };
        }
        pack_bytes_into_wires_le(&mut filler, &root, &root_value);
        merkle_path.populate_inner(&mut filler, leaf_value, &siblings_value);
        filler[length] = Word(12);
        circuit.populate_wire_witness(&mut filler).unwrap();

        // check
        let constraint_system = circuit.constraint_system();
        verify_constraints(constraint_system, &filler.into_value_vec()).unwrap();
    }
}
