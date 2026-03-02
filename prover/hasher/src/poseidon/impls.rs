use std::fmt::Debug;

use arith::Field;
use tiny_keccak::{Hasher, Keccak};

use crate::{FiatShamirHasher, PoseidonStateTrait};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PoseidonPermutation<State: PoseidonStateTrait> {
    pub mds_matrix: Vec<State>,
    pub round_constants: Vec<State>,
}

pub const POSEIDON_SEED_PREFIX: &str = "poseidon_seed";

pub fn get_constants<State: PoseidonStateTrait>(round_num: usize) -> Vec<State> {
    let seed = format!(
        "{POSEIDON_SEED_PREFIX}_{}_{}",
        State::ElemT::NAME,
        State::STATE_WIDTH
    );

    let mut keccak = Keccak::v256();
    let mut buffer = [0u8; 32];
    keccak.update(seed.as_bytes());
    keccak.finalize(&mut buffer);

    (0..round_num)
        .map(|_| {
            let state_elems: Vec<_> = (0..State::STATE_WIDTH)
                .map(|_| {
                    let mut keccak = Keccak::v256();
                    keccak.update(&buffer);
                    keccak.finalize(&mut buffer);
                    State::ElemT::from_uniform_bytes(&buffer)
                })
                .collect();
            State::from_elems(&state_elems)
        })
        .collect()
}

pub const MATRIX_CIRC_MDS_8_SML_ROW: [u32; 8] = [7, 1, 3, 8, 8, 3, 4, 9];

pub const MATRIX_CIRC_MDS_12_SML_ROW: [u32; 12] = [1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10];

pub const MATRIX_CIRC_MDS_16_SML_ROW: [u32; 16] =
    [1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3];

pub fn get_mds_matrix<State: PoseidonStateTrait>() -> Vec<State> {
    let mds_first_row: &[u32] = match State::STATE_WIDTH {
        8 => &MATRIX_CIRC_MDS_8_SML_ROW,
        12 => &MATRIX_CIRC_MDS_12_SML_ROW,
        16 => &MATRIX_CIRC_MDS_16_SML_ROW,
        _ => unimplemented!("unsupported state width for MDS matrix"),
    };

    let buffer: Vec<_> = [mds_first_row, mds_first_row]
        .concat()
        .iter()
        .cloned()
        .map(From::from)
        .collect();

    (0..State::STATE_WIDTH)
        .map(|i| State::from_elems(&buffer[i..i + State::STATE_WIDTH]))
        .collect()
}
impl<State: PoseidonStateTrait> PoseidonPermutation<State> {
    fn new() -> Self {
        let total_rounds = State::FULL_ROUNDS + State::PARTIAL_ROUNDS;

        Self {
            mds_matrix: get_mds_matrix::<State>(),
            round_constants: get_constants::<State>(total_rounds),
        }
    }

    fn permute(&self, state: &mut State) {
        let half_full_rounds = State::FULL_ROUNDS / 2;
        let partial_ends = State::FULL_ROUNDS / 2 + State::PARTIAL_ROUNDS;

        (0..half_full_rounds).for_each(|i| {
            *state += &self.round_constants[i];
            state.apply_mds_matrix(&self.mds_matrix);
            state.full_round_sbox();
        });

        (half_full_rounds..partial_ends).for_each(|i| {
            *state += &self.round_constants[i];
            state.apply_mds_matrix(&self.mds_matrix);
            state.partial_round_sbox();
        });

        (partial_ends..half_full_rounds + partial_ends).for_each(|i| {
            *state += &self.round_constants[i];
            state.apply_mds_matrix(&self.mds_matrix);
            state.full_round_sbox();
        });
    }
}

impl<State: PoseidonStateTrait> PoseidonPermutation<State> {
    fn hash_u8_to_state(&self, input: &[u8]) -> State {
        let u8_chunk_size = State::RATE * State::ElemT::SIZE;
        let mut res = State::default();
        let chunks = input.chunks_exact(u8_chunk_size);
        let mut remainder = chunks.remainder().to_vec();

        for chunk in chunks {
            let mut state_elts = vec![State::ElemT::ZERO; State::STATE_WIDTH];
            for (elem, elts) in chunk
                .chunks(State::ElemT::SIZE)
                .zip(state_elts[State::CAPACITY..].iter_mut())
            {
                *elts = State::ElemT::from_uniform_bytes(elem);
            }
            let state = State::from_elems(&state_elts);

            res += state;
            self.permute(&mut res);
        }

        if !remainder.is_empty() {
            remainder.resize(u8_chunk_size, 0);

            let mut state_elts = vec![State::ElemT::ZERO; State::STATE_WIDTH];
            for (elem, elts) in remainder
                .chunks(State::ElemT::SIZE)
                .zip(state_elts[State::CAPACITY..].iter_mut())
            {
                *elts = State::ElemT::from_uniform_bytes(elem);
            }
            let state = State::from_elems(&state_elts);

            res += state;
            self.permute(&mut res);
        }

        res
    }
}

impl<State: PoseidonStateTrait> FiatShamirHasher for PoseidonPermutation<State> {
    const NAME: &'static str = "Poseidon Field Hasher";

    const DIGEST_SIZE: usize = State::STATE_WIDTH * State::ElemT::SIZE;

    fn new() -> Self {
        Self::new()
    }

    fn hash(&self, output: &mut [u8], input: &[u8]) {
        assert!(output.len() == Self::DIGEST_SIZE);
        let res = self.hash_u8_to_state(input);
        res.to_u8_slices(output);
    }

    fn hash_inplace(&self, buffer: &mut [u8]) {
        assert!(buffer.len() == Self::DIGEST_SIZE);
        let res = self.hash_u8_to_state(buffer);
        res.to_u8_slices(buffer);
    }
}

pub type PoseidonFiatShamirHasher<State> = PoseidonPermutation<State>;
