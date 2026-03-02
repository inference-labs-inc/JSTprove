use arith::{Field, SimdField};
use gkr_hashers::PoseidonStateTrait;

use crate::{M31x16, M31};

impl PoseidonStateTrait for M31x16 {
    type ElemT = M31;

    const SBOX_POW: usize = 5;

    const FULL_ROUNDS: usize = 8;

    const PARTIAL_ROUNDS: usize = 14;

    const STATE_WIDTH: usize = 16;

    fn from_elems(elems: &[Self::ElemT]) -> Self {
        Self::pack(elems)
    }

    fn to_elems(&self) -> Vec<Self::ElemT> {
        self.unpack()
    }

    fn full_round_sbox(&mut self) {
        *self = self.exp(Self::SBOX_POW as u128);
    }

    fn to_u8_slices(&self, output: &mut [u8]) {
        self.to_bytes(output);
    }
}
