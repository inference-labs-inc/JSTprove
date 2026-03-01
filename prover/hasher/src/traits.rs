use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign},
};

use arith::Field;

pub trait FiatShamirHasher: Clone + Debug {
    /// Name for the hasher
    const NAME: &'static str;

    /// The size of the hash output in bytes.
    const DIGEST_SIZE: usize;

    /// Create a new hash instance.
    fn new() -> Self;

    /// Hash the input into the output.
    fn hash(&self, output: &mut [u8], input: &[u8]);

    /// Hash the input in place.
    fn hash_inplace(&self, buffer: &mut [u8]);
}

pub trait PoseidonStateTrait:
    Sized
    + Default
    + Clone
    + Debug
    + Add<Self, Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + AddAssign<Self>
    + for<'a> AddAssign<&'a Self>
    + Mul<Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + MulAssign<Self>
    + for<'a> MulAssign<&'a Self>
{
    /// ElemT is the element (base field) type constructing the poseidon state instance.
    type ElemT: Field;

    /// SBOX_POW is a pow \alpha for poseidon sbox, that \alpha >= 3 and gcd(\alpha, p - 1) = 1,
    /// where p is the modulus of the prime field ElemT.
    const SBOX_POW: usize;

    /// FULL_ROUNDS in a poseidon permutation.
    const FULL_ROUNDS: usize;

    /// PARTIAL_ROUNDS in a poseidon permutation.
    const PARTIAL_ROUNDS: usize;

    /// STATE_WIDTH stands for the number of field elements that build up a poseidon state.
    const STATE_WIDTH: usize;

    /// CAPACITY is the number of output field elements (field here stands for InputF).
    ///
    /// We ensure (roughly) the collision resilience is 128 bits by FIELD_SIZE, i.e.,
    /// how many bits we need to store a field element.
    const CAPACITY: usize = 128 / <Self::ElemT as Field>::FIELD_SIZE * 2;

    /// RATE is the number of input elements in a round of sponge absorbing.
    /// The invariant here is RATE + CAPACITY = STATE_WIDTH
    const RATE: usize = Self::STATE_WIDTH - Self::CAPACITY;

    /// from_elems constructs an instance of field hasher state from the input field elements.
    fn from_elems(elems: &[Self::ElemT]) -> Self;

    /// to_elems is a conjugate method of from_elems, that separates the state into elems.
    fn to_elems(&self) -> Vec<Self::ElemT>;

    /// apply_mds_matrix applies MDS matrix back to the Poseidon state.
    fn apply_mds_matrix(&mut self, mds_matrix: &[Self]) {
        let res = mds_matrix
            .iter()
            .map(|mds_col| {
                let hadamard: Self = self.clone() * mds_col;
                hadamard.to_elems().iter().sum()
            })
            .collect::<Vec<_>>();

        *self = Self::from_elems(&res)
    }

    /// full_round_sbox applies x -> x^(sbox_exp) to all elements in the state.
    fn full_round_sbox(&mut self) {
        let mut elts = self.to_elems();
        elts.iter_mut()
            .for_each(|e| *e = e.exp(Self::SBOX_POW as u128));
        *self = Self::from_elems(&elts);
    }

    /// partial_round_sbox applies x -> x^(sbox_exp) to the first element in the state.
    fn partial_round_sbox(&mut self) {
        let mut elts = self.to_elems();
        elts[0] = elts[0].exp(Self::SBOX_POW as u128);
        *self = Self::from_elems(&elts);
    }

    fn to_u8_slices(&self, output: &mut [u8]);
}
