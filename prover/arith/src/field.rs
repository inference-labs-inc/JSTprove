use ark_std::Zero;
use ethnum::U256;
use rand::RngCore;
use serdes::ExpSerde;

use std::{
    fmt::Debug,
    hash::Hash,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Field definitions.
pub trait Field:
    Copy
    + Clone
    + Debug
    + Default
    + PartialEq
    + From<u32>
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Sum
    + Product
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> Sum<&'a Self>
    + for<'a> Product<&'a Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + ExpSerde
    + Hash
    + Eq
    + PartialOrd
    + Ord
    + Send
    + Sync
{
    /// name
    const NAME: &'static str;

    /// size required to store the data
    const SIZE: usize;

    /// Field element size in bits, e.g., log_2(modulus), rounded up to the next power of 2.
    const FIELD_SIZE: usize;

    /// zero
    const ZERO: Self;

    /// One
    const ONE: Self;

    /// Inverse of 2
    const INV_2: Self;

    /// MODULUS
    const MODULUS: U256;

    // ====================================
    // constants
    // ====================================
    /// Zero element
    fn zero() -> Self {
        Self::ZERO
    }

    /// Is zero
    fn is_zero(&self) -> bool;

    /// Identity element
    fn one() -> Self {
        Self::ONE
    }

    // ====================================
    // generators
    // ====================================
    /// create a random element from rng.
    /// test only -- the output may not be uniformly random.
    fn random_unsafe(rng: impl RngCore) -> Self;

    /// create a random boolean element from rng
    fn random_bool(rng: impl RngCore) -> Self;

    /// expose the element as u32.
    fn as_u32_unchecked(&self) -> u32;

    /// sample from a 32 bytes
    fn from_uniform_bytes(bytes: &[u8]) -> Self;

    /// convert to bytes
    #[inline(always)]
    fn to_bytes(&self, target: &mut [u8]) {
        let mut res = vec![];
        self.serialize_into(&mut res).unwrap();
        target.copy_from_slice(&res);
    }

    /// convert from bytes
    #[inline(always)]
    fn set_in_bytes(&mut self, source: &[u8]) {
        *self = Self::deserialize_from(source).unwrap();
    }

    /// convert to u256
    // todo: a cleaner way to do this is to trait bound Into<U256>
    // but this requires modifications on ff or ethnum crate
    fn to_u256(&self) -> U256 {
        unimplemented!()
    }

    /// convert from u256
    // todo: a cleaner way to do this is to trait bound From<U256>
    // but this requires modifications on ff or ethnum crate
    fn from_u256(_: U256) -> Self {
        unimplemented!()
    }

    // ====================================
    // arithmetic
    // ====================================
    #[inline(always)]
    fn square(&self) -> Self {
        *self * *self
    }

    /// Doubling
    #[inline(always)]
    fn double(&self) -> Self {
        *self + *self
    }

    /// Exp
    fn exp(&self, exponent: u128) -> Self {
        let mut e = exponent;
        let mut res = Self::one();
        let mut t = *self;
        while !e.is_zero() {
            let b = e & 1;
            if b == 1 {
                res *= t;
            }
            t = t * t;
            e >>= 1;
        }
        res
    }

    #[inline(always)]
    fn exp_power_of_2(&self, power_log: usize) -> Self {
        let mut res = *self;
        for _ in 0..power_log {
            res = res.square();
        }
        res
    }

    /// find the inverse of the element; return None if not exist
    fn inv(&self) -> Option<Self>;

    /// multiply by 2
    #[inline(always)]
    fn mul_by_2(&self) -> Self {
        *self + *self
    }

    #[inline(always)]
    /// multiply by 3
    fn mul_by_3(&self) -> Self {
        *self + *self + *self
    }

    #[inline(always)]
    /// multiply by 5
    fn mul_by_5(&self) -> Self {
        let double = self.mul_by_2();
        let quad = double.mul_by_2();
        *self + quad
    }

    #[inline(always)]
    /// multiply by 6
    fn mul_by_6(&self) -> Self {
        let t = self.mul_by_3();
        t + t
    }
}
