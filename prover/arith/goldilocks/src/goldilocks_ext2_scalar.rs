use std::{
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::{field_common, ExtensionField, FFTField, Field, SimdField};
use ethnum::U256;
use rand::RngCore;
use serdes::{ExpSerde, SerdeResult};

use crate::goldilocks_ext::GoldilocksExt2;

#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq)]
#[repr(transparent)]
pub struct GoldilocksExt2Scalar(pub GoldilocksExt2);

impl From<GoldilocksExt2> for GoldilocksExt2Scalar {
    #[inline(always)]
    fn from(v: GoldilocksExt2) -> Self {
        Self(v)
    }
}

impl From<GoldilocksExt2Scalar> for GoldilocksExt2 {
    #[inline(always)]
    fn from(v: GoldilocksExt2Scalar) -> Self {
        v.0
    }
}

impl ExpSerde for GoldilocksExt2Scalar {
    #[inline(always)]
    fn serialize_into<W: std::io::Write>(&self, writer: W) -> SerdeResult<()> {
        self.0.serialize_into(writer)
    }

    #[inline(always)]
    fn deserialize_from<R: std::io::Read>(reader: R) -> SerdeResult<Self> {
        GoldilocksExt2::deserialize_from(reader).map(Self)
    }
}

impl From<u32> for GoldilocksExt2Scalar {
    #[inline(always)]
    fn from(v: u32) -> Self {
        Self(GoldilocksExt2::from(v))
    }
}

impl Neg for GoldilocksExt2Scalar {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

#[inline(always)]
fn add_internal(a: &GoldilocksExt2Scalar, b: &GoldilocksExt2Scalar) -> GoldilocksExt2Scalar {
    GoldilocksExt2Scalar(a.0 + b.0)
}

#[inline(always)]
fn sub_internal(a: &GoldilocksExt2Scalar, b: &GoldilocksExt2Scalar) -> GoldilocksExt2Scalar {
    GoldilocksExt2Scalar(a.0 - b.0)
}

#[inline(always)]
fn mul_internal(a: &GoldilocksExt2Scalar, b: &GoldilocksExt2Scalar) -> GoldilocksExt2Scalar {
    GoldilocksExt2Scalar(a.0 * b.0)
}

field_common!(GoldilocksExt2Scalar);

impl Field for GoldilocksExt2Scalar {
    const NAME: &'static str = "GoldilocksExt2Scalar";
    const SIZE: usize = GoldilocksExt2::SIZE;
    const FIELD_SIZE: usize = GoldilocksExt2::FIELD_SIZE;
    const ZERO: Self = Self(GoldilocksExt2::ZERO);
    const ONE: Self = Self(GoldilocksExt2::ONE);
    const INV_2: Self = Self(GoldilocksExt2::INV_2);
    const MODULUS: U256 = GoldilocksExt2::MODULUS;

    #[inline(always)]
    fn zero() -> Self {
        Self(GoldilocksExt2::zero())
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
    #[inline(always)]
    fn one() -> Self {
        Self(GoldilocksExt2::one())
    }
    fn random_unsafe(rng: impl RngCore) -> Self {
        Self(GoldilocksExt2::random_unsafe(rng))
    }
    fn random_bool(rng: impl RngCore) -> Self {
        Self(GoldilocksExt2::random_bool(rng))
    }
    #[inline(always)]
    fn as_u32_unchecked(&self) -> u32 {
        self.0.as_u32_unchecked()
    }
    #[inline(always)]
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        Self(GoldilocksExt2::from_uniform_bytes(bytes))
    }
    #[inline(always)]
    fn square(&self) -> Self {
        Self(self.0.square())
    }
    fn inv(&self) -> Option<Self> {
        self.0.inv().map(Self)
    }
    #[inline(always)]
    fn to_u256(&self) -> U256 {
        let lo = self.0.v[0].to_u256();
        let hi = self.0.v[1].to_u256();
        lo | (hi << 64)
    }
    #[inline(always)]
    fn from_u256(v: U256) -> Self {
        use crate::Goldilocks;
        let lo = v & U256::from(u64::MAX);
        let hi = (v >> 64) & U256::from(u64::MAX);
        Self(GoldilocksExt2 {
            v: [Goldilocks::from_u256(lo), Goldilocks::from_u256(hi)],
        })
    }
}

impl Ord for GoldilocksExt2Scalar {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.v[0]
            .cmp(&other.0.v[0])
            .then(self.0.v[1].cmp(&other.0.v[1]))
    }
}

impl PartialOrd for GoldilocksExt2Scalar {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl ExtensionField for GoldilocksExt2Scalar {
    const DEGREE: usize = 1;
    const W: u32 = 1;
    const X: Self = Self(GoldilocksExt2::ZERO);

    type BaseField = Self;

    #[inline(always)]
    fn mul_by_base_field(&self, base: &Self::BaseField) -> Self {
        Self(self.0 * base.0)
    }

    #[inline(always)]
    fn add_by_base_field(&self, base: &Self::BaseField) -> Self {
        Self(self.0 + base.0)
    }

    #[inline(always)]
    fn mul_by_x(&self) -> Self {
        *self
    }

    #[inline(always)]
    fn to_limbs(&self) -> Vec<Self::BaseField> {
        vec![*self]
    }

    #[inline(always)]
    fn from_limbs(limbs: &[Self::BaseField]) -> Self {
        limbs[0]
    }
}

impl SimdField for GoldilocksExt2Scalar {
    type Scalar = Self;
    const PACK_SIZE: usize = 1;

    #[inline(always)]
    fn scale(&self, challenge: &Self::Scalar) -> Self {
        Self(self.0 * challenge.0)
    }
    #[inline(always)]
    fn pack_full(x: &Self::Scalar) -> Self {
        *x
    }
    #[inline(always)]
    fn pack(base_vec: &[Self::Scalar]) -> Self {
        assert_eq!(base_vec.len(), 1);
        base_vec[0]
    }
    #[inline(always)]
    fn unpack(&self) -> Vec<Self::Scalar> {
        vec![*self]
    }
}

impl FFTField for GoldilocksExt2Scalar {
    const TWO_ADICITY: usize = GoldilocksExt2::TWO_ADICITY;

    #[inline(always)]
    fn root_of_unity() -> Self {
        Self(GoldilocksExt2::root_of_unity())
    }
}
