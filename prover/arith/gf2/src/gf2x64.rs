use std::{
    mem::transmute,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::{Field, SimdField};
use ethnum::U256;
use serdes::ExpSerde;

use crate::{GF2x8, GF2};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, ExpSerde)]
pub struct GF2x64 {
    pub v: u64,
}

impl Field for GF2x64 {
    const NAME: &'static str = "Galois Field 2 SIMD 64";

    const SIZE: usize = 8;

    const FIELD_SIZE: usize = 1;

    const ZERO: Self = GF2x64 { v: 0 };

    const ONE: Self = GF2x64 { v: !0u64 };

    #[doc(hidden)]
    const INV_2: Self = unimplemented!(); // NOTE: should not be used

    #[doc(hidden)]
    const MODULUS: U256 = unimplemented!(); // should not be used

    #[inline(always)]
    fn zero() -> Self {
        GF2x64::ZERO
    }

    #[inline(always)]
    fn one() -> Self {
        GF2x64::ONE
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.v == 0
    }

    #[inline(always)]
    fn random_unsafe(mut rng: impl rand::RngCore) -> Self {
        GF2x64 { v: rng.next_u64() }
    }

    #[inline(always)]
    fn random_bool(mut rng: impl rand::RngCore) -> Self {
        GF2x64 { v: rng.next_u64() }
    }

    #[inline(always)]
    fn exp(&self, exponent: u128) -> Self {
        if exponent == 0 {
            return Self::one();
        }
        *self
    }

    #[inline(always)]
    fn inv(&self) -> Option<Self> {
        unimplemented!()
    }

    #[inline(always)]
    fn as_u32_unchecked(&self) -> u32 {
        self.v as u32
    }

    #[inline(always)]
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        let mut buf = [0u8; 8];
        buf[..].copy_from_slice(&bytes[..8]);
        GF2x64 {
            v: u64::from_le_bytes(buf),
        }
    }

    #[inline(always)]
    fn mul_by_5(&self) -> Self {
        *self
    }

    #[inline(always)]
    fn mul_by_6(&self) -> Self {
        Self::ZERO
    }
}

impl Mul<&GF2x64> for GF2x64 {
    type Output = GF2x64;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: &GF2x64) -> Self::Output {
        GF2x64 { v: self.v & rhs.v }
    }
}

impl Mul<GF2x64> for GF2x64 {
    type Output = GF2x64;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: GF2x64) -> GF2x64 {
        GF2x64 { v: self.v & rhs.v }
    }
}

impl MulAssign<&GF2x64> for GF2x64 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn mul_assign(&mut self, rhs: &GF2x64) {
        self.v &= rhs.v;
    }
}

impl MulAssign<GF2x64> for GF2x64 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn mul_assign(&mut self, rhs: GF2x64) {
        self.v &= rhs.v;
    }
}

impl Sub for GF2x64 {
    type Output = GF2x64;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: GF2x64) -> GF2x64 {
        GF2x64 { v: self.v ^ rhs.v }
    }
}

impl SubAssign for GF2x64 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn sub_assign(&mut self, rhs: GF2x64) {
        self.v ^= rhs.v;
    }
}

impl Add for GF2x64 {
    type Output = GF2x64;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: GF2x64) -> GF2x64 {
        GF2x64 { v: self.v ^ rhs.v }
    }
}

impl AddAssign for GF2x64 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, rhs: GF2x64) {
        self.v ^= rhs.v;
    }
}

impl Add<&GF2x64> for GF2x64 {
    type Output = GF2x64;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: &GF2x64) -> GF2x64 {
        GF2x64 { v: self.v ^ rhs.v }
    }
}

impl Sub<&GF2x64> for GF2x64 {
    type Output = GF2x64;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: &GF2x64) -> GF2x64 {
        GF2x64 { v: self.v ^ rhs.v }
    }
}

impl<T: std::borrow::Borrow<GF2x64>> std::iter::Sum<T> for GF2x64 {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, item| acc + item.borrow())
    }
}

impl<T: std::borrow::Borrow<GF2x64>> std::iter::Product<T> for GF2x64 {
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, item| acc * item.borrow())
    }
}

impl Neg for GF2x64 {
    type Output = GF2x64;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn neg(self) -> GF2x64 {
        GF2x64 { v: self.v }
    }
}

impl AddAssign<&GF2x64> for GF2x64 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, rhs: &GF2x64) {
        self.v ^= rhs.v;
    }
}

impl SubAssign<&GF2x64> for GF2x64 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn sub_assign(&mut self, rhs: &GF2x64) {
        self.v ^= rhs.v;
    }
}

impl From<u32> for GF2x64 {
    #[inline(always)]
    fn from(v: u32) -> Self {
        assert!(v < 2);
        if v == 0 {
            GF2x64 { v: 0 }
        } else {
            GF2x64 { v: !0u64 }
        }
    }
}

impl From<GF2> for GF2x64 {
    #[inline(always)]
    fn from(v: GF2) -> Self {
        assert!(v.v < 2);
        if v.v == 0 {
            GF2x64 { v: 0 }
        } else {
            GF2x64 { v: !0u64 }
        }
    }
}

impl SimdField for GF2x64 {
    #[inline(always)]
    fn scale(&self, challenge: &Self::Scalar) -> Self {
        if challenge.v == 0 {
            Self::zero()
        } else {
            *self
        }
    }

    #[inline(always)]
    fn pack_full(base: &Self::Scalar) -> Self {
        match base.v {
            0 => Self::zero(),
            1 => Self::one(),
            _ => unimplemented!(),
        }
    }

    #[inline(always)]
    fn pack(base_vec: &[Self::Scalar]) -> Self {
        assert!(base_vec.len() == Self::PACK_SIZE);
        let mut packed_to_gf2x8 = [GF2x8::ZERO; Self::PACK_SIZE / GF2x8::PACK_SIZE];
        packed_to_gf2x8
            .iter_mut()
            .zip(base_vec.chunks(GF2x8::PACK_SIZE))
            .for_each(|(gf2x8, pack)| *gf2x8 = GF2x8::pack(pack));
        unsafe { transmute(packed_to_gf2x8) }
    }

    #[inline(always)]
    fn unpack(&self) -> Vec<Self::Scalar> {
        let packed_to_gf2x8: [GF2x8; Self::PACK_SIZE / GF2x8::PACK_SIZE] =
            unsafe { transmute(*self) };

        packed_to_gf2x8
            .iter()
            .flat_map(|packed| packed.unpack())
            .collect()
    }

    type Scalar = crate::GF2;

    const PACK_SIZE: usize = 64;
}

impl Ord for GF2x64 {
    #[inline(always)]
    fn cmp(&self, _: &Self) -> std::cmp::Ordering {
        unimplemented!("Ord for GF2x64 is not supported")
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for GF2x64 {
    #[inline(always)]
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        unimplemented!("PartialOrd for GF2x64 is not supported")
    }
}
