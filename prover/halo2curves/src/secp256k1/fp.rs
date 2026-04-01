use crate::arithmetic::{adc, bigint_geq, mac, macx, sbb};
use crate::extend_field_legendre;
use crate::ff::{FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use crate::{
    field_arithmetic, field_bits, field_common, field_specific, impl_add_binop_specify_output,
    impl_binops_additive, impl_binops_additive_specify_output, impl_binops_multiplicative,
    impl_binops_multiplicative_mixed, impl_from_u64_u32, impl_sub_binop_specify_output,
    impl_sum_prod,
};
use core::convert::TryInto;
use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
use rand::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

/// This represents an element of $\mathbb{F}_p$ where
///
/// `p = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f`
///
/// is the base field of the secp256k1 curve.
// The internal representation of this type is four 64-bit unsigned
// integers in little-endian order. `Fp` values are always in
// Montgomery form; i.e., Fp(a) = aR mod p, with R = 2^256.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fp(pub(crate) [u64; 4]);

#[cfg(feature = "derive_serde")]
crate::serialize_deserialize_32_byte_primefield!(Fp);

/// Constant representing the modulus
/// p = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f
const MODULUS: Fp = Fp([
    0xfffffffefffffc2f,
    0xffffffffffffffff,
    0xffffffffffffffff,
    0xffffffffffffffff,
]);

/// Constant representing the multiplicative generator of the modulus.
/// It's derived with SageMath with: `GF(MODULUS).primitive_element()`.
const MULTIPLICATIVE_GENERATOR: Fp = Fp::from_raw([0x03, 0x00, 0x00, 0x00]);

/// The modulus as u32 limbs.
#[cfg(not(target_pointer_width = "64"))]
const MODULUS_LIMBS_32: [u32; 8] = [
    0xffff_fc2f,
    0xffff_fffe,
    0xffff_ffff,
    0xffff_ffff,
    0xffff_ffff,
    0xffff_ffff,
    0xffff_ffff,
    0xffff_ffff,
];

/// Constant representing the modulus as static str
const MODULUS_STR: &str = "0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f";

/// INV = -(p^{-1} mod 2^64) mod 2^64
const INV: u64 = 0xd838091dd2253531;

/// R = 2^256 mod p
/// 0x1000003d1
const R: Fp = Fp([0x1000003d1, 0, 0, 0]);

/// R^2 = 2^512 mod p
/// 0x1000007a2000e90a1
const R2: Fp = Fp([0x000007a2000e90a1, 0x1, 0, 0]);

/// R^3 = 2^768 mod p
/// 0x100000b73002bb1e33795f671
const R3: Fp = Fp([0x002bb1e33795f671, 0x100000b73, 0, 0]);

/// 1 / 2 mod p
const TWO_INV: Fp = Fp::from_raw([
    0xffffffff7ffffe18,
    0xffffffffffffffff,
    0xffffffffffffffff,
    0x7fffffffffffffff,
]);

const ZETA: Fp = Fp::from_raw([
    0xc1396c28719501ee,
    0x9cf0497512f58995,
    0x6e64479eac3434e9,
    0x7ae96a2b657c0710,
]);

/// Generator of the t-order multiplicative subgroup.
/// Computed by exponentiating Self::MULTIPLICATIVE_GENERATOR by 2^s, where s is Self::S.
/// `0x0000000000000000000000000000000000000000000000000000000000000009`.
const DELTA: Fp = Fp([0x900002259u64, 0, 0, 0]);

/// Implementations of this trait MUST ensure that this is the generator used to derive Self::ROOT_OF_UNITY.
/// Derived from:
/// ```ignore
/// Zp(Zp(mul_generator)^t) where t = (modulus - 1 )/ 2
/// 115792089237316195423570985008687907853269984665640564039457584007908834671662
/// ```
const ROOT_OF_UNITY: Fp = Fp([
    0xfffffffdfffff85eu64,
    0xffffffffffffffffu64,
    0xffffffffffffffffu64,
    0xffffffffffffffffu64,
]);

/// Inverse of [`ROOT_OF_UNITY`].
const ROOT_OF_UNITY_INV: Fp = Fp([
    0xfffffffdfffff85eu64,
    0xffffffffffffffffu64,
    0xffffffffffffffffu64,
    0xffffffffffffffffu64,
]);

impl_binops_additive!(Fp, Fp);
impl_binops_multiplicative!(Fp, Fp);
field_common!(
    Fp,
    MODULUS,
    INV,
    MODULUS_STR,
    TWO_INV,
    ROOT_OF_UNITY_INV,
    DELTA,
    ZETA,
    R,
    R2,
    R3
);
impl_from_u64_u32!(Fp, R2);
field_arithmetic!(Fp, MODULUS, INV, dense);
impl_sum_prod!(Fp);

#[cfg(target_pointer_width = "64")]
field_bits!(Fp, MODULUS);
#[cfg(not(target_pointer_width = "64"))]
field_bits!(Fp, MODULUS, MODULUS_LIMBS_32);

impl Fp {
    pub const fn size() -> usize {
        32
    }
}

impl ff::Field for Fp {
    const ZERO: Self = Self::zero();
    const ONE: Self = Self::one();

    fn random(mut rng: impl RngCore) -> Self {
        Self::from_u512([
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ])
    }

    fn double(&self) -> Self {
        self.double()
    }

    #[inline(always)]
    fn square(&self) -> Self {
        self.square()
    }

    /// Computes the square root of this element, if it exists.
    fn sqrt(&self) -> CtOption<Self> {
        let tmp = self.pow([
            0xffffffffbfffff0c,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0x3fffffffffffffff,
        ]);

        CtOption::new(tmp, tmp.square().ct_eq(self))
    }

    /// Returns the multiplicative inverse of the
    /// element. If it is zero, the method fails.
    fn invert(&self) -> CtOption<Self> {
        self.invert()
    }

    fn pow_vartime<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let mut res = Self::one();
        let mut found_one = false;
        for e in exp.as_ref().iter().rev() {
            for i in (0..64).rev() {
                if found_one {
                    res = res.square();
                }

                if ((*e >> i) & 1) == 1 {
                    found_one = true;
                    res *= self;
                }
            }
        }
        res
    }

    fn sqrt_ratio(num: &Self, div: &Self) -> (Choice, Self) {
        ff::helpers::sqrt_ratio_generic(num, div)
    }
}

impl ff::PrimeField for Fp {
    type Repr = [u8; 32];

    const MODULUS: &'static str = MODULUS_STR;
    const MULTIPLICATIVE_GENERATOR: Self = MULTIPLICATIVE_GENERATOR;
    const TWO_INV: Self = TWO_INV;
    const ROOT_OF_UNITY: Self = ROOT_OF_UNITY;
    const ROOT_OF_UNITY_INV: Self = ROOT_OF_UNITY_INV;
    const DELTA: Self = DELTA;
    const NUM_BITS: u32 = 256;
    const CAPACITY: u32 = 255;
    const S: u32 = 1;

    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        let mut tmp = Fp([0, 0, 0, 0]);

        tmp.0[0] = u64::from_le_bytes(repr[0..8].try_into().unwrap());
        tmp.0[1] = u64::from_le_bytes(repr[8..16].try_into().unwrap());
        tmp.0[2] = u64::from_le_bytes(repr[16..24].try_into().unwrap());
        tmp.0[3] = u64::from_le_bytes(repr[24..32].try_into().unwrap());

        // Try to subtract the modulus
        let (_, borrow) = sbb(tmp.0[0], MODULUS.0[0], 0);
        let (_, borrow) = sbb(tmp.0[1], MODULUS.0[1], borrow);
        let (_, borrow) = sbb(tmp.0[2], MODULUS.0[2], borrow);
        let (_, borrow) = sbb(tmp.0[3], MODULUS.0[3], borrow);

        // If the element is smaller than MODULUS then the
        // subtraction will underflow, producing a borrow value
        // of 0xffff...ffff. Otherwise, it'll be zero.
        let is_some = (borrow as u8) & 1;

        // Convert to Montgomery form by computing
        // (a.R^0 * R^2) / R = a.R
        tmp *= &R2;

        CtOption::new(tmp, Choice::from(is_some))
    }

    fn to_repr(&self) -> Self::Repr {
        let tmp: [u64; 4] = (*self).into();
        let mut res = [0; 32];
        res[0..8].copy_from_slice(&tmp[0].to_le_bytes());
        res[8..16].copy_from_slice(&tmp[1].to_le_bytes());
        res[16..24].copy_from_slice(&tmp[2].to_le_bytes());
        res[24..32].copy_from_slice(&tmp[3].to_le_bytes());

        res
    }

    fn from_u128(v: u128) -> Self {
        Self::from_raw([v as u64, (v >> 64) as u64, 0, 0])
    }

    fn is_odd(&self) -> Choice {
        Choice::from(self.to_repr()[0] & 1)
    }
}

impl FromUniformBytes<64> for Fp {
    /// Converts a 512-bit little endian integer into
    /// an `Fp` by reducing by the modulus.
    fn from_uniform_bytes(bytes: &[u8; 64]) -> Self {
        Self::from_u512([
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
            u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
            u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
            u64::from_le_bytes(bytes[56..64].try_into().unwrap()),
        ])
    }
}

impl FromUniformBytes<48> for Fp {
    fn from_uniform_bytes(bytes: &[u8; 48]) -> Self {
        let repr = &mut [0u8; 64];
        (*repr)[0..48].copy_from_slice(&bytes[..48]);
        Fp::from_uniform_bytes(repr)
    }
}

impl WithSmallOrderMulGroup<3> for Fp {
    const ZETA: Self = ZETA;
}

extend_field_legendre!(Fp);

#[cfg(test)]
mod test {
    use super::*;
    crate::field_testing_suite!(Fp, "field_arithmetic");
    crate::field_testing_suite!(Fp, "conversion");
    crate::field_testing_suite!(Fp, "serialization");
    crate::field_testing_suite!(Fp, "quadratic_residue");
    crate::field_testing_suite!(Fp, "bits");
    crate::field_testing_suite!(Fp, "serialization_check");
    crate::field_testing_suite!(Fp, "constants", MODULUS_STR);
    crate::field_testing_suite!(Fp, "sqrt");
    crate::field_testing_suite!(Fp, "zeta");
    crate::field_testing_suite!(Fp, "from_uniform_bytes", 48, 64);
}
