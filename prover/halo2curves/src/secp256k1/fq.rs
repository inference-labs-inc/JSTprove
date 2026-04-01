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

/// This represents an element of $\mathbb{F}_q$ where
///
/// `q = 0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141`
///
/// is the scalar field of the secp256k1 curve.
// The internal representation of this type is four 64-bit unsigned
// integers in little-endian order. `Fq` values are always in
// Montgomery form; i.e., Fq(a) = aR mod q, with R = 2^256.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fq(pub(crate) [u64; 4]);

#[cfg(feature = "derive_serde")]
crate::serialize_deserialize_32_byte_primefield!(Fq);

/// Constant representing the modulus
/// q = 0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141
const MODULUS: Fq = Fq([
    0xbfd25e8cd0364141,
    0xbaaedce6af48a03b,
    0xfffffffffffffffe,
    0xffffffffffffffff,
]);

/// The modulus as u32 limbs.
#[cfg(not(target_pointer_width = "64"))]
const MODULUS_LIMBS_32: [u32; 8] = [
    0xd036_4141,
    0xbfd2_5e8c,
    0xaf48_a03b,
    0xbaae_dce6,
    0xffff_fffe,
    0xffff_ffff,
    0xffff_ffff,
    0xffff_ffff,
];

///Constant representing the modulus as static str
const MODULUS_STR: &str = "0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141";

/// INV = -(q^{-1} mod 2^64) mod 2^64
const INV: u64 = 0x4b0dff665588b13f;

/// R = 2^256 mod q
/// 0x14551231950b75fc4402da1732fc9bebf
const R: Fq = Fq([0x402da1732fc9bebf, 0x4551231950b75fc4, 0x1, 0]);

/// R^2 = 2^512 mod q
/// 0x9d671cd581c69bc5e697f5e45bcd07c6741496c20e7cf878896cf21467d7d140
const R2: Fq = Fq([
    0x896cf21467d7d140,
    0x741496c20e7cf878,
    0xe697f5e45bcd07c6,
    0x9d671cd581c69bc5,
]);

/// R^3 = 2^768 mod q
/// 0x555d800c18ef116db1b31347f1d0b2da0017648444d4322c7bc0cfe0e9ff41ed
const R3: Fq = Fq([
    0x7bc0cfe0e9ff41ed,
    0x0017648444d4322c,
    0xb1b31347f1d0b2da,
    0x555d800c18ef116d,
]);

/// `GENERATOR = 7 mod r` is a generator of the `q - 1` order multiplicative
/// subgroup, or in other words a primitive root of the field.
/// It's derived with SageMath with: `GF(MODULUS).primitive_element()`.
const GENERATOR: Fq = Fq::from_raw([0x07, 0x00, 0x00, 0x00]);

/// GENERATOR^t where t * 2^s + 1 = r with t odd. In other words, this is a 2^s root of unity.
/// `0xc1dc060e7a91986df9879a3fbc483a898bdeab680756045992f4b5402b052f2`
const ROOT_OF_UNITY: Fq = Fq::from_raw([
    0x992f4b5402b052f2,
    0x98bdeab680756045,
    0xdf9879a3fbc483a8,
    0xc1dc060e7a91986,
]);

/// 1 / ROOT_OF_UNITY mod q
const ROOT_OF_UNITY_INV: Fq = Fq::from_raw([
    0xb6fb30a0884f0d1c,
    0x77a275910aa413c3,
    0xefc7b0c75b8cbb72,
    0xfd3ae181f12d7096,
]);

/// 1 / 2 mod q
const TWO_INV: Fq = Fq::from_raw([
    0xdfe92f46681b20a1,
    0x5d576e7357a4501d,
    0xffffffffffffffff,
    0x7fffffffffffffff,
]);

const ZETA: Fq = Fq::from_raw([
    0xdf02967c1b23bd72,
    0x122e22ea20816678,
    0xa5261c028812645a,
    0x5363ad4cc05c30e0,
]);

/// Generator of the t-order multiplicative subgroup.
/// Computed by exponentiating Self::MULTIPLICATIVE_GENERATOR by 2^s, where s is Self::S.
/// `0x0000000000000000000cbc21fe4561c8d63b78e780e1341e199417c8c0bb7601`
const DELTA: Fq = Fq([
    0xd91b33d24319d9e8,
    0xb81c6596ff5d6740,
    0xa463969ca14c51c1,
    0x1900960de4b7929c,
]);

impl_binops_additive!(Fq, Fq);
impl_binops_multiplicative!(Fq, Fq);
field_common!(
    Fq,
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
impl_from_u64_u32!(Fq, R2);
field_arithmetic!(Fq, MODULUS, INV, dense);
impl_sum_prod!(Fq);

#[cfg(target_pointer_width = "64")]
field_bits!(Fq, MODULUS);
#[cfg(not(target_pointer_width = "64"))]
field_bits!(Fq, MODULUS, MODULUS_LIMBS_32);

impl Fq {
    pub const fn size() -> usize {
        32
    }
}

impl ff::Field for Fq {
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

    fn sqrt(&self) -> CtOption<Self> {
        let tm1d2 = [
            0x777fa4bd19a06c82,
            0xfd755db9cd5e9140,
            0xffffffffffffffff,
            0x01ffffffffffffff,
        ];

        ff::helpers::sqrt_tonelli_shanks(self, tm1d2)
    }

    fn sqrt_ratio(num: &Self, div: &Self) -> (Choice, Self) {
        ff::helpers::sqrt_ratio_generic(num, div)
    }
}

impl ff::PrimeField for Fq {
    type Repr = [u8; 32];

    const NUM_BITS: u32 = 256;
    const CAPACITY: u32 = 255;
    const MODULUS: &'static str = MODULUS_STR;
    const MULTIPLICATIVE_GENERATOR: Self = GENERATOR;
    const ROOT_OF_UNITY: Self = ROOT_OF_UNITY;
    const ROOT_OF_UNITY_INV: Self = ROOT_OF_UNITY_INV;
    const TWO_INV: Self = TWO_INV;
    const DELTA: Self = DELTA;
    const S: u32 = 6;

    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        let mut tmp = Fq([0, 0, 0, 0]);

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

    fn is_odd(&self) -> Choice {
        Choice::from(self.to_repr()[0] & 1)
    }
}

impl FromUniformBytes<64> for Fq {
    /// Converts a 512-bit little endian integer into
    /// an `Fq` by reducing by the modulus.
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

impl FromUniformBytes<48> for Fq {
    fn from_uniform_bytes(bytes: &[u8; 48]) -> Self {
        let repr = &mut [0u8; 64];
        (*repr)[0..48].copy_from_slice(&bytes[..48]);
        Fq::from_uniform_bytes(repr)
    }
}

impl WithSmallOrderMulGroup<3> for Fq {
    const ZETA: Self = ZETA;
}

extend_field_legendre!(Fq);

#[cfg(test)]
mod test {
    use super::*;
    crate::field_testing_suite!(Fq, "field_arithmetic");
    crate::field_testing_suite!(Fq, "conversion");
    crate::field_testing_suite!(Fq, "serialization");
    crate::field_testing_suite!(Fq, "quadratic_residue");
    crate::field_testing_suite!(Fq, "bits");
    crate::field_testing_suite!(Fq, "serialization_check");
    crate::field_testing_suite!(Fq, "constants", MODULUS_STR);
    crate::field_testing_suite!(Fq, "sqrt");
    crate::field_testing_suite!(Fq, "zeta");
    crate::field_testing_suite!(Fq, "from_uniform_bytes", 48, 64);
}
