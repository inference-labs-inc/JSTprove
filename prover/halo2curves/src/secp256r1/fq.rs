use crate::arithmetic::{adc, bigint_geq, mac, macx, sbb};
use crate::extend_field_legendre;
use crate::ff::{FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
use rand::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

/// This represents an element of $\mathbb{F}_q$ where
///
/// `q = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551`
///
/// is the scalar field of the secp256r1 curve.
// The internal representation of this type is four 64-bit unsigned
// integers in little-endian order. `Fq` values are always in
// Montgomery form; i.e., Fq(a) = aR mod q, with R = 2^256.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fq(pub(crate) [u64; 4]);

#[cfg(feature = "derive_serde")]
crate::serialize_deserialize_32_byte_primefield!(Fq);

/// Constant representing the modulus
/// q = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551
const MODULUS: Fq = Fq([
    0xf3b9cac2fc632551,
    0xbce6faada7179e84,
    0xffffffffffffffff,
    0xffffffff00000000,
]);

/// The modulus as u32 limbs.
#[cfg(not(target_pointer_width = "64"))]
const MODULUS_LIMBS_32: [u32; 8] = [
    0xfc63_2551,
    0xf3b9_cac2,
    0xa717_9e84,
    0xbce6_faad,
    0xffff_ffff,
    0xffff_ffff,
    0x0000_0000,
    0xffff_ffff,
];

///Constant representing the modulus as static str
const MODULUS_STR: &str = "0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551";

/// INV = -(q^{-1} mod 2^64) mod 2^64
const INV: u64 = 0xccd1c8aaee00bc4f;

/// R = 2^256 mod q
/// 0xffffffff00000000000000004319055258e8617b0c46353d039cdaaf
const R: Fq = Fq([
    0x0c46353d039cdaaf,
    0x4319055258e8617b,
    0x0000000000000000,
    0xffffffff,
]);

/// R^2 = 2^512 mod q
/// 0x66e12d94f3d956202845b2392b6bec594699799c49bd6fa683244c95be79eea2
const R2: Fq = Fq([
    0x83244c95be79eea2,
    0x4699799c49bd6fa6,
    0x2845b2392b6bec59,
    0x66e12d94f3d95620,
]);

/// R^3 = 2^768 mod q
/// 0x503a54e76407be652543b9246ba5e93f111f28ae0c0555c9ac8ebec90b65a624
const R3: Fq = Fq([
    0xac8ebec90b65a624,
    0x111f28ae0c0555c9,
    0x2543b9246ba5e93f,
    0x503a54e76407be65,
]);

/// `GENERATOR = 7 mod r` is a generator of the `q - 1` order multiplicative
/// subgroup, or in other words a primitive root of the field.
/// It's derived with SageMath with: `GF(MODULUS).primitive_element()`.
const GENERATOR: Fq = Fq::from_raw([0x07, 0x00, 0x00, 0x00]);

/// GENERATOR^t where t * 2^s + 1 = r with t odd. In other words, this is a 2^s root of unity.
/// `ffc97f062a770992ba807ace842a3dfc1546cad004378daf0592d7fbb41e6602`
const ROOT_OF_UNITY: Fq = Fq::from_raw([
    0x0592d7fbb41e6602,
    0x1546cad004378daf,
    0xba807ace842a3dfc,
    0xffc97f062a770992,
]);

/// 1 / ROOT_OF_UNITY mod q
/// `a0a66a5562d46f2ac645fa0458131caee3ac117c794c4137379c7f0657c73764`
const ROOT_OF_UNITY_INV: Fq = Fq::from_raw([
    0x379c7f0657c73764,
    0xe3ac117c794c4137,
    0xc645fa0458131cae,
    0xa0a66a5562d46f2a,
]);

/// 1 / 2 mod q
const TWO_INV: Fq = Fq::from_raw([
    0x79dce5617e3192a9,
    0xde737d56d38bcf42,
    0x7fffffffffffffff,
    0x7fffffff80000000,
]);

const ZETA: Fq = Fq::from_raw([
    0x7cbf87ff12884e21,
    0x9405335ce9c83e1d,
    0x4e786d0777fd6aef,
    0x52891d43d946a035,
]);

/// Generator of the t-order multiplicative subgroup.
/// Computed by exponentiating Self::MULTIPLICATIVE_GENERATOR by 2^s, where s is Self::S.
const DELTA: Fq = Fq::from_raw([0x1e39a5057d81, 0, 0, 0]);

use crate::{
    field_arithmetic, field_bits, field_common, field_specific, impl_add_binop_specify_output,
    impl_binops_additive, impl_binops_additive_specify_output, impl_binops_multiplicative,
    impl_binops_multiplicative_mixed, impl_from_u64_u32, impl_sub_binop_specify_output,
    impl_sum_prod,
};
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
        // 7fffffff800000007fffffffffffffffde737d56d38bcf4279dce5617e3192a
        let tm1d2 = [
            0x279dce5617e3192a,
            0xfde737d56d38bcf4,
            0x07ffffffffffffff,
            0x7fffffff8000000,
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
    const S: u32 = 4;

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
    crate::field_testing_suite!(Fq, "serialization_check");
    crate::field_testing_suite!(Fq, "constants", MODULUS_STR);
    crate::field_testing_suite!(Fq, "sqrt");
    crate::field_testing_suite!(Fq, "zeta");
    crate::field_testing_suite!(Fq, "from_uniform_bytes", 64);
}
