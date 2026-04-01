use super::fp::{Fp, MODULUS_STR};
use crate::ff::{Field, FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use crate::ff_ext::Legendre;
use core::convert::TryInto;
use core::ops::{Add, Mul, Neg, Sub};
use rand::RngCore;
use std::cmp::Ordering;
use std::ops::MulAssign;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

#[cfg(feature = "derive_serde")]
use serde::{Deserialize, Serialize};

/// -ALPHA is a quadratic non-residue in Fp. Fp2 = Fp[X]/(X^2 + ALPHA)
/// We introduce the variable u such that u^2 = -ALPHA

/// U_SQUARE = -5
/// 0x24000000000024000130e0000d7f70e4a803ca76f439266f443f9a5cda8a6c7be4a7a5fe8fadffd6a2a7e8c30006b9459ffffcd2fffffffc
const U_SQUARE: Fp = Fp::from_raw([
    0x9ffffcd2fffffffc,
    0xa2a7e8c30006b945,
    0xe4a7a5fe8fadffd6,
    0x443f9a5cda8a6c7b,
    0xa803ca76f439266f,
    0x0130e0000d7f70e4,
    0x2400000000002400,
]);

const NEG_ONE: Fp2 = Fp2 {
    c0: super::fp::NEG_ONE,
    c1: Fp::ZERO,
};

/// An element of Fp2, represented by c0 + c1 * u.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "derive_serde", derive(Serialize, Deserialize))]
pub struct Fp2 {
    pub c0: Fp,
    pub c1: Fp,
}

/// `Fp2` elements are ordered lexicographically.
impl Ord for Fp2 {
    #[inline(always)]
    fn cmp(&self, other: &Fp2) -> Ordering {
        match self.c1.cmp(&other.c1) {
            Ordering::Greater => Ordering::Greater,
            Ordering::Less => Ordering::Less,
            Ordering::Equal => self.c0.cmp(&other.c0),
        }
    }
}

impl PartialOrd for Fp2 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Fp2) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl ConditionallySelectable for Fp2 {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Fp2 {
            c0: Fp::conditional_select(&a.c0, &b.c0, choice),
            c1: Fp::conditional_select(&a.c1, &b.c1, choice),
        }
    }
}

impl ConstantTimeEq for Fp2 {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.c0.ct_eq(&other.c0) & self.c1.ct_eq(&other.c1)
    }
}

impl Default for Fp2 {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl From<Fp2> for [u8; 112] {
    fn from(value: Fp2) -> [u8; 112] {
        value.to_bytes()
    }
}

impl<'a> From<&'a Fp2> for [u8; 112] {
    fn from(value: &'a Fp2) -> [u8; 112] {
        value.to_bytes()
    }
}

impl Neg for Fp2 {
    type Output = Fp2;

    #[inline]
    fn neg(self) -> Fp2 {
        -&self
    }
}

impl<'a> Neg for &'a Fp2 {
    type Output = Fp2;

    #[inline]
    fn neg(self) -> Fp2 {
        self.neg()
    }
}

impl<'a, 'b> Sub<&'b Fp2> for &'a Fp2 {
    type Output = Fp2;

    #[inline]
    fn sub(self, rhs: &'b Fp2) -> Fp2 {
        self.sub(rhs)
    }
}

impl<'a, 'b> Add<&'b Fp2> for &'a Fp2 {
    type Output = Fp2;

    #[inline]
    fn add(self, rhs: &'b Fp2) -> Fp2 {
        self.add(rhs)
    }
}

impl<'a, 'b> Mul<&'b Fp2> for &'a Fp2 {
    type Output = Fp2;

    #[inline]
    fn mul(self, rhs: &'b Fp2) -> Fp2 {
        self.mul(rhs)
    }
}

use crate::{
    impl_add_binop_specify_output, impl_binops_additive, impl_binops_additive_specify_output,
    impl_binops_multiplicative, impl_binops_multiplicative_mixed, impl_sub_binop_specify_output,
    impl_sum_prod,
};
impl_binops_additive!(Fp2, Fp2);
impl_binops_multiplicative!(Fp2, Fp2);
impl_sum_prod!(Fp2);

/// Size in bytes of a `Fp2` element.
const SIZE: usize = 112;
/// Size in bytes of a each coefficient of `Fp2`.
const COEF_SIZE: usize = 56;

impl Fp2 {
    /// Returns the zero element.
    #[inline]
    pub const fn zero() -> Fp2 {
        Fp2 {
            c0: Fp::zero(),
            c1: Fp::zero(),
        }
    }

    /// Returns the unit.
    #[inline]
    pub const fn one() -> Fp2 {
        Fp2 {
            c0: Fp::one(),
            c1: Fp::zero(),
        }
    }

    /// Given its `Fp` coefficients c0, c1. Returns the element of `Fp2`:  c0 + c1 * u.
    pub const fn new(c0: Fp, c1: Fp) -> Self {
        Fp2 { c0, c1 }
    }

    /// Returns the size in bytes of a `Fp2` element.
    pub const fn size() -> usize {
        SIZE
    }

    /// Attempts to convert a little-endian byte representation of
    /// a scalar into a `Fp`, failing if the input is not canonical.
    pub fn from_bytes(bytes: &[u8; SIZE]) -> CtOption<Fp2> {
        let c0 = Fp::from_bytes(bytes[0..COEF_SIZE].try_into().unwrap());
        let c1 = Fp::from_bytes(bytes[COEF_SIZE..SIZE].try_into().unwrap());
        CtOption::new(
            Fp2 {
                c0: c0.unwrap(),
                c1: c1.unwrap(),
            },
            c0.is_some() & c1.is_some(),
        )
    }

    /// Converts an element of `Fp` into a byte representation in
    /// little-endian byte order.
    pub fn to_bytes(self) -> [u8; SIZE] {
        let mut res = [0u8; SIZE];
        let c0_bytes = self.c0.to_bytes();
        let c1_bytes = self.c1.to_bytes();
        res[0..COEF_SIZE].copy_from_slice(&c0_bytes[..]);
        res[COEF_SIZE..SIZE].copy_from_slice(&c1_bytes[..]);
        res
    }

    // TODO: This is a naive method using 4 multiplications
    pub fn mul_assign(&mut self, other: &Self) {
        // r0 = s0 * s0 + U_SQUARE * s1 * o1
        // r1 = s0 * o1 - s1 * o0

        let t0 = self.c0 * other.c0;
        let t1 = self.c0 * other.c1;
        let t2 = self.c1 * other.c0;
        let t3 = self.c1 * other.c1;

        self.c0 = t0 + U_SQUARE * t3;
        self.c1 = t1 + t2
    }

    // TODO: This is a naive method using 3 multiplications
    pub fn square_assign(&mut self) {
        // r0 = s0^2 + U_SQUARE * s1^2
        // r1 = 2* s0s1

        let ab = self.c0 * self.c1;
        let a2 = self.c0 * self.c0;
        let b2 = self.c1 * self.c1;

        self.c1 = ab.double();
        self.c0 = a2 + U_SQUARE * b2;
    }

    pub fn double(&self) -> Self {
        Self {
            c0: self.c0.double(),
            c1: self.c1.double(),
        }
    }

    pub fn double_assign(&mut self) {
        self.c0 = self.c0.double();
        self.c1 = self.c1.double();
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            c0: self.c0.add(&other.c0),
            c1: self.c1.add(&other.c1),
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self {
            c0: self.c0.sub(&other.c0),
            c1: self.c1.sub(&other.c1),
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut t = *other;
        t.mul_assign(self);
        t
    }

    pub fn square(&self) -> Self {
        let mut t = *self;
        t.square_assign();
        t
    }

    pub fn neg(&self) -> Self {
        Self {
            c0: self.c0.neg(),
            c1: self.c1.neg(),
        }
    }

    // conjucate by negating c1
    pub fn conjugate(&mut self) {
        self.c1 = -self.c1;
    }

    pub fn frobenius_map(&mut self, power: usize) {
        //TODO Replace with constant time version if needed
        if power % 2 != 0 {
            self.conjugate()
        }
    }

    /// Multiply this element by cubic nonresidue: V_CUBE = 57/(u+3)
    pub fn mul_by_nonresidue(&mut self) {
        // (x + y * u) * 57/(u + 3)
        self.mul_assign(&super::fp6::V_CUBE)
    }

    pub fn invert(&self) -> CtOption<Self> {
        let mut t1 = self.c1;
        t1 = t1.square();
        t1 *= U_SQUARE;
        let mut t0 = self.c0;
        t0 = t0.square();
        //t0 = c0^2 - U_SQUARE c1^2
        t0 -= &t1;
        t0.invert().map(|t| {
            let mut tmp = Fp2 {
                c0: self.c0,
                c1: self.c1,
            };
            tmp.c0 *= &t;
            tmp.c1 *= &t;
            tmp.c1 = -tmp.c1;

            tmp
        })
    }

    /// Norm of Fp2 as extension field in u over Fp
    fn norm(&self) -> Fp {
        // norm = self * self.conjugate()
        let t0 = self.c0.square();
        let t1 = self.c1.square() * U_SQUARE;
        t0 - t1
    }
}

impl Legendre for Fp2 {
    fn legendre(&self) -> i64 {
        self.norm().legendre()
    }
}

impl Field for Fp2 {
    const ZERO: Self = Self::zero();
    const ONE: Self = Self::one();

    fn random(mut rng: impl RngCore) -> Self {
        Fp2 {
            c0: Fp::random(&mut rng),
            c1: Fp::random(&mut rng),
        }
    }

    fn is_zero(&self) -> Choice {
        self.c0.is_zero() & self.c1.is_zero()
    }

    fn square(&self) -> Self {
        self.square()
    }

    fn double(&self) -> Self {
        self.double()
    }

    fn sqrt(&self) -> CtOption<Self> {
        // Algorithm 10, https://eprint.iacr.org/2012/685.pdf

        // Aux elements. Described in PRECOMPUTATION of Algorithm 10.
        // As element of Fp2: E = 0 +  U *
        // 0x13e275a1fa6a13af7a82a3d83bc9e63a667c70cf991a36e603b21f15823a404a021848271d63f0875d232408689b4c6c67153f9701e19938
        const E: Fp2 = Fp2 {
            c0: Fp::ZERO,
            c1: Fp::from_raw([
                0x67153f9701e19938,
                0x5d232408689b4c6c,
                0x021848271d63f087,
                0x03b21f15823a404a,
                0x667c70cf991a36e6,
                0x7a82a3d83bc9e63a,
                0x13e275a1fa6a13af,
            ]),
        };

        // As element of Fp2: f = 5 + 0 * U
        // 0x5
        const F: Fp2 = Fp2 {
            c0: Fp::from_raw([0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            c1: Fp::ZERO,
        };

        // Algorithm (not constant time)
        let b = self.pow_vartime([
            // (p-1)/4 =
            // 0x900000000000900004c3800035fdc392a00f29dbd0e499bd10fe69736a29b1ef929e97fa3eb7ff5a8a9fa30c001ae5167ffff34c0000000
            0x67ffff34c0000000,
            0xa8a9fa30c001ae51,
            0xf929e97fa3eb7ff5,
            0xd10fe69736a29b1e,
            0x2a00f29dbd0e499b,
            0x004c3800035fdc39,
            0x0900000000000900,
        ]);

        let b_2 = b.square();
        let mut b_2_q = b_2;
        b_2_q.frobenius_map(1);

        let a0 = b_2_q * b_2;
        if a0 == NEG_ONE {
            CtOption::new(a0, Choice::from(0))
        } else {
            let mut x = b;
            x.frobenius_map(1);
            if x * b == Fp2::ONE {
                let x0 = (b_2 * self).c0.sqrt().unwrap();
                x.c0.mul_assign(x0);
                x.c1.mul_assign(x0);
                CtOption::new(x, Choice::from(1))
            } else {
                let x0 = (self * b_2 * F).sqrt().unwrap();
                x *= x0 * E;
                CtOption::new(x, Choice::from(1))
            }
        }
    }

    fn sqrt_ratio(num: &Self, div: &Self) -> (Choice, Self) {
        ff::helpers::sqrt_ratio_generic(num, div)
    }

    fn invert(&self) -> CtOption<Self> {
        self.invert()
    }
}

impl From<bool> for Fp2 {
    fn from(bit: bool) -> Fp2 {
        if bit {
            Fp2::ONE
        } else {
            Fp2::ZERO
        }
    }
}

impl From<u64> for Fp2 {
    fn from(val: u64) -> Self {
        Fp2 {
            c0: Fp::from(val),
            c1: Fp::zero(),
        }
    }
}

// This trait is only implemented to satisfy the requirement of CurveExt
impl PrimeField for Fp2 {
    type Repr = Fp2Bytes;

    const MODULUS: &'static str = MODULUS_STR;
    const MULTIPLICATIVE_GENERATOR: Self = Fp2 {
        c0: Fp::MULTIPLICATIVE_GENERATOR,
        c1: Fp::ZERO,
    };
    const NUM_BITS: u32 = 446;
    const CAPACITY: u32 = 445;
    const S: u32 = 0;

    // TODO: Check that we can just 0 this and forget.
    const ROOT_OF_UNITY: Self = Fp2::zero();
    const ROOT_OF_UNITY_INV: Self = Fp2 {
        c0: Fp::zero(),
        c1: Fp::zero(),
    };
    const DELTA: Self = Fp2 {
        c0: Fp::zero(),
        c1: Fp::zero(),
    };
    const TWO_INV: Self = Fp2 {
        c0: Fp::TWO_INV,
        c1: Fp::zero(),
    };

    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        let c0 = Fp::from_bytes(&repr.0[..COEF_SIZE].try_into().unwrap());
        let c1 = Fp::from_bytes(&repr.0[COEF_SIZE..].try_into().unwrap());
        // Disallow overflow representation
        CtOption::new(Fp2::new(c0.unwrap(), c1.unwrap()), Choice::from(1))
    }

    fn to_repr(&self) -> Self::Repr {
        Fp2Bytes(self.to_bytes())
    }

    fn is_odd(&self) -> Choice {
        self.c0.is_odd() | (self.c0.is_zero() & self.c1.is_odd())
    }
}

impl FromUniformBytes<64> for Fp2 {
    fn from_uniform_bytes(bytes: &[u8; 64]) -> Self {
        Self::new(Fp::from_uniform_bytes(bytes), Fp::zero())
    }
}
#[derive(Clone, Copy, Debug)]
/// Canonical little-endian representation of a `Fp2` element.
/// First half of the bytes represent `c0`, the second half represent `c1`.
pub struct Fp2Bytes([u8; SIZE]);

impl Default for Fp2Bytes {
    fn default() -> Self {
        Self([0u8; SIZE])
    }
}

impl AsMut<[u8]> for Fp2Bytes {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl AsRef<[u8]> for Fp2Bytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl crate::serde::SerdeObject for Fp2 {
    fn from_raw_bytes_unchecked(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), 112);
        let [c0, c1] = [0, 56].map(|i| Fp::from_raw_bytes_unchecked(&bytes[i..i + 56]));
        Self { c0, c1 }
    }
    fn from_raw_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != SIZE {
            return None;
        }
        let [c0, c1] = [0, COEF_SIZE].map(|i| Fp::from_raw_bytes(&bytes[i..i + COEF_SIZE]));
        c0.zip(c1).map(|(c0, c1)| Self { c0, c1 })
    }
    fn to_raw_bytes(&self) -> Vec<u8> {
        let mut res = Vec::with_capacity(SIZE);
        for limb in self.c0.0.iter().chain(self.c1.0.iter()) {
            res.extend_from_slice(&limb.to_le_bytes());
        }
        res
    }
    fn read_raw_unchecked<R: std::io::Read>(reader: &mut R) -> Self {
        let [c0, c1] = [(); 2].map(|_| Fp::read_raw_unchecked(reader));
        Self { c0, c1 }
    }
    fn read_raw<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let c0 = Fp::read_raw(reader)?;
        let c1 = Fp::read_raw(reader)?;
        Ok(Self { c0, c1 })
    }
    fn write_raw<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.c0.write_raw(writer)?;
        self.c1.write_raw(writer)
    }
}

impl WithSmallOrderMulGroup<3> for Fp2 {
    const ZETA: Self = Fp2 {
        // 0x24000000000024000130e0000d7f28e4a803ca76be3924a5f43f8cddf9a5c4781b50d5e1ff708dc8d9fa5d8a200bc4398ffff80f80000002
        c0: Fp::from_raw([
            0x8ffff80f80000002,
            0xd9fa5d8a200bc439,
            0x1b50d5e1ff708dc8,
            0xf43f8cddf9a5c478,
            0xa803ca76be3924a5,
            0x0130e0000d7f28e4,
            0x2400000000002400,
        ]),
        c1: Fp::zero(),
    };
}

#[cfg(test)]
mod test {
    use super::*;
    crate::field_testing_suite!(Fp2, "field_arithmetic");
    crate::field_testing_suite!(Fp2, "conversion");
    crate::field_testing_suite!(Fp2, "serialization");
    crate::field_testing_suite!(Fp2, "quadratic_residue");
    crate::field_testing_suite!(Fp2, "sqrt");
    crate::field_testing_suite!(Fp2, "zeta", Fp);
    // extension field-specific
    crate::field_testing_suite!(Fp2, "f2_tests", Fp);
    crate::field_testing_suite!(
        Fp2,
        "frobenius",
        // Frobenius endomorphism power parameter for extension field
        //  ϕ: E → E
        //  (x, y) ↦ (x^p, y^p)
        // p: modulus of base field (Here, Fp::MODULUS)
        [
            0x9ffffcd300000001,
            0xa2a7e8c30006b945,
            0xe4a7a5fe8fadffd6,
            0x443f9a5cda8a6c7b,
            0xa803ca76f439266f,
            0x0130e0000d7f70e4,
            0x2400000000002400,
        ]
    );

    #[test]
    fn test_fp2_squaring() {
        // u + 1
        let mut a = Fp2 {
            c0: Fp::one(),
            c1: Fp::one(),
        };
        // (u + 1) ^2 = 1 + u^2 + 2u = -4 + 2u
        a.square_assign();
        let minus_4 = -Fp::from(4u64);
        assert_eq!(
            a,
            Fp2 {
                c0: minus_4,
                c1: Fp::one() + Fp::one(),
            }
        );

        // u
        let mut a = Fp2 {
            c0: Fp::zero(),
            c1: Fp::one(),
        };
        // u^2
        a.square_assign();
        assert_eq!(
            a,
            Fp2 {
                c0: U_SQUARE,
                c1: Fp::zero(),
            }
        );
    }

    #[test]
    fn test_fp2_mul_nonresidue() {
        let mut rng = XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ]);
        let nqr = crate::pluto_eris::fields::fp6::V_CUBE;
        for _ in 0..1000 {
            let mut a = Fp2::random(&mut rng);
            let mut b = a;
            a.mul_by_nonresidue();
            b.mul_assign(&nqr);

            assert_eq!(a, b);
        }
    }
}
