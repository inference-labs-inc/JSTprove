use super::fq::{Fq, NEGATIVE_ONE};
use crate::ff::{Field, FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use crate::ff_ext::Legendre;
use core::convert::TryInto;
use core::ops::{Add, Mul, Neg, Sub};
use rand::RngCore;
use std::cmp::Ordering;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

#[cfg(feature = "derive_serde")]
use serde::{Deserialize, Serialize};

/// An element of Fq2, represented by c0 + c1 * u; where u^2 = -1.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "derive_serde", derive(Serialize, Deserialize))]
pub struct Fq2 {
    pub c0: Fq,
    pub c1: Fq,
}

/// `Fq2` elements are ordered lexicographically.
impl Ord for Fq2 {
    #[inline(always)]
    fn cmp(&self, other: &Fq2) -> Ordering {
        match self.c1.cmp(&other.c1) {
            Ordering::Greater => Ordering::Greater,
            Ordering::Less => Ordering::Less,
            Ordering::Equal => self.c0.cmp(&other.c0),
        }
    }
}

impl PartialOrd for Fq2 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Fq2) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl ConditionallySelectable for Fq2 {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Fq2 {
            c0: Fq::conditional_select(&a.c0, &b.c0, choice),
            c1: Fq::conditional_select(&a.c1, &b.c1, choice),
        }
    }
}

impl ConstantTimeEq for Fq2 {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.c0.ct_eq(&other.c0) & self.c1.ct_eq(&other.c1)
    }
}

impl Default for Fq2 {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl From<Fq2> for [u8; 64] {
    fn from(value: Fq2) -> [u8; 64] {
        value.to_bytes()
    }
}

impl<'a> From<&'a Fq2> for [u8; 64] {
    fn from(value: &'a Fq2) -> [u8; 64] {
        value.to_bytes()
    }
}

impl Neg for Fq2 {
    type Output = Fq2;

    #[inline]
    fn neg(self) -> Fq2 {
        -&self
    }
}

impl<'a> Neg for &'a Fq2 {
    type Output = Fq2;

    #[inline]
    fn neg(self) -> Fq2 {
        self.neg()
    }
}

impl<'a, 'b> Sub<&'b Fq2> for &'a Fq2 {
    type Output = Fq2;

    #[inline]
    fn sub(self, rhs: &'b Fq2) -> Fq2 {
        self.sub(rhs)
    }
}

impl<'a, 'b> Add<&'b Fq2> for &'a Fq2 {
    type Output = Fq2;

    #[inline]
    fn add(self, rhs: &'b Fq2) -> Fq2 {
        self.add(rhs)
    }
}

impl<'a, 'b> Mul<&'b Fq2> for &'a Fq2 {
    type Output = Fq2;

    #[inline]
    fn mul(self, rhs: &'b Fq2) -> Fq2 {
        self.mul(rhs)
    }
}

use crate::{
    impl_add_binop_specify_output, impl_binops_additive, impl_binops_additive_specify_output,
    impl_binops_multiplicative, impl_binops_multiplicative_mixed, impl_sub_binop_specify_output,
    impl_sum_prod,
};
impl_binops_additive!(Fq2, Fq2);
impl_binops_multiplicative!(Fq2, Fq2);
impl_sum_prod!(Fq2);

impl Fq2 {
    #[inline]
    pub const fn zero() -> Fq2 {
        Fq2 {
            c0: Fq::zero(),
            c1: Fq::zero(),
        }
    }

    #[inline]
    pub const fn one() -> Fq2 {
        Fq2 {
            c0: Fq::one(),
            c1: Fq::zero(),
        }
    }

    pub const fn new(c0: Fq, c1: Fq) -> Self {
        Fq2 { c0, c1 }
    }

    pub const fn size() -> usize {
        64
    }
    /// Attempts to convert a little-endian byte representation of
    /// a scalar into a `Fq`, failing if the input is not canonical.
    pub fn from_bytes(bytes: &[u8; 64]) -> CtOption<Fq2> {
        let c0 = Fq::from_bytes(bytes[0..32].try_into().unwrap());
        let c1 = Fq::from_bytes(bytes[32..64].try_into().unwrap());
        CtOption::new(
            Fq2 {
                c0: c0.unwrap(),
                c1: c1.unwrap(),
            },
            c0.is_some() & c1.is_some(),
        )
    }

    /// Converts an element of `Fq` into a byte representation in
    /// little-endian byte order.
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut res = [0u8; 64];
        let c0_bytes = self.c0.to_bytes();
        let c1_bytes = self.c1.to_bytes();
        res[0..32].copy_from_slice(&c0_bytes[..]);
        res[32..64].copy_from_slice(&c1_bytes[..]);
        res
    }

    pub fn mul_assign(&mut self, other: &Self) {
        let mut t0 = self.c0 + self.c1;
        let mut t1 = self.c0 * other.c0;
        let t2 = self.c1 * other.c1;

        self.c0 = t1 - t2;
        self.c1 = other.c0 + other.c1;
        t1 += t2;
        t0 *= self.c1;
        self.c1 = t0 - t1;
    }

    pub fn square_assign(&mut self) {
        let ab = self.c0 * self.c1;
        let c0c1 = self.c0 + self.c1;
        let mut c0 = -self.c1;
        c0 += self.c0;
        c0 *= c0c1;
        c0 -= ab;
        self.c1 = ab.double();
        self.c0 = c0 + ab;
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

    // conjugate by negating c1
    pub fn conjugate(&mut self) {
        self.c1 = -self.c1;
    }

    pub fn frobenius_map(&mut self, power: usize) {
        if power % 2 != 0 {
            self.conjugate()
        }
    }

    /// Multiply this element by quadratic nonresidue 9 + u.
    pub fn mul_by_nonresidue(&mut self) {
        // (xu+y)(u+9) = (9x+y)u+(9y-x)
        let t0 = self.c0;
        let t1 = self.c1;

        // 8*x*i + 8*y
        self.double_assign();
        self.double_assign();
        self.double_assign();

        // 9*y
        self.c0 += &t0;
        // (9*y - x)
        self.c0 -= &t1;

        // (9*x)u
        self.c1 += &t1;
        // (9*x + y)
        self.c1 += &t0;
    }

    pub fn invert(&self) -> CtOption<Self> {
        let mut t1 = self.c1;
        t1 = t1.square();
        let mut t0 = self.c0;
        t0 = t0.square();
        t0 += &t1;
        t0.invert().map(|t| {
            let mut tmp = Fq2 {
                c0: self.c0,
                c1: self.c1,
            };
            tmp.c0 *= &t;
            tmp.c1 *= &t;
            tmp.c1 = -tmp.c1;

            tmp
        })
    }

    /// Norm of Fq2 as extension field in i over Fq
    #[inline]
    fn norm(&self) -> Fq {
        let mut t0 = self.c0;
        let mut t1 = self.c1;
        t0 = t0.square();
        t1 = t1.square();
        t1 + t0
    }
}

impl Legendre for Fq2 {
    fn legendre(&self) -> i64 {
        self.norm().legendre()
    }
}

impl Field for Fq2 {
    const ZERO: Self = Self::zero();
    const ONE: Self = Self::one();

    fn random(mut rng: impl RngCore) -> Self {
        Fq2 {
            c0: Fq::random(&mut rng),
            c1: Fq::random(&mut rng),
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
        // Algorithm 9, https://eprint.iacr.org/2012/685.pdf

        if self.is_zero().into() {
            CtOption::new(Self::ZERO, Choice::from(1))
        } else {
            // a1 = self^((q - 3) / 4)
            // 0xc19139cb84c680a6e14116da060561765e05aa45a1c72a34f082305b61f3f51
            let u: [u64; 4] = [
                0x4f082305b61f3f51,
                0x65e05aa45a1c72a3,
                0x6e14116da0605617,
                0x0c19139cb84c680a,
            ];
            let mut a1 = self.pow(u);
            let mut alpha = a1;

            alpha.square_assign();
            alpha.mul_assign(self);
            let mut a0 = alpha;
            a0.frobenius_map(1);
            a0.mul_assign(&alpha);

            let neg1 = Fq2 {
                c0: NEGATIVE_ONE,
                c1: Fq::zero(),
            };

            if a0 == neg1 {
                CtOption::new(a0, Choice::from(0))
            } else {
                a1.mul_assign(self);

                if alpha == neg1 {
                    a1.mul_assign(&Fq2 {
                        c0: Fq::zero(),
                        c1: Fq::one(),
                    });
                } else {
                    alpha += &Fq2::ONE;
                    // alpha = alpha^((q - 1) / 2)
                    // 0x183227397098d014dc2822db40c0ac2ecbc0b548b438e5469e10460b6c3e7ea3
                    let u: [u64; 4] = [
                        0x9e10460b6c3e7ea3,
                        0xcbc0b548b438e546,
                        0xdc2822db40c0ac2e,
                        0x183227397098d014,
                    ];
                    alpha = alpha.pow(u);
                    a1.mul_assign(&alpha);
                }
                CtOption::new(a1, Choice::from(1))
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

impl From<bool> for Fq2 {
    fn from(bit: bool) -> Fq2 {
        if bit {
            Fq2::ONE
        } else {
            Fq2::ZERO
        }
    }
}

impl From<u64> for Fq2 {
    fn from(val: u64) -> Self {
        Fq2 {
            c0: Fq::from(val),
            c1: Fq::zero(),
        }
    }
}

// This trait is only implemented to satisfy the requirement of CurveExt
impl PrimeField for Fq2 {
    type Repr = Fq2Bytes;

    const MODULUS: &'static str =
        "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47";
    const MULTIPLICATIVE_GENERATOR: Self = Fq2 {
        c0: Fq::from_raw([0x03, 0x0, 0x0, 0x0]),
        c1: Fq::ZERO,
    };
    const NUM_BITS: u32 = 254;
    const CAPACITY: u32 = 253;
    const S: u32 = 0;
    // TODO: Check that we can just 0 this and forget.
    const ROOT_OF_UNITY: Self = Fq2::zero();
    const ROOT_OF_UNITY_INV: Self = Fq2 {
        c0: Fq::zero(),
        c1: Fq::zero(),
    };
    const DELTA: Self = Fq2 {
        c0: Fq::zero(),
        c1: Fq::zero(),
    };
    const TWO_INV: Self = Fq2 {
        c0: Fq::from_raw([
            0x9e10460b6c3e7ea4,
            0xcbc0b548b438e546,
            0xdc2822db40c0ac2e,
            0x183227397098d014,
        ]),
        c1: Fq([0, 0, 0, 0]),
    };

    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        let c0 = Fq::from_bytes(&repr.0[..32].try_into().unwrap());
        let c1 = Fq::from_bytes(&repr.0[32..].try_into().unwrap());
        // Disallow overflow representation
        CtOption::new(Fq2::new(c0.unwrap(), c1.unwrap()), Choice::from(1))
    }

    fn to_repr(&self) -> Self::Repr {
        Fq2Bytes(self.to_bytes())
    }

    fn is_odd(&self) -> Choice {
        self.c0.is_odd() | (self.c0.is_zero() & self.c1.is_odd())
    }
}

impl FromUniformBytes<96> for Fq2 {
    fn from_uniform_bytes(bytes: &[u8; 96]) -> Self {
        let c0: [u8; 48] = bytes[..48].try_into().unwrap();
        let c1: [u8; 48] = bytes[48..].try_into().unwrap();
        Self::new(Fq::from_uniform_bytes(&c1), Fq::from_uniform_bytes(&c0))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Fq2Bytes([u8; 64]);

impl Default for Fq2Bytes {
    fn default() -> Self {
        Self([0u8; 64])
    }
}

impl AsMut<[u8]> for Fq2Bytes {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl AsRef<[u8]> for Fq2Bytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl crate::serde::SerdeObject for Fq2 {
    fn from_raw_bytes_unchecked(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), 64);
        let [c0, c1] = [0, 32].map(|i| Fq::from_raw_bytes_unchecked(&bytes[i..i + 32]));
        Self { c0, c1 }
    }
    fn from_raw_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 64 {
            return None;
        }
        let [c0, c1] = [0, 32].map(|i| Fq::from_raw_bytes(&bytes[i..i + 32]));
        c0.zip(c1).map(|(c0, c1)| Self { c0, c1 })
    }
    fn to_raw_bytes(&self) -> Vec<u8> {
        let mut res = Vec::with_capacity(64);
        for limb in self.c0.0.iter().chain(self.c1.0.iter()) {
            res.extend_from_slice(&limb.to_le_bytes());
        }
        res
    }
    fn read_raw_unchecked<R: std::io::Read>(reader: &mut R) -> Self {
        let [c0, c1] = [(); 2].map(|_| Fq::read_raw_unchecked(reader));
        Self { c0, c1 }
    }
    fn read_raw<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let c0 = Fq::read_raw(reader)?;
        let c1 = Fq::read_raw(reader)?;
        Ok(Self { c0, c1 })
    }
    fn write_raw<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.c0.write_raw(writer)?;
        self.c1.write_raw(writer)
    }
}

impl WithSmallOrderMulGroup<3> for Fq2 {
    // Fq::ZETA ^2
    const ZETA: Self = Fq2 {
        c0: Fq::from_raw([
            0x5763473177fffffe,
            0xd4f263f1acdb5c4f,
            0x59e26bcea0d48bac,
            0x0000000000000000,
        ]),
        c1: Fq::zero(),
    };
}

#[cfg(test)]
mod test {
    use super::*;
    crate::field_testing_suite!(Fq2, "field_arithmetic");
    crate::field_testing_suite!(Fq2, "conversion");
    crate::field_testing_suite!(Fq2, "serialization");
    crate::field_testing_suite!(Fq2, "quadratic_residue");
    crate::field_testing_suite!(Fq2, "sqrt");
    crate::field_testing_suite!(Fq2, "zeta", Fq);
    // extension field-specific
    crate::field_testing_suite!(Fq2, "f2_tests", Fq);
    crate::field_testing_suite!(
        Fq2,
        "frobenius",
        // Frobenius endomorphism power parameter for extension field
        //  ϕ: E → E
        //  (x, y) ↦ (x^p, y^p)
        // p: modulus of base field (Here, Fq::MODULUS)
        [
            0x3c208c16d87cfd47,
            0x97816a916871ca8d,
            0xb85045b68181585d,
            0x30644e72e131a029,
        ]
    );

    #[test]
    fn test_fq2_squaring() {
        let mut a = Fq2 {
            c0: Fq::one(),
            c1: Fq::one(),
        }; // u + 1
        a.square_assign();
        assert_eq!(
            a,
            Fq2 {
                c0: Fq::zero(),
                c1: Fq::one() + Fq::one(),
            }
        ); // 2u

        let mut a = Fq2 {
            c0: Fq::zero(),
            c1: Fq::one(),
        }; // u
        a.square_assign();
        assert_eq!(a, {
            let neg1 = -Fq::one();
            Fq2 {
                c0: neg1,
                c1: Fq::zero(),
            }
        }); // -1
    }

    #[test]
    fn test_fq2_mul_nonresidue() {
        let mut rng = XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ]);
        let nine = Fq::one().double().double().double() + Fq::one();
        let nqr = Fq2 {
            c0: nine,
            c1: Fq::one(),
        };

        for _ in 0..1000 {
            let mut a = Fq2::random(&mut rng);
            let mut b = a;
            a.mul_by_nonresidue();
            b.mul_assign(&nqr);

            assert_eq!(a, b);
        }
    }
}
