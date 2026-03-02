use std::{
    hash::Hash,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::{field_common, ExtensionField, FFTField, Field, SimdField};

use ethnum::U256;
use rand::RngCore;
use serdes::ExpSerde;

use crate::{Goldilocks, GoldilocksExt2, Goldilocksx8};

/// Degree-2 extension of Goldilocks field with 8-element SIMD operations
/// Represents elements as a + bX where X^2 = 7
#[derive(Copy, Clone, Debug, Default, Hash, PartialEq, Eq, ExpSerde)]
pub struct GoldilocksExt2x8 {
    pub c0: Goldilocksx8, // constant term
    pub c1: Goldilocksx8, // coefficient of X
}

field_common!(GoldilocksExt2x8);

impl SimdField for GoldilocksExt2x8 {
    type Scalar = GoldilocksExt2;

    const PACK_SIZE: usize = Goldilocksx8::PACK_SIZE;

    #[inline]
    fn scale(&self, challenge: &Self::Scalar) -> Self {
        *self * *challenge
    }

    #[inline]
    fn pack_full(base: &Self::Scalar) -> Self {
        Self {
            c0: Goldilocksx8::pack_full(&base.v[0]),
            c1: Goldilocksx8::pack_full(&base.v[1]),
        }
    }

    #[inline]
    fn pack(base_vec: &[Self::Scalar]) -> Self {
        assert!(base_vec.len() == Self::PACK_SIZE);
        let mut v0s = vec![];
        let mut v1s = vec![];
        for scalar in base_vec {
            v0s.push(scalar.v[0]);
            v1s.push(scalar.v[1]);
        }
        Self {
            c0: Goldilocksx8::pack(&v0s),
            c1: Goldilocksx8::pack(&v1s),
        }
    }

    #[inline]
    fn unpack(&self) -> Vec<Self::Scalar> {
        let v0s = self.c0.unpack();
        let v1s = self.c1.unpack();
        v0s.into_iter()
            .zip(v1s)
            .map(|(v0, v1)| GoldilocksExt2 { v: [v0, v1] })
            .collect()
    }
}

impl From<Goldilocksx8> for GoldilocksExt2x8 {
    #[inline]
    fn from(x: Goldilocksx8) -> Self {
        Self {
            c0: x,
            c1: Goldilocksx8::ZERO,
        }
    }
}

impl ExtensionField for GoldilocksExt2x8 {
    type BaseField = Goldilocksx8;

    const DEGREE: usize = 2;

    const W: u32 = 7;

    const X: Self = Self {
        c0: Goldilocksx8::ZERO,
        c1: Goldilocksx8::ONE,
    };

    #[inline]
    fn mul_by_base_field(&self, base: &Self::BaseField) -> Self {
        Self {
            c0: self.c0 * base,
            c1: self.c1 * base,
        }
    }

    #[inline]
    fn add_by_base_field(&self, base: &Self::BaseField) -> Self {
        Self {
            c0: self.c0 + base,
            c1: self.c1,
        }
    }

    #[inline]
    fn mul_by_x(&self) -> Self {
        // (a + bX) * X = aX + bX^2
        // where X^2 = 7
        // = 7b + aX
        Self {
            c0: self.c1 * Goldilocksx8::pack_full(&Goldilocks { v: 7u64 }),
            c1: self.c0,
        }
    }

    #[inline]
    fn to_limbs(&self) -> Vec<Self::BaseField> {
        vec![self.c0, self.c1]
    }

    #[inline]
    fn from_limbs(limbs: &[Self::BaseField]) -> Self {
        assert!(limbs.len() >= 2);
        Self {
            c0: limbs[0],
            c1: limbs[1],
        }
    }
}

impl Mul<Goldilocksx8> for GoldilocksExt2x8 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Goldilocksx8) -> Self {
        self.mul_by_base_field(&rhs)
    }
}

impl From<GoldilocksExt2> for GoldilocksExt2x8 {
    #[inline]
    fn from(x: GoldilocksExt2) -> Self {
        Self {
            c0: Goldilocksx8::pack_full(&x.v[0]),
            c1: Goldilocksx8::pack_full(&x.v[1]),
        }
    }
}

impl Field for GoldilocksExt2x8 {
    const NAME: &'static str = "Goldilocks Extension Field 2x8";

    const SIZE: usize = 512 / 8 * 2;

    const FIELD_SIZE: usize = 64 * 2;

    const MODULUS: U256 = Goldilocks::MODULUS;

    const ZERO: Self = Self {
        c0: Goldilocksx8::ZERO,
        c1: Goldilocksx8::ZERO,
    };

    const ONE: Self = Self {
        c0: Goldilocksx8::ONE,
        c1: Goldilocksx8::ZERO,
    };

    const INV_2: Self = Self {
        c0: Goldilocksx8::INV_2,
        c1: Goldilocksx8::ZERO,
    };

    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn one() -> Self {
        Self::ONE
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    #[inline]
    fn random_unsafe(mut rng: impl RngCore) -> Self {
        Self {
            c0: Goldilocksx8::random_unsafe(&mut rng),
            c1: Goldilocksx8::random_unsafe(&mut rng),
        }
    }

    #[inline]
    fn random_bool(mut rng: impl RngCore) -> Self {
        Self {
            c0: Goldilocksx8::random_bool(&mut rng),
            c1: Goldilocksx8::ZERO,
        }
    }

    #[inline]
    fn as_u32_unchecked(&self) -> u32 {
        unimplemented!("self is a vector, cannot convert to u32")
    }

    #[inline]
    fn from_uniform_bytes(_bytes: &[u8]) -> Self {
        unimplemented!("vec Goldilocks: cannot convert from 32 bytes")
    }

    #[inline]
    fn square(&self) -> Self {
        square_internal(self)
    }

    fn inv(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        let compliment = Self {
            c0: -self.c0,
            c1: self.c1,
        };

        let w_base = Goldilocksx8::pack_full(&Goldilocks { v: Self::W as u64 });
        let normalize = (-self.c0.square() + self.c1.square() * w_base).inv()?;

        Some(compliment * normalize)
    }
}

impl Mul<GoldilocksExt2> for GoldilocksExt2x8 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: GoldilocksExt2) -> Self::Output {
        // (a0 + a1*x) * (b0 + b1*x) mod (x^2 - 7)
        // = a0*b0 + (a0*b1 + a1*b0)*x + a1*b1*x^2 mod (x^2 - 7)
        // = (a0*b0 + 7*a1*b1) + (a0*b1 + a1*b0)*x
        let seven = Goldilocks::from(7u32);
        Self {
            c0: self.c0 * rhs.v[0] + self.c1 * rhs.v[1] * seven,
            c1: self.c0 * rhs.v[1] + self.c1 * rhs.v[0],
        }
    }
}

impl Mul<Goldilocks> for GoldilocksExt2x8 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Goldilocks) -> Self {
        Self {
            c0: self.c0 * rhs,
            c1: self.c1 * rhs,
        }
    }
}

impl Add<Goldilocks> for GoldilocksExt2x8 {
    type Output = GoldilocksExt2x8;
    #[inline(always)]
    fn add(self, rhs: Goldilocks) -> Self::Output {
        GoldilocksExt2x8 {
            // Goldilocksx8 + Goldilocks
            c0: self.c0 + rhs,
            c1: self.c1,
        }
    }
}

impl Neg for GoldilocksExt2x8 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            c0: -self.c0,
            c1: -self.c1,
        }
    }
}

impl From<u32> for GoldilocksExt2x8 {
    #[inline]
    fn from(value: u32) -> Self {
        Self {
            c0: Goldilocksx8::from(value),
            c1: Goldilocksx8::ZERO,
        }
    }
}

#[inline(always)]
fn add_internal(a: &GoldilocksExt2x8, b: &GoldilocksExt2x8) -> GoldilocksExt2x8 {
    GoldilocksExt2x8 {
        c0: a.c0 + b.c0,
        c1: a.c1 + b.c1,
    }
}

#[inline(always)]
fn sub_internal(a: &GoldilocksExt2x8, b: &GoldilocksExt2x8) -> GoldilocksExt2x8 {
    GoldilocksExt2x8 {
        c0: a.c0 - b.c0,
        c1: a.c1 - b.c1,
    }
}

#[inline(always)]
fn mul_internal(a: &GoldilocksExt2x8, b: &GoldilocksExt2x8) -> GoldilocksExt2x8 {
    // (a + bX)(c + dX) = ac + (ad + bc)X + bdX^2
    // where X^2 = 7
    // = (ac + 7bd) + (ad + bc)X
    let ac = a.c0 * b.c0;
    let bd = a.c1 * b.c1;
    let ad = a.c0 * b.c1;
    let bc = a.c1 * b.c0;
    GoldilocksExt2x8 {
        c0: ac + bd * Goldilocksx8::from(7u64),
        c1: ad + bc,
    }
}

#[inline(always)]
fn square_internal(a: &GoldilocksExt2x8) -> GoldilocksExt2x8 {
    let r0 = a.c0.square() + a.c1.square() * Goldilocksx8::pack_full(&7u64.into());
    let r1 = a.c0 * a.c1.double();
    GoldilocksExt2x8 { c0: r0, c1: r1 }
}

impl Ord for GoldilocksExt2x8 {
    #[inline(always)]
    fn cmp(&self, _: &Self) -> std::cmp::Ordering {
        unimplemented!("Ord for GoldilocksExt2x8 is not supported")
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for GoldilocksExt2x8 {
    #[inline(always)]
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        unimplemented!("PartialOrd for GoldilocksExt2x8 is not supported")
    }
}

impl FFTField for GoldilocksExt2x8 {
    const TWO_ADICITY: usize = 33;

    #[inline(always)]
    fn root_of_unity() -> Self {
        let var = GoldilocksExt2 {
            v: [
                Goldilocks::ZERO,
                Goldilocks {
                    v: 0xd95051a31cf4a6ef,
                },
            ],
        };
        Self::pack_full(&var)
    }
}

impl Add<Goldilocksx8> for GoldilocksExt2x8 {
    type Output = GoldilocksExt2x8;

    #[inline(always)]
    fn add(self, rhs: Goldilocksx8) -> Self::Output {
        self.add_by_base_field(&rhs)
    }
}
