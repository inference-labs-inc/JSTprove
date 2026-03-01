use std::{
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::{field_common, ExtensionField, FFTField, Field, SimdField};
use ethnum::U256;
use serdes::ExpSerde;

use crate::{babybear::BabyBear, BabyBearExt3, BabyBearx16};

#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq, ExpSerde)]
pub struct BabyBearExt3x16 {
    pub v: [BabyBearx16; 3],
}

field_common!(BabyBearExt3x16);

impl SimdField for BabyBearExt3x16 {
    type Scalar = BabyBearExt3;

    const PACK_SIZE: usize = BabyBearx16::PACK_SIZE;

    #[inline]
    fn scale(&self, challenge: &Self::Scalar) -> Self {
        *self * *challenge
    }

    #[inline]
    fn pack_full(base: &Self::Scalar) -> Self {
        Self {
            v: [
                BabyBearx16::pack_full(&base.v[0]),
                BabyBearx16::pack_full(&base.v[1]),
                BabyBearx16::pack_full(&base.v[2]),
            ],
        }
    }

    #[inline(always)]
    fn pack(base_vec: &[Self::Scalar]) -> Self {
        assert!(base_vec.len() == Self::PACK_SIZE);
        let mut v0s = vec![];
        let mut v1s = vec![];
        let mut v2s = vec![];

        for scalar in base_vec {
            v0s.push(scalar.v[0]);
            v1s.push(scalar.v[1]);
            v2s.push(scalar.v[2]);
        }

        Self {
            v: [
                BabyBearx16::pack(&v0s),
                BabyBearx16::pack(&v1s),
                BabyBearx16::pack(&v2s),
            ],
        }
    }

    #[inline(always)]
    fn unpack(&self) -> Vec<Self::Scalar> {
        let v0s = self.v[0].unpack();
        let v1s = self.v[1].unpack();
        let v2s = self.v[2].unpack();

        v0s.into_iter()
            .zip(v1s)
            .zip(v2s)
            .map(|((v0, v1), v2)| BabyBearExt3 { v: [v0, v1, v2] })
            .collect()
    }

    #[inline(always)]
    fn horizontal_sum(&self) -> Self::Scalar {
        let limbs = self.to_limbs();
        Self::Scalar {
            v: [
                limbs[0].horizontal_sum(),
                limbs[1].horizontal_sum(),
                limbs[2].horizontal_sum(),
            ],
        }
    }
}

impl From<BabyBearx16> for BabyBearExt3x16 {
    #[inline(always)]
    fn from(x: BabyBearx16) -> Self {
        Self {
            v: [x, BabyBearx16::zero(), BabyBearx16::zero()],
        }
    }
}

impl ExtensionField for BabyBearExt3x16 {
    const DEGREE: usize = 3;

    const W: u32 = 2;

    const X: Self = BabyBearExt3x16 {
        v: [BabyBearx16::ZERO, BabyBearx16::ONE, BabyBearx16::ZERO],
    };

    type BaseField = BabyBearx16;

    #[inline(always)]
    fn mul_by_base_field(&self, base: &Self::BaseField) -> Self {
        BabyBearExt3x16 {
            v: [self.v[0] * base, self.v[1] * base, self.v[2] * base],
        }
    }

    #[inline(always)]
    fn add_by_base_field(&self, base: &Self::BaseField) -> Self {
        BabyBearExt3x16 {
            v: [self.v[0] + base, self.v[1], self.v[2]],
        }
    }

    #[inline(always)]
    fn mul_by_x(&self) -> Self {
        Self {
            v: [self.v[2].mul_by_2(), self.v[0], self.v[1]],
        }
    }

    #[inline(always)]
    fn from_limbs(limbs: &[Self::BaseField]) -> Self {
        let mut v = [Self::BaseField::default(); Self::DEGREE];
        if limbs.len() < Self::DEGREE {
            v[..limbs.len()].copy_from_slice(limbs)
        } else {
            v.copy_from_slice(&limbs[..Self::DEGREE])
        }
        Self { v }
    }

    #[inline(always)]
    fn to_limbs(&self) -> Vec<Self::BaseField> {
        vec![self.v[0], self.v[1], self.v[2]]
    }
}

impl Mul<BabyBearx16> for BabyBearExt3x16 {
    type Output = BabyBearExt3x16;

    #[inline]
    fn mul(self, rhs: BabyBearx16) -> Self::Output {
        self.mul_by_base_field(&rhs)
    }
}

impl From<BabyBearExt3> for BabyBearExt3x16 {
    #[inline(always)]
    fn from(x: BabyBearExt3) -> Self {
        Self {
            v: [
                BabyBearx16::pack_full(&x.v[0]),
                BabyBearx16::pack_full(&x.v[1]),
                BabyBearx16::pack_full(&x.v[2]),
            ],
        }
    }
}

impl Field for BabyBearExt3x16 {
    #[cfg(target_arch = "x86_64")]
    const NAME: &'static str = "AVX Vectorized Baby Bear Extension 3";

    #[cfg(target_arch = "aarch64")]
    const NAME: &'static str = "Neon Vectorized Baby Bear Extension 3";

    const SIZE: usize = 512 / 8 * 3;

    const FIELD_SIZE: usize = 32 * 3;

    const ZERO: Self = Self {
        v: [BabyBearx16::ZERO; 3],
    };

    const ONE: Self = Self {
        v: [BabyBearx16::ONE, BabyBearx16::ZERO, BabyBearx16::ZERO],
    };

    const INV_2: Self = Self {
        v: [BabyBearx16::INV_2, BabyBearx16::ZERO, BabyBearx16::ZERO],
    };

    const MODULUS: U256 = BabyBear::MODULUS;

    #[inline(always)]
    fn zero() -> Self {
        BabyBearExt3x16 {
            v: [BabyBearx16::zero(); 3],
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.v[0].is_zero() && self.v[1].is_zero() && self.v[2].is_zero()
    }

    #[inline(always)]
    fn one() -> Self {
        BabyBearExt3x16 {
            v: [BabyBearx16::one(), BabyBearx16::zero(), BabyBearx16::zero()],
        }
    }

    #[inline(always)]
    fn random_unsafe(mut rng: impl rand::RngCore) -> Self {
        BabyBearExt3x16 {
            v: [
                BabyBearx16::random_unsafe(&mut rng),
                BabyBearx16::random_unsafe(&mut rng),
                BabyBearx16::random_unsafe(&mut rng),
            ],
        }
    }

    #[inline(always)]
    fn random_bool(mut rng: impl rand::RngCore) -> Self {
        BabyBearExt3x16 {
            v: [
                BabyBearx16::random_bool(&mut rng),
                BabyBearx16::zero(),
                BabyBearx16::zero(),
            ],
        }
    }

    #[inline(always)]
    fn square(&self) -> Self {
        Self {
            v: square_internal(&self.v),
        }
    }

    fn inv(&self) -> Option<Self> {
        // slow, should not be used in production
        let mut m31_ext3_vec = self.unpack();
        let is_non_zero = m31_ext3_vec.iter().all(|x| !x.is_zero());
        if !is_non_zero {
            return None;
        }

        m31_ext3_vec.iter_mut().for_each(|x| *x = x.inv().unwrap()); // safe unwrap

        Some(Self::pack(&m31_ext3_vec))
    }

    fn as_u32_unchecked(&self) -> u32 {
        unimplemented!("self is a vector, cannot convert to u32")
    }

    fn from_uniform_bytes(_bytes: &[u8]) -> Self {
        unimplemented!("vec babybear: cannot convert from 32 bytes")
    }
}

impl FFTField for BabyBearExt3x16 {
    const TWO_ADICITY: usize = 27;

    fn root_of_unity() -> Self {
        Self::from(0x1a427a41)
    }
}

impl Mul<BabyBearExt3> for BabyBearExt3x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: BabyBearExt3) -> Self::Output {
        // polynomial mod (x^3 - 2)
        //
        //   (a0 + a1*x + a2*x^2) * (b0 + b1*x + b2*x^2) mod (x^3 - 2)
        // = a0*b0 + (a0*b1 + a1*b0)*x + (a0*b2 + a1*b1 + a2*b0)*x^2
        // + (a1*b2 + a2*b1)*x^3 + a2*b2*x^4 mod (x^3 - 2)
        // = a0*b0 + 2*(a1*b2 + a2*b1)
        // + (a0*b1 + a1*b0)*x + 2* a2*b2
        // + (a0*b2 + a1*b1 + a2*b0)*x^2

        let two = BabyBear::new(2);
        let mut res = [BabyBearx16::default(); 3];
        res[0] = self.v[0] * rhs.v[0] + self.v[1] * (rhs.v[2] * two) + self.v[2] * (rhs.v[1] * two);
        res[1] = self.v[0] * rhs.v[1] + self.v[1] * rhs.v[0] + self.v[2] * (rhs.v[2] * two);
        res[2] = self.v[0] * rhs.v[2] + self.v[1] * rhs.v[1] + self.v[2] * rhs.v[0];
        Self { v: res }
    }
}

impl Mul<BabyBear> for BabyBearExt3x16 {
    type Output = BabyBearExt3x16;
    #[inline(always)]
    fn mul(self, rhs: BabyBear) -> Self::Output {
        BabyBearExt3x16 {
            v: [self.v[0] * rhs, self.v[1] * rhs, self.v[2] * rhs],
        }
    }
}

impl Add<BabyBear> for BabyBearExt3x16 {
    type Output = BabyBearExt3x16;
    #[inline(always)]
    fn add(self, rhs: BabyBear) -> Self::Output {
        BabyBearExt3x16 {
            v: [self.v[0] + rhs, self.v[1], self.v[2]],
        }
    }
}

impl Neg for BabyBearExt3x16 {
    type Output = BabyBearExt3x16;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        BabyBearExt3x16 {
            v: [-self.v[0], -self.v[1], -self.v[2]],
        }
    }
}

impl From<u32> for BabyBearExt3x16 {
    #[inline(always)]
    fn from(x: u32) -> Self {
        BabyBearExt3x16 {
            v: [
                BabyBearx16::from(x),
                BabyBearx16::zero(),
                BabyBearx16::zero(),
            ],
        }
    }
}

#[inline(always)]
fn add_internal(a: &BabyBearExt3x16, b: &BabyBearExt3x16) -> BabyBearExt3x16 {
    let mut vv = a.v;
    vv[0] += b.v[0];
    vv[1] += b.v[1];
    vv[2] += b.v[2];

    BabyBearExt3x16 { v: vv }
}

#[inline(always)]
fn sub_internal(a: &BabyBearExt3x16, b: &BabyBearExt3x16) -> BabyBearExt3x16 {
    let mut vv = a.v;
    vv[0] -= b.v[0];
    vv[1] -= b.v[1];
    vv[2] -= b.v[2];

    BabyBearExt3x16 { v: vv }
}

#[inline(always)]
fn mul_internal(a: &BabyBearExt3x16, b: &BabyBearExt3x16) -> BabyBearExt3x16 {
    let a = &a.v;
    let b = &b.v;
    let mut res = [BabyBearx16::default(); 3];
    res[0] = a[0] * b[0] + (a[1] * b[2] + a[2] * b[1]).mul_by_2();
    res[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[2].mul_by_2();
    res[2] = a[0] * b[2] + a[1] * b[1] + a[2] * b[0];
    BabyBearExt3x16 { v: res }
}

#[inline(always)]
fn square_internal(a: &[BabyBearx16; 3]) -> [BabyBearx16; 3] {
    let mut res = [BabyBearx16::default(); 3];
    let a2_w = a[2].mul_by_2();
    res[0] = a[0].square() + a[1] * a2_w.double();
    res[1] = a[0] * a[1].double() + a[2] * a2_w;
    res[2] = a[0] * a[2].double() + a[1] * a[1];
    res
}

impl Ord for BabyBearExt3x16 {
    #[inline(always)]
    fn cmp(&self, _: &Self) -> std::cmp::Ordering {
        unimplemented!("Ord for BabyBearExt3x16 is not supported")
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for BabyBearExt3x16 {
    #[inline(always)]
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        unimplemented!("PartialOrd for BabyBearExt3x16 is not supported")
    }
}

impl Add<BabyBearx16> for BabyBearExt3x16 {
    type Output = BabyBearExt3x16;

    #[inline(always)]
    fn add(self, rhs: BabyBearx16) -> Self::Output {
        self.add_by_base_field(&rhs)
    }
}
