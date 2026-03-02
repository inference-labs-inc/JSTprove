use ethnum::U256;
use rand::RngCore;
use std::{
    iter::{Product, Sum},
    mem::transmute,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::{field_common, ExtensionField, FFTField, Field};
use serdes::ExpSerde;

use crate::{babybear::BabyBear, BabyBearExt3x16, BabyBearx16};

#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq, ExpSerde)]
pub struct BabyBearExt3 {
    pub v: [BabyBear; 3],
}

field_common!(BabyBearExt3);

impl Field for BabyBearExt3 {
    const NAME: &'static str = "Baby Bear Extension 3";

    const SIZE: usize = 32 / 8 * 3;

    const FIELD_SIZE: usize = 32 * 3;

    const ZERO: Self = BabyBearExt3 {
        v: [BabyBear::ZERO, BabyBear::ZERO, BabyBear::ZERO],
    };

    const ONE: Self = BabyBearExt3 {
        v: [BabyBear::ONE, BabyBear::ZERO, BabyBear::ZERO],
    };

    const INV_2: BabyBearExt3 = BabyBearExt3 {
        v: [BabyBear::INV_2, BabyBear::new(0), BabyBear::new(0)],
    };

    const MODULUS: U256 = BabyBear::MODULUS;

    #[inline(always)]
    fn zero() -> Self {
        BabyBearExt3 {
            v: [BabyBear::new(0); 3],
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.v[0].is_zero() && self.v[1].is_zero() && self.v[2].is_zero()
    }

    #[inline(always)]
    fn one() -> Self {
        BabyBearExt3 {
            v: [BabyBear::new(1), BabyBear::new(0), BabyBear::new(0)],
        }
    }

    fn random_unsafe(mut rng: impl RngCore) -> Self {
        BabyBearExt3 {
            v: [
                BabyBear::random_unsafe(&mut rng),
                BabyBear::random_unsafe(&mut rng),
                BabyBear::random_unsafe(&mut rng),
            ],
        }
    }

    fn random_bool(mut rng: impl RngCore) -> Self {
        BabyBearExt3 {
            v: [
                BabyBear::random_bool(&mut rng),
                BabyBear::zero(),
                BabyBear::zero(),
            ],
        }
    }

    fn inv(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            let base_field_size = (1u128 << 31) - 2u128.pow(27) + 1;
            Some(self.exp(base_field_size * base_field_size * base_field_size - 2))
        }
    }

    /// Squaring
    #[inline(always)]
    fn square(&self) -> Self {
        Self {
            v: square_internal(&self.v),
        }
    }

    #[inline(always)]
    fn as_u32_unchecked(&self) -> u32 {
        self.v[0].as_u32_unchecked()
    }

    #[inline(always)]
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= 12);
        let v1 = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
        let v2 = u32::from_be_bytes(bytes[4..8].try_into().unwrap());
        let v3 = u32::from_be_bytes(bytes[8..12].try_into().unwrap());
        Self {
            v: [BabyBear::new(v1), BabyBear::new(v2), BabyBear::new(v3)],
        }
    }
}

impl ExtensionField for BabyBearExt3 {
    const DEGREE: usize = 3;

    /// Extension Field
    const W: u32 = 2;

    const X: Self = BabyBearExt3 {
        v: [BabyBear::ZERO, BabyBear::ONE, BabyBear::ZERO],
    };

    /// Base field for the extension
    type BaseField = BabyBear;

    #[inline(always)]
    /// Multiply the extension field with the base field
    fn mul_by_base_field(&self, base: &Self::BaseField) -> Self {
        let mut res = self.v;
        res[0] *= base;
        res[1] *= base;
        res[2] *= base;
        Self { v: res }
    }

    #[inline(always)]
    /// Add the extension field with the base field
    fn add_by_base_field(&self, base: &Self::BaseField) -> Self {
        let mut res = self.v;
        res[0] += base;
        Self { v: res }
    }

    /// Multiply the extension field by x, i.e, 0 + x + 0 x^2 + 0 x^3 + ...
    #[inline(always)]
    fn mul_by_x(&self) -> Self {
        Self {
            v: [self.v[2].mul_by_2(), self.v[0], self.v[1]],
        }
    }

    /// Extract polynomial field coefficients from the extension field instance
    #[inline(always)]
    fn to_limbs(&self) -> Vec<Self::BaseField> {
        vec![self.v[0], self.v[1], self.v[2]]
    }

    /// Construct a new instance of extension field from coefficients
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
}

impl Mul<BabyBear> for BabyBearExt3 {
    type Output = BabyBearExt3;

    #[inline(always)]
    fn mul(self, rhs: BabyBear) -> Self::Output {
        self.mul_by_base_field(&rhs)
    }
}

impl Add<BabyBear> for BabyBearExt3 {
    type Output = BabyBearExt3;

    #[inline(always)]
    fn add(self, rhs: BabyBear) -> Self::Output {
        self + BabyBearExt3::from(rhs)
    }
}

impl Neg for BabyBearExt3 {
    type Output = BabyBearExt3;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        BabyBearExt3 {
            v: [-self.v[0], -self.v[1], -self.v[2]],
        }
    }
}

impl From<u32> for BabyBearExt3 {
    #[inline(always)]
    fn from(x: u32) -> Self {
        BabyBearExt3 {
            v: [BabyBear::from(x), BabyBear::zero(), BabyBear::zero()],
        }
    }
}

impl FFTField for BabyBearExt3 {
    const TWO_ADICITY: usize = 27;

    fn root_of_unity() -> Self {
        Self::from(0x1a427a41)
    }
}

impl BabyBearExt3 {
    #[inline(always)]
    pub fn to_base_field(&self) -> BabyBear {
        assert!(
            self.v[1].is_zero() && self.v[2].is_zero(),
            "BabyBearExt3 cannot be converted to base field"
        );

        self.to_base_field_unsafe()
    }

    #[inline(always)]
    pub fn to_base_field_unsafe(&self) -> BabyBear {
        self.v[0]
    }

    #[inline(always)]
    pub fn as_u32_array(&self) -> [u32; 3] {
        unsafe { transmute(self.v) }
    }
}

impl From<BabyBear> for BabyBearExt3 {
    #[inline(always)]
    fn from(x: BabyBear) -> Self {
        BabyBearExt3 {
            v: [x, BabyBear::zero(), BabyBear::zero()],
        }
    }
}

impl From<&BabyBear> for BabyBearExt3 {
    #[inline(always)]
    fn from(x: &BabyBear) -> Self {
        BabyBearExt3 {
            v: [*x, BabyBear::zero(), BabyBear::zero()],
        }
    }
}

impl From<BabyBearExt3> for BabyBear {
    #[inline(always)]
    fn from(x: BabyBearExt3) -> Self {
        x.to_base_field()
    }
}

impl From<&BabyBearExt3> for BabyBear {
    #[inline(always)]
    fn from(x: &BabyBearExt3) -> Self {
        x.to_base_field()
    }
}

#[inline(always)]
fn add_internal(a: &BabyBearExt3, b: &BabyBearExt3) -> BabyBearExt3 {
    let mut vv = a.v;
    vv[0] += b.v[0];
    vv[1] += b.v[1];
    vv[2] += b.v[2];

    BabyBearExt3 { v: vv }
}

#[inline(always)]
fn sub_internal(a: &BabyBearExt3, b: &BabyBearExt3) -> BabyBearExt3 {
    let mut vv = a.v;
    vv[0] -= b.v[0];
    vv[1] -= b.v[1];
    vv[2] -= b.v[2];

    BabyBearExt3 { v: vv }
}

// polynomial mod (x^3 - 2)
//
//   (a0 + a1*x + a2*x^2) * (b0 + b1*x + b2*x^2) mod (x^3 - 2)
// = a0*b0 + (a0*b1 + a1*b0)*x + (a0*b2 + a1*b1 + a2*b0)*x^2
// + (a1*b2 + a2*b1)*x^3 + a2*b2*x^4 mod (x^3 - 2)
// = a0*b0 + 2*(a1*b2 + a2*b1)
// + (a0*b1 + a1*b0)*x + 2* a2*b2
// + (a0*b2 + a1*b1 + a2*b0)*x^2
#[inline(always)]
fn mul_internal(a: &BabyBearExt3, b: &BabyBearExt3) -> BabyBearExt3 {
    let a = &a.v;
    let b = &b.v;
    let mut res = [BabyBear::default(); 3];
    res[0] = a[0] * b[0] + BabyBear::new(2) * (a[1] * b[2] + a[2] * b[1]);
    res[1] = a[0] * b[1] + a[1] * b[0] + BabyBear::new(2) * a[2] * b[2];
    res[2] = a[0] * b[2] + a[1] * b[1] + a[2] * b[0];
    BabyBearExt3 { v: res }
}

#[inline(always)]
fn square_internal(a: &[BabyBear; 3]) -> [BabyBear; 3] {
    let mut res = [BabyBear::default(); 3];
    res[0] = a[0].square() + BabyBear::new(4) * (a[1] * a[2]);
    res[1] = a[0] * a[1].double() + BabyBear::new(2) * a[2].square();
    res[2] = a[0] * a[2].double() + a[1].square();
    res
}

impl Ord for BabyBearExt3 {
    #[inline(always)]
    fn cmp(&self, _: &Self) -> std::cmp::Ordering {
        unimplemented!("Ord for BabyBearExt3 is not supported")
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for BabyBearExt3 {
    #[inline(always)]
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        unimplemented!("PartialOrd for BabyBearExt3 is not supported")
    }
}

impl Mul<BabyBearx16> for BabyBearExt3 {
    type Output = BabyBearExt3x16;

    #[inline(always)]
    fn mul(self, rhs: BabyBearx16) -> Self::Output {
        let mut res = Self::Output::from(self);
        for v in res.v.iter_mut() {
            *v *= rhs;
        }
        res
    }
}
