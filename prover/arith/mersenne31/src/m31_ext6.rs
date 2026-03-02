use arith::{field_common, ExtensionField, FFTField, Field};
use ethnum::U256;
use rand::RngCore;
use serdes::ExpSerde;
use std::{
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{m31::mod_reduce_u32_safe, M31Ext3, M31};

#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq, ExpSerde)]
pub struct M31Ext6 {
    pub v: [M31Ext3; 2],
}

field_common!(M31Ext6);

impl Field for M31Ext6 {
    const NAME: &'static str = "Mersenne 31 Extension 6";

    const SIZE: usize = 32 / 8 * 6;

    const FIELD_SIZE: usize = 32 * 6;

    const ZERO: Self = M31Ext6 {
        v: [M31Ext3::ZERO, M31Ext3::ZERO],
    };

    const ONE: Self = M31Ext6 {
        v: [M31Ext3::ONE, M31Ext3::ZERO],
    };

    const INV_2: M31Ext6 = M31Ext6 {
        v: [M31Ext3::INV_2, M31Ext3::ZERO],
    };

    const MODULUS: U256 = M31::MODULUS;

    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.v[0].is_zero() && self.v[1].is_zero()
    }

    #[inline(always)]
    fn one() -> Self {
        Self::ONE
    }

    fn random_unsafe(mut rng: impl RngCore) -> Self {
        Self {
            v: [
                M31Ext3::random_unsafe(&mut rng),
                M31Ext3::random_unsafe(&mut rng),
            ],
        }
    }

    fn random_bool(mut rng: impl RngCore) -> Self {
        M31Ext6 {
            v: [M31Ext3::random_bool(&mut rng), M31Ext3::ZERO],
        }
    }

    fn inv(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        let normalize = (-self.v[0].square() - self.v[1].square().double()).inv()?;

        let compliment = Self {
            v: [-self.v[0] * normalize, self.v[1] * normalize],
        };

        Some(compliment)
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
        assert!(bytes.len() >= 24);
        let v1 = mod_reduce_u32_safe(u32::from_be_bytes(bytes[0..4].try_into().unwrap()));
        let v2 = mod_reduce_u32_safe(u32::from_be_bytes(bytes[4..8].try_into().unwrap()));
        let v3 = mod_reduce_u32_safe(u32::from_be_bytes(bytes[8..12].try_into().unwrap()));

        let a0 = M31Ext3 {
            v: [M31 { v: v1 }, M31 { v: v2 }, M31 { v: v3 }],
        };

        let v4 = mod_reduce_u32_safe(u32::from_be_bytes(bytes[12..16].try_into().unwrap()));
        let v5 = mod_reduce_u32_safe(u32::from_be_bytes(bytes[16..20].try_into().unwrap()));
        let v6 = mod_reduce_u32_safe(u32::from_be_bytes(bytes[20..24].try_into().unwrap()));

        let a1 = M31Ext3 {
            v: [M31 { v: v4 }, M31 { v: v5 }, M31 { v: v6 }],
        };

        Self { v: [a0, a1] }
    }
}

impl ExtensionField for M31Ext6 {
    const DEGREE: usize = 2;

    /// Extension Field
    /// (Y^2 + 2), then the (Y^2 - W) has W = mod - 2
    const W: u32 = (1 << 31) - 3;

    const X: Self = M31Ext6 {
        v: [M31Ext3::ZERO, M31Ext3::ONE],
    };

    /// Base field for the extension
    type BaseField = M31Ext3;

    #[inline(always)]
    /// Multiply the extension field with the base field
    fn mul_by_base_field(&self, base: &Self::BaseField) -> Self {
        let mut res = self.v;
        res[0] *= base;
        res[1] *= base;
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
            v: [-self.v[1].double(), self.v[0]],
        }
    }

    /// Extract polynomial field coefficients from the extension field instance
    #[inline(always)]
    fn to_limbs(&self) -> Vec<Self::BaseField> {
        vec![self.v[0], self.v[1]]
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

impl FFTField for M31Ext6 {
    const TWO_ADICITY: usize = 32;

    #[inline(always)]
    fn root_of_unity() -> Self {
        Self {
            v: [M31Ext3::from(1840555991), M31Ext3::from(599996438)],
        }
    }
}

impl Mul<M31Ext3> for M31Ext6 {
    type Output = M31Ext6;

    #[inline(always)]
    fn mul(self, rhs: M31Ext3) -> Self::Output {
        self.mul_by_base_field(&rhs)
    }
}

impl Neg for M31Ext6 {
    type Output = M31Ext6;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        M31Ext6 {
            v: [-self.v[0], -self.v[1]],
        }
    }
}

impl From<u32> for M31Ext6 {
    #[inline(always)]
    fn from(x: u32) -> Self {
        Self {
            v: [M31Ext3::from(x), M31Ext3::ZERO],
        }
    }
}

impl From<M31Ext3> for M31Ext6 {
    #[inline(always)]
    fn from(x: M31Ext3) -> Self {
        Self {
            v: [x, M31Ext3::ZERO],
        }
    }
}

#[inline(always)]
fn add_internal(a: &M31Ext6, b: &M31Ext6) -> M31Ext6 {
    let mut vv = a.v;
    vv[0] += b.v[0];
    vv[1] += b.v[1];

    M31Ext6 { v: vv }
}

#[inline(always)]
fn sub_internal(a: &M31Ext6, b: &M31Ext6) -> M31Ext6 {
    let mut vv = a.v;
    vv[0] -= b.v[0];
    vv[1] -= b.v[1];

    M31Ext6 { v: vv }
}

// polynomial mod (y^2 + 2)
//
//   (a0 + a1*y) * (b0 + b1*y)              mod (y^2 + 2)
// = a0*b0 + (a0*b1 + a1*b0)*y + a1*b1*y^2  mod (y^2 + 2)
// = (a0*b0 - 2*a1*b1) + (a0*b1 + a1*b0)*y
#[inline(always)]
fn mul_internal(a: &M31Ext6, b: &M31Ext6) -> M31Ext6 {
    let a = &a.v;
    let b = &b.v;
    let mut res = [M31Ext3::default(); 2];

    res[0] = a[0] * b[0] - (a[1] * b[1]).double();
    res[1] = a[0] * b[1] + a[1] * b[0];

    M31Ext6 { v: res }
}

#[inline(always)]
fn square_internal(a: &[M31Ext3; 2]) -> [M31Ext3; 2] {
    let mut res = [M31Ext3::ZERO; 2];
    res[0] = a[0].square() - a[1].square().double();
    res[1] = a[0] * a[1].double();
    res
}

impl Ord for M31Ext6 {
    #[inline(always)]
    fn cmp(&self, _: &Self) -> std::cmp::Ordering {
        unimplemented!("Ord for M31Ext6 is not supported")
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for M31Ext6 {
    #[inline(always)]
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        unimplemented!("PartialOrd for M31Ext6 is not supported")
    }
}
