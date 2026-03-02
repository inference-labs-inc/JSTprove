use std::{
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::{field_common, ExtensionField, FFTField, Field, SimdField};
use ethnum::U256;
use rand::RngCore;
use serdes::ExpSerde;

use crate::{
    goldilocks::{mod_reduce_u64, Goldilocks},
    GoldilocksExt2x8, Goldilocksx8,
};

#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq, ExpSerde)]
pub struct GoldilocksExt2 {
    pub v: [Goldilocks; 2],
}

field_common!(GoldilocksExt2);

impl Field for GoldilocksExt2 {
    const NAME: &'static str = "Goldilocks Extension 2";

    const SIZE: usize = 64 / 8 * 2;

    const FIELD_SIZE: usize = 64 * 2;

    const ZERO: Self = GoldilocksExt2 {
        v: [Goldilocks::ZERO, Goldilocks::ZERO],
    };

    const ONE: Self = GoldilocksExt2 {
        v: [Goldilocks::ONE, Goldilocks::ZERO],
    };

    const INV_2: Self = GoldilocksExt2 {
        v: [Goldilocks::INV_2, Goldilocks::ZERO],
    };

    const MODULUS: U256 = Goldilocks::MODULUS;

    #[inline(always)]
    fn zero() -> Self {
        GoldilocksExt2 {
            v: [Goldilocks::zero(); 2],
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.v[0].is_zero() && self.v[1].is_zero()
    }

    #[inline(always)]
    fn one() -> Self {
        GoldilocksExt2 {
            v: [Goldilocks::one(), Goldilocks::zero()],
        }
    }

    fn random_unsafe(mut rng: impl RngCore) -> Self {
        GoldilocksExt2 {
            v: [
                Goldilocks::random_unsafe(&mut rng),
                Goldilocks::random_unsafe(&mut rng),
            ],
        }
    }

    fn random_bool(mut rng: impl RngCore) -> Self {
        GoldilocksExt2 {
            v: [Goldilocks::random_bool(&mut rng), Goldilocks::zero()],
        }
    }

    fn inv(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        let a_pow_r_minus_1 = self.frobenius();
        let a_pow_r = a_pow_r_minus_1 * *self;
        debug_assert!(a_pow_r.v[1] == Goldilocks::ZERO);
        let a_pow_r_inv = a_pow_r.v[0].inv().expect("inverse does not exist");

        let res = [
            a_pow_r_minus_1.v[0] * a_pow_r_inv,
            a_pow_r_minus_1.v[1] * a_pow_r_inv,
        ];

        Some(Self { v: res })
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
        let mut v1 = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        v1 = mod_reduce_u64(v1);
        let mut v2 = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        v2 = mod_reduce_u64(v2);

        GoldilocksExt2 {
            v: [Goldilocks { v: v1 }, Goldilocks { v: v2 }],
        }
    }
}

impl ExtensionField for GoldilocksExt2 {
    const DEGREE: usize = 2;

    const W: u32 = 7; // x^2 - 7 is the irreducible polynomial

    const X: Self = GoldilocksExt2 {
        v: [Goldilocks::ZERO, Goldilocks::ONE],
    };

    type BaseField = Goldilocks;

    #[inline(always)]
    fn mul_by_base_field(&self, base: &Self::BaseField) -> Self {
        let mut res = self.v;
        res[0] *= base;
        res[1] *= base;
        Self { v: res }
    }

    #[inline(always)]
    fn add_by_base_field(&self, base: &Self::BaseField) -> Self {
        let mut res = self.v;
        res[0] += base;
        Self { v: res }
    }

    #[inline(always)]
    fn mul_by_x(&self) -> Self {
        Self {
            v: [self.v[1].mul_by_7(), self.v[0]],
        }
    }

    #[inline(always)]
    fn to_limbs(&self) -> Vec<Self::BaseField> {
        vec![self.v[0], self.v[1]]
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
}

impl Mul<Goldilocks> for GoldilocksExt2 {
    type Output = GoldilocksExt2;

    #[inline(always)]
    fn mul(self, rhs: Goldilocks) -> Self::Output {
        self.mul_by_base_field(&rhs)
    }
}

impl Add<Goldilocks> for GoldilocksExt2 {
    type Output = GoldilocksExt2;

    #[inline(always)]
    fn add(self, rhs: Goldilocks) -> Self::Output {
        self + GoldilocksExt2::from(rhs)
    }
}

impl Neg for GoldilocksExt2 {
    type Output = GoldilocksExt2;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        GoldilocksExt2 {
            v: [-self.v[0], -self.v[1]],
        }
    }
}

impl From<u32> for GoldilocksExt2 {
    #[inline(always)]
    fn from(x: u32) -> Self {
        GoldilocksExt2 {
            v: [Goldilocks::from(x), Goldilocks::zero()],
        }
    }
}

impl GoldilocksExt2 {
    #[inline(always)]
    pub fn to_base_field(&self) -> Goldilocks {
        assert!(
            self.v[1].is_zero(),
            "GoldilocksExt2 cannot be converted to base field"
        );
        self.to_base_field_unsafe()
    }

    #[inline(always)]
    pub fn to_base_field_unsafe(&self) -> Goldilocks {
        self.v[0]
    }
}

impl From<u64> for GoldilocksExt2 {
    #[inline(always)]
    fn from(x: u64) -> Self {
        GoldilocksExt2 {
            v: [Goldilocks::from(x), Goldilocks::zero()],
        }
    }
}

impl From<Goldilocks> for GoldilocksExt2 {
    #[inline(always)]
    fn from(x: Goldilocks) -> Self {
        GoldilocksExt2 {
            v: [x, Goldilocks::zero()],
        }
    }
}

impl From<&Goldilocks> for GoldilocksExt2 {
    #[inline(always)]
    fn from(x: &Goldilocks) -> Self {
        GoldilocksExt2 {
            v: [*x, Goldilocks::zero()],
        }
    }
}

impl From<GoldilocksExt2> for Goldilocks {
    #[inline(always)]
    fn from(x: GoldilocksExt2) -> Self {
        x.to_base_field()
    }
}

impl From<&GoldilocksExt2> for Goldilocks {
    #[inline(always)]
    fn from(x: &GoldilocksExt2) -> Self {
        x.to_base_field()
    }
}

#[inline(always)]
fn add_internal(a: &GoldilocksExt2, b: &GoldilocksExt2) -> GoldilocksExt2 {
    let mut vv = a.v;
    vv[0] += b.v[0];
    vv[1] += b.v[1];
    GoldilocksExt2 { v: vv }
}

#[inline(always)]
fn sub_internal(a: &GoldilocksExt2, b: &GoldilocksExt2) -> GoldilocksExt2 {
    let mut vv = a.v;
    vv[0] -= b.v[0];
    vv[1] -= b.v[1];
    GoldilocksExt2 { v: vv }
}

// polynomial mod (x^2 - 7)
//
//   (a0 + a1*x) * (b0 + b1*x) mod (x^2 - 7)
// = a0*b0 + (a0*b1 + a1*b0)*x + a1*b1*x^2 mod (x^2 - 7)
// = a0*b0 + 7*a1*b1 + (a0*b1 + a1*b0)*x
#[inline(always)]
fn mul_internal(a: &GoldilocksExt2, b: &GoldilocksExt2) -> GoldilocksExt2 {
    let a = &a.v;
    let b = &b.v;
    let mut res = [Goldilocks::default(); 2];
    res[0] = a[0] * b[0] + a[1] * b[1].mul_by_7();
    res[1] = a[0] * b[1] + a[1] * b[0];
    GoldilocksExt2 { v: res }
}

#[inline(always)]
fn square_internal(a: &[Goldilocks; 2]) -> [Goldilocks; 2] {
    let mut res = [Goldilocks::default(); 2];
    res[0] = a[0].square() + a[1].square().mul_by_7();
    res[1] = a[0] * a[1].double();
    res
}

impl Ord for GoldilocksExt2 {
    #[inline(always)]
    fn cmp(&self, _: &Self) -> std::cmp::Ordering {
        unimplemented!("Ord for GoldilocksExt2 is not supported")
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for GoldilocksExt2 {
    #[inline(always)]
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        unimplemented!("PartialOrd for GoldilocksExt2 is not supported")
    }
}

impl SimdField for GoldilocksExt2 {
    type Scalar = Self;

    const PACK_SIZE: usize = 1;

    #[inline(always)]
    fn scale(&self, challenge: &Self::Scalar) -> Self {
        *self * challenge
    }

    #[inline(always)]
    fn pack_full(x: &Self::Scalar) -> Self {
        *x
    }

    #[inline(always)]
    fn pack(base_vec: &[Self::Scalar]) -> Self {
        assert_eq!(base_vec.len(), 1);
        base_vec[0]
    }

    #[inline(always)]
    fn unpack(&self) -> Vec<Self::Scalar> {
        vec![*self]
    }
}

impl FFTField for GoldilocksExt2 {
    const TWO_ADICITY: usize = 33;

    #[inline(always)]
    fn root_of_unity() -> Self {
        GoldilocksExt2 {
            v: [
                Goldilocks::ZERO,
                Goldilocks {
                    v: 0xd95051a31cf4a6ef,
                },
            ],
        }
    }
}

impl GoldilocksExt2 {
    /// FrobeniusField automorphisms: x -> x^p, where p is the order of BaseField.
    fn frobenius(&self) -> Self {
        self.repeated_frobenius(1)
    }

    /// Repeated Frobenius automorphisms: x -> x^(p^count).
    ///
    /// Follows precomputation suggestion in Section 11.3.3 of the
    /// Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    fn repeated_frobenius(&self, count: usize) -> Self {
        if count == 0 {
            return *self;
        } else if count >= 2 {
            // x |-> x^(p^D) is the identity, so x^(p^count) ==
            // x^(p^(count % D))
            return self.repeated_frobenius(count % 2);
        }
        let arr = self.v;

        // z0 = DTH_ROOT^count = W^(k * count) where k = floor((p^D-1)/D)
        let mut z0 = Goldilocks {
            v: 18446744069414584320,
        };
        for _ in 1..count {
            z0 *= Goldilocks {
                v: 18446744069414584320,
            };
        }
        let z0square = z0 * z0;

        let mut res = [Goldilocks::ZERO; 2];

        res[0] = arr[0] * z0;
        res[1] = arr[1] * z0square;

        Self { v: res }
    }
}

impl Mul<Goldilocksx8> for GoldilocksExt2 {
    type Output = GoldilocksExt2x8;

    #[inline(always)]
    fn mul(self, rhs: Goldilocksx8) -> Self::Output {
        let b_simd_ext = Self::Output::from(self);
        Self::Output {
            c0: b_simd_ext.c0 * rhs,
            c1: b_simd_ext.c1 * rhs,
        }
    }
}
