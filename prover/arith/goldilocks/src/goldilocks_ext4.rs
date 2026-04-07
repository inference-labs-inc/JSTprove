use std::{
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::{field_common, ExtensionField, FFTField, Field, SimdField};
use ethnum::U256;
use rand::RngCore;
use serdes::ExpSerde;

use crate::goldilocks::{mod_reduce_u64, Goldilocks};

#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq, ExpSerde)]
pub struct GoldilocksExt4 {
    pub v: [Goldilocks; 4],
}

field_common!(GoldilocksExt4);

impl Field for GoldilocksExt4 {
    const NAME: &'static str = "Goldilocks Extension 4";

    const SIZE: usize = 64 / 8 * 4;

    const FIELD_SIZE: usize = 64 * 4;

    const ZERO: Self = GoldilocksExt4 {
        v: [
            Goldilocks::ZERO,
            Goldilocks::ZERO,
            Goldilocks::ZERO,
            Goldilocks::ZERO,
        ],
    };

    const ONE: Self = GoldilocksExt4 {
        v: [
            Goldilocks::ONE,
            Goldilocks::ZERO,
            Goldilocks::ZERO,
            Goldilocks::ZERO,
        ],
    };

    const INV_2: Self = GoldilocksExt4 {
        v: [
            Goldilocks::INV_2,
            Goldilocks::ZERO,
            Goldilocks::ZERO,
            Goldilocks::ZERO,
        ],
    };

    const MODULUS: U256 = Goldilocks::MODULUS;

    #[inline(always)]
    fn zero() -> Self {
        GoldilocksExt4 {
            v: [Goldilocks::zero(); 4],
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.v[0].is_zero() && self.v[1].is_zero() && self.v[2].is_zero() && self.v[3].is_zero()
    }

    #[inline(always)]
    fn one() -> Self {
        GoldilocksExt4 {
            v: [
                Goldilocks::one(),
                Goldilocks::zero(),
                Goldilocks::zero(),
                Goldilocks::zero(),
            ],
        }
    }

    fn random_unsafe(mut rng: impl RngCore) -> Self {
        GoldilocksExt4 {
            v: [
                Goldilocks::random_unsafe(&mut rng),
                Goldilocks::random_unsafe(&mut rng),
                Goldilocks::random_unsafe(&mut rng),
                Goldilocks::random_unsafe(&mut rng),
            ],
        }
    }

    fn random_bool(mut rng: impl RngCore) -> Self {
        GoldilocksExt4 {
            v: [
                Goldilocks::random_bool(&mut rng),
                Goldilocks::zero(),
                Goldilocks::zero(),
                Goldilocks::zero(),
            ],
        }
    }

    fn inv(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        let f1 = self.frobenius(1);
        let f2 = self.frobenius(2);
        let f3 = self.frobenius(3);
        let norm_val = *self * f1 * f2 * f3;
        debug_assert!(
            norm_val.v[1] == Goldilocks::ZERO
                && norm_val.v[2] == Goldilocks::ZERO
                && norm_val.v[3] == Goldilocks::ZERO
        );
        let norm_inv = norm_val.v[0].inv().expect("norm inverse does not exist");
        let a_pow = f1 * f2 * f3;

        Some(Self {
            v: [
                a_pow.v[0] * norm_inv,
                a_pow.v[1] * norm_inv,
                a_pow.v[2] * norm_inv,
                a_pow.v[3] * norm_inv,
            ],
        })
    }

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
        let mut v0 = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        v0 = mod_reduce_u64(v0);
        let mut v1 = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        v1 = mod_reduce_u64(v1);
        let mut v2 = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        v2 = mod_reduce_u64(v2);
        let mut v3 = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        v3 = mod_reduce_u64(v3);

        GoldilocksExt4 {
            v: [
                Goldilocks { v: v0 },
                Goldilocks { v: v1 },
                Goldilocks { v: v2 },
                Goldilocks { v: v3 },
            ],
        }
    }
}

impl ExtensionField for GoldilocksExt4 {
    const DEGREE: usize = 4;

    const W: u32 = 7;

    const X: Self = GoldilocksExt4 {
        v: [
            Goldilocks::ZERO,
            Goldilocks::ONE,
            Goldilocks::ZERO,
            Goldilocks::ZERO,
        ],
    };

    type BaseField = Goldilocks;

    #[inline(always)]
    fn mul_by_base_field(&self, base: &Self::BaseField) -> Self {
        let mut res = self.v;
        res[0] *= base;
        res[1] *= base;
        res[2] *= base;
        res[3] *= base;
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
            v: [self.v[3].mul_by_7(), self.v[0], self.v[1], self.v[2]],
        }
    }

    #[inline(always)]
    fn to_limbs(&self) -> Vec<Self::BaseField> {
        vec![self.v[0], self.v[1], self.v[2], self.v[3]]
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

impl Mul<Goldilocks> for GoldilocksExt4 {
    type Output = GoldilocksExt4;

    #[inline(always)]
    fn mul(self, rhs: Goldilocks) -> Self::Output {
        self.mul_by_base_field(&rhs)
    }
}

impl Add<Goldilocks> for GoldilocksExt4 {
    type Output = GoldilocksExt4;

    #[inline(always)]
    fn add(self, rhs: Goldilocks) -> Self::Output {
        self + GoldilocksExt4::from(rhs)
    }
}

impl Neg for GoldilocksExt4 {
    type Output = GoldilocksExt4;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        GoldilocksExt4 {
            v: [-self.v[0], -self.v[1], -self.v[2], -self.v[3]],
        }
    }
}

impl From<u32> for GoldilocksExt4 {
    #[inline(always)]
    fn from(x: u32) -> Self {
        GoldilocksExt4 {
            v: [
                Goldilocks::from(x),
                Goldilocks::zero(),
                Goldilocks::zero(),
                Goldilocks::zero(),
            ],
        }
    }
}

impl From<u64> for GoldilocksExt4 {
    #[inline(always)]
    fn from(x: u64) -> Self {
        GoldilocksExt4 {
            v: [
                Goldilocks::from(x),
                Goldilocks::zero(),
                Goldilocks::zero(),
                Goldilocks::zero(),
            ],
        }
    }
}

impl From<Goldilocks> for GoldilocksExt4 {
    #[inline(always)]
    fn from(x: Goldilocks) -> Self {
        GoldilocksExt4 {
            v: [
                x,
                Goldilocks::zero(),
                Goldilocks::zero(),
                Goldilocks::zero(),
            ],
        }
    }
}

impl From<&Goldilocks> for GoldilocksExt4 {
    #[inline(always)]
    fn from(x: &Goldilocks) -> Self {
        GoldilocksExt4 {
            v: [
                *x,
                Goldilocks::zero(),
                Goldilocks::zero(),
                Goldilocks::zero(),
            ],
        }
    }
}

#[inline(always)]
fn add_internal(a: &GoldilocksExt4, b: &GoldilocksExt4) -> GoldilocksExt4 {
    let mut vv = a.v;
    vv[0] += b.v[0];
    vv[1] += b.v[1];
    vv[2] += b.v[2];
    vv[3] += b.v[3];
    GoldilocksExt4 { v: vv }
}

#[inline(always)]
fn sub_internal(a: &GoldilocksExt4, b: &GoldilocksExt4) -> GoldilocksExt4 {
    let mut vv = a.v;
    vv[0] -= b.v[0];
    vv[1] -= b.v[1];
    vv[2] -= b.v[2];
    vv[3] -= b.v[3];
    GoldilocksExt4 { v: vv }
}

#[inline(always)]
fn mul_internal(a: &GoldilocksExt4, b: &GoldilocksExt4) -> GoldilocksExt4 {
    let a = &a.v;
    let b = &b.v;
    let seven = Goldilocks::from(7u32);
    let mut res = [Goldilocks::default(); 4];
    res[0] = a[0] * b[0] + seven * (a[1] * b[3] + a[2] * b[2] + a[3] * b[1]);
    res[1] = a[0] * b[1] + a[1] * b[0] + seven * (a[2] * b[3] + a[3] * b[2]);
    res[2] = a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + seven * a[3] * b[3];
    res[3] = a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0];
    GoldilocksExt4 { v: res }
}

#[inline(always)]
fn square_internal(a: &[Goldilocks; 4]) -> [Goldilocks; 4] {
    let seven = Goldilocks::from(7u32);
    let fourteen = Goldilocks::from(14u32);
    let mut res = [Goldilocks::default(); 4];
    res[0] = a[0].square() + fourteen * (a[1] * a[3]) + seven * a[2].square();
    res[1] = a[0] * a[1].double() + seven * (a[2] * a[3]).double();
    res[2] = a[0] * a[2].double() + a[1].square() + seven * a[3].square();
    res[3] = (a[0] * a[3] + a[1] * a[2]).double();
    res
}

impl Ord for GoldilocksExt4 {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.v[3]
            .cmp(&other.v[3])
            .then(self.v[2].cmp(&other.v[2]))
            .then(self.v[1].cmp(&other.v[1]))
            .then(self.v[0].cmp(&other.v[0]))
    }
}

impl PartialOrd for GoldilocksExt4 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl SimdField for GoldilocksExt4 {
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

impl FFTField for GoldilocksExt4 {
    const TWO_ADICITY: usize = 32;

    #[inline(always)]
    fn root_of_unity() -> Self {
        GoldilocksExt4 {
            v: [
                Goldilocks::root_of_unity(),
                Goldilocks::ZERO,
                Goldilocks::ZERO,
                Goldilocks::ZERO,
            ],
        }
    }
}

const OMEGA: Goldilocks = Goldilocks { v: 281474976710656 };
const OMEGA_SQ: Goldilocks = Goldilocks {
    v: 18446744069414584320,
};
const OMEGA_CU: Goldilocks = Goldilocks {
    v: 18446462594437873665,
};

impl GoldilocksExt4 {
    fn frobenius(&self, count: usize) -> Self {
        let count = count % 4;
        if count == 0 {
            return *self;
        }

        let (c1, c2, c3) = match count {
            1 => (OMEGA, OMEGA_SQ, OMEGA_CU),
            2 => (OMEGA_SQ, Goldilocks::ONE, OMEGA_SQ),
            3 => (OMEGA_CU, OMEGA_SQ, OMEGA),
            _ => unreachable!(),
        };

        Self {
            v: [self.v[0], self.v[1] * c1, self.v[2] * c2, self.v[3] * c3],
        }
    }
}
