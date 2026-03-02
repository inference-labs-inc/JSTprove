use std::{
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::{field_common, FFTField, Field, SimdField};
use ethnum::U256;
use rand::RngCore;
use serdes::ExpSerde;

use crate::{goldilocks::p2_instructions, Goldilocks};

/// Number of Goldilocks elements packed
const GOLDILOCKS_PACK_SIZE: usize = 8;

/// NeonGoldilocks packs 8 Goldilocks elements
/// Unlike NeonM31 we end up not using neon's 128-bit vectorized operations
/// Working on vectors seems to be slower since we only pack 2 elements per slot
#[derive(Clone, Copy, Debug, PartialEq, Eq, ExpSerde)]
pub struct NeonGoldilocks {
    pub v: [Goldilocks; 8],
}

field_common!(NeonGoldilocks);

impl Field for NeonGoldilocks {
    const NAME: &'static str = "Neon Packed Goldilocks";

    const SIZE: usize = 64; // 8 * 8 bytes

    const FIELD_SIZE: usize = 64;

    const ZERO: Self = Self {
        v: [Goldilocks::ZERO; 8],
    };

    const ONE: Self = Self {
        v: [Goldilocks::ONE; 8],
    };

    const INV_2: Self = Self {
        v: [Goldilocks::INV_2; 8],
    };

    const MODULUS: U256 = Goldilocks::MODULUS;

    #[inline(always)]
    fn zero() -> Self {
        Self {
            v: [Goldilocks::zero(); 8],
        }
    }

    #[inline(always)]
    fn one() -> Self {
        Self {
            v: [Goldilocks::one(); 8],
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.v.iter().all(|x| x.is_zero())
    }

    #[inline(always)]
    fn random_unsafe(mut rng: impl RngCore) -> Self {
        let mut v = [Goldilocks::zero(); 8];
        for elem in &mut v {
            *elem = Goldilocks::random_unsafe(&mut rng);
        }
        Self { v }
    }

    #[inline(always)]
    fn random_bool(mut rng: impl RngCore) -> Self {
        let mut v = [Goldilocks::zero(); 8];
        for elem in &mut v {
            *elem = Goldilocks::random_bool(&mut rng);
        }
        Self { v }
    }

    #[inline(always)]
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        let m = Goldilocks::from_uniform_bytes(bytes);
        Self::pack_full(&m)
    }

    #[inline(always)]
    fn inv(&self) -> Option<Self> {
        if self.v.iter().any(|x| x.is_zero()) {
            return None;
        }

        let mut res = Self::zero();
        for i in 0..8 {
            res.v[i] = self.v[i].inv()?;
        }
        Some(res)
    }

    fn as_u32_unchecked(&self) -> u32 {
        unimplemented!("self is a vector, cannot convert to u32")
    }
}

impl SimdField for NeonGoldilocks {
    type Scalar = Goldilocks;

    const PACK_SIZE: usize = GOLDILOCKS_PACK_SIZE;

    #[inline(always)]
    fn pack_full(x: &Goldilocks) -> Self {
        Self { v: [*x; 8] }
    }

    #[inline]
    fn scale(&self, challenge: &Self::Scalar) -> Self {
        let res = self.v.map(|x| x * challenge);
        Self { v: res }
    }

    #[inline(always)]
    fn pack(base_vec: &[Self::Scalar]) -> Self {
        assert!(base_vec.len() == GOLDILOCKS_PACK_SIZE);
        let mut v = [Goldilocks::zero(); 8];
        v.copy_from_slice(base_vec);
        Self { v }
    }

    #[inline(always)]
    fn unpack(&self) -> Vec<Self::Scalar> {
        self.v.to_vec()
    }

    #[inline(always)]
    fn horizontal_sum(&self) -> Self::Scalar {
        let mut temp: u128 = 0;
        self.v.iter().for_each(|c| temp += c.v as u128);

        p2_instructions::reduce128(temp)
    }
}

impl Default for NeonGoldilocks {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl From<u32> for NeonGoldilocks {
    #[inline(always)]
    fn from(x: u32) -> Self {
        Self {
            v: [Goldilocks::from(x); 8],
        }
    }
}

impl From<u64> for NeonGoldilocks {
    #[inline(always)]
    fn from(x: u64) -> Self {
        Self {
            v: [Goldilocks::from(x); 8],
        }
    }
}

impl From<Goldilocks> for NeonGoldilocks {
    #[inline(always)]
    fn from(x: Goldilocks) -> Self {
        Self { v: [x; 8] }
    }
}

impl std::hash::Hash for NeonGoldilocks {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for elem in &self.v {
            elem.hash(state);
        }
    }
}

impl Mul<&Goldilocks> for NeonGoldilocks {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: &Goldilocks) -> Self::Output {
        let res = self.v.map(|x| x * rhs);
        Self { v: res }
    }
}

impl Mul<Goldilocks> for NeonGoldilocks {
    type Output = Self;

    #[inline(always)]
    #[allow(clippy::op_ref)]
    fn mul(self, rhs: Goldilocks) -> Self::Output {
        self * &rhs
    }
}

impl Add<Goldilocks> for NeonGoldilocks {
    type Output = NeonGoldilocks;
    #[inline(always)]
    fn add(self, rhs: Goldilocks) -> Self::Output {
        let res = self.v.map(|x| x * rhs);
        Self { v: res }
    }
}

impl Neg for NeonGoldilocks {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        let mut res = Self::zero();
        for i in 0..8 {
            res.v[i] = -self.v[i];
        }
        res
    }
}

#[inline(always)]
fn add_internal(a: &NeonGoldilocks, b: &NeonGoldilocks) -> NeonGoldilocks {
    let mut res = NeonGoldilocks::zero();
    for i in 0..8 {
        res.v[i] = a.v[i] + b.v[i];
    }
    res
}

#[inline(always)]
fn sub_internal(a: &NeonGoldilocks, b: &NeonGoldilocks) -> NeonGoldilocks {
    let mut res = NeonGoldilocks::zero();
    for i in 0..8 {
        res.v[i] = a.v[i] - b.v[i];
    }
    res
}

#[inline(always)]
fn mul_internal(a: &NeonGoldilocks, b: &NeonGoldilocks) -> NeonGoldilocks {
    let mut res = NeonGoldilocks::zero();
    for i in 0..8 {
        res.v[i] = a.v[i] * b.v[i];
    }
    res
}

impl FFTField for NeonGoldilocks {
    const TWO_ADICITY: usize = 32;

    /// The `2^s` root of unity.
    ///
    /// It can be calculated by exponentiating `Self::MULTIPLICATIVE_GENERATOR` by `t`,
    /// where `t = (modulus - 1) >> Self::S`.
    fn root_of_unity() -> Self {
        Self::pack_full(&Goldilocks {
            v: 0x185629dcda58878c,
        })
    }
}
