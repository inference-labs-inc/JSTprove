#![allow(clippy::needless_range_loop)]

use std::{
    io::{Read, Write},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use ethnum::U256;
use halo2curves::{bn256::Fr, ff::Field as _};
use rand::RngCore;
use serdes::ExpSerde;

use crate::{rep_field_common, ExtensionField, Field, SimdField};

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct FrxN<const N: usize> {
    pub v: [Fr; N],
}

impl<const N: usize> Default for FrxN<N> {
    fn default() -> Self {
        Self { v: [Fr::zero(); N] }
    }
}

impl<const N: usize> ExpSerde for FrxN<N> {
    fn serialize_into<W: Write>(&self, mut writer: W) -> serdes::SerdeResult<()> {
        for elem in &self.v {
            elem.serialize_into(&mut writer)?;
        }
        Ok(())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> serdes::SerdeResult<Self> {
        let mut v = [Fr::zero(); N];
        for i in 0..N {
            v[i] = Fr::deserialize_from(&mut reader)?;
        }
        Ok(Self { v })
    }
}

impl<const N: usize> Ord for FrxN<N> {
    #[inline(always)]
    fn cmp(&self, _: &Self) -> std::cmp::Ordering {
        unimplemented!("Ord for FrxN is not supported")
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl<const N: usize> PartialOrd for FrxN<N> {
    #[inline(always)]
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        unimplemented!("PartialOrd for FrxN is not supported")
    }
}

impl<const N: usize> FrxN<N> {
    pub fn add_internal(&self, b: &Self) -> Self {
        let mut v = [Fr::zero(); N];
        for i in 0..N {
            v[i] = self.v[i] + b.v[i];
        }
        Self { v }
    }

    pub fn sub_internal(&self, b: &Self) -> Self {
        let mut v = [Fr::zero(); N];
        for i in 0..N {
            v[i] = self.v[i] - b.v[i];
        }
        Self { v }
    }

    pub fn mul_internal(&self, b: &Self) -> Self {
        let mut v = [Fr::zero(); N];
        for i in 0..N {
            v[i] = self.v[i] * b.v[i];
        }
        Self { v }
    }
}

impl<const N: usize> From<u32> for FrxN<N> {
    fn from(x: u32) -> Self {
        let v = Fr::from(x);
        Self { v: [v; N] }
    }
}

rep_field_common!( FrxN <const N: usize>);

impl<const N: usize> Field for FrxN<N> {
    /// name
    const NAME: &'static str = "bn254 X N scalar field";

    /// size required to store the data
    const SIZE: usize = Fr::SIZE * N;

    const FIELD_SIZE: usize = 256;

    /// zero
    const ZERO: Self = Self { v: [Fr::zero(); N] };

    /// One
    const ONE: Self = Self { v: [Fr::one(); N] };

    /// Inverse of 2
    const INV_2: Self = Self { v: [Fr::INV_2; N] };

    /// MODULUS in [u64; 4]
    const MODULUS: U256 = super::bn254::MODULUS;

    // ====================================
    // constants
    // ====================================
    /// Zero element
    #[inline(always)]
    fn zero() -> Self {
        Self { v: [Fr::zero(); N] }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }

    /// Identity element
    #[inline(always)]
    fn one() -> Self {
        Self { v: [Fr::one(); N] }
    }

    // ====================================
    // generators
    // ====================================
    /// create a random element from rng.
    /// test only -- the output may not be uniformly random.
    #[inline(always)]
    fn random_unsafe(mut rng: impl RngCore) -> Self {
        Self {
            v: std::array::from_fn(|_| Fr::random(&mut rng)),
        }
    }

    /// create a random boolean element from rng
    #[inline(always)]
    fn random_bool(mut rng: impl RngCore) -> Self {
        Self {
            v: (0..N)
                .map(|_| Fr::random_bool(&mut rng))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap_or_else(|_| panic!("Failed to create FrxN with N = {N}")),
        }
    }

    /// expose the element as u32.
    fn as_u32_unchecked(&self) -> u32 {
        todo!()
    }

    #[inline(always)]
    // TODO: better implementation
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= 32 * N);
        Self {
            v: (0..N)
                .map(|i| Fr::from_uniform_bytes(&bytes[i * 32..(i + 1) * 32]))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }

    #[inline(always)]
    fn from_u256(x: ethnum::U256) -> Self {
        let v = Fr::from_bytes(&(x % Self::MODULUS).to_le_bytes()).unwrap();
        Self { v: [v; N] }
    }

    #[inline(always)]
    fn to_u256(&self) -> ethnum::U256 {
        unimplemented!("to_u256 for FrxN is not implemented");
    }

    // ====================================
    // arithmetics
    // ====================================
    /// Squaring
    #[inline(always)]
    fn square(&self) -> Self {
        *self * *self
    }

    /// Doubling
    #[inline(always)]
    fn double(&self) -> Self {
        *self + *self
    }

    /// Exp
    #[inline(always)]
    fn exp(&self, exp: u128) -> Self {
        let exp_limbs = [exp as u64, (exp >> 64) as u64];
        Self {
            v: std::array::from_fn(|i| self.v[i].pow_vartime(exp_limbs)),
        }
    }

    /// find the inverse of the element; return None if not exist
    #[inline(always)]
    fn inv(&self) -> Option<Self> {
        let invs = self.v.iter().map(|elem| elem.inv()).collect::<Vec<_>>();
        if invs.iter().any(|elem| elem.is_none()) {
            return None;
        }
        Some(Self {
            v: invs
                .into_iter()
                .map(|elem| elem.unwrap())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        })
    }
}

impl<const N: usize> From<Fr> for FrxN<N> {
    fn from(x: Fr) -> Self {
        Self { v: [x; N] }
    }
}

impl<const N: usize> Add<Fr> for FrxN<N> {
    type Output = Self;

    fn add(self, other: Fr) -> Self {
        let mut v = [Fr::zero(); N];
        for i in 0..N {
            v[i] = self.v[i] + other;
        }
        Self { v }
    }
}

impl<const N: usize> Add<FrxN<N>> for Fr {
    type Output = FrxN<N>;

    fn add(self, other: FrxN<N>) -> FrxN<N> {
        let mut v = [Fr::zero(); N];
        for i in 0..N {
            v[i] = self + other.v[i];
        }
        FrxN { v }
    }
}

impl<const N: usize> Mul<Fr> for FrxN<N> {
    type Output = Self;

    fn mul(self, other: Fr) -> Self {
        let mut v = [Fr::zero(); N];
        for i in 0..N {
            v[i] = self.v[i] * other;
        }
        Self { v }
    }
}

impl<const N: usize> Mul<FrxN<N>> for Fr {
    type Output = FrxN<N>;

    fn mul(self, other: FrxN<N>) -> FrxN<N> {
        let mut v = [Fr::zero(); N];
        for i in 0..N {
            v[i] = self * other.v[i];
        }
        FrxN { v }
    }
}

impl<const N: usize> SimdField for FrxN<N> {
    type Scalar = Fr;

    const PACK_SIZE: usize = N;

    fn scale(&self, challenge: &Self::Scalar) -> Self {
        let mut v = [Fr::zero(); N];
        for i in 0..N {
            v[i] = self.v[i] * challenge;
        }
        Self { v }
    }

    fn pack_full(base: &Self::Scalar) -> Self {
        Self { v: [*base; N] }
    }

    fn pack(base_vec: &[Self::Scalar]) -> Self {
        assert_eq!(base_vec.len(), N);
        Self {
            v: base_vec.try_into().unwrap(),
        }
    }

    fn unpack(&self) -> Vec<Self::Scalar> {
        self.v.to_vec()
    }
}

impl<const N: usize> ExtensionField for FrxN<N> {
    const DEGREE: usize = 1;

    const W: u32 = 1;

    const X: Self = Self::ZERO;

    type BaseField = Self;

    fn mul_by_base_field(&self, base: &Self::BaseField) -> Self {
        self * base
    }

    fn add_by_base_field(&self, base: &Self::BaseField) -> Self {
        *self + base
    }

    fn mul_by_x(&self) -> Self {
        self * Self::X
    }

    fn to_limbs(&self) -> Vec<Self::BaseField> {
        vec![*self]
    }

    fn from_limbs(limbs: &[Self::BaseField]) -> Self {
        assert_eq!(limbs.len(), 1);
        limbs[0]
    }
}
