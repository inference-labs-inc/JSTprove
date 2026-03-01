use ethnum::U256;
use halo2curves::ff::{Field as Halo2Field, FromUniformBytes, PrimeField};
use rand::RngCore;

use crate::{ExtensionField, FFTField, Field, SimdField};

pub use halo2curves::bn256::Fr;

pub(crate) const MODULUS: U256 = U256([
    0x2833e84879b9709143e1f593f0000001,
    0x30644e72e131a029b85045b68181585d,
]);

impl Field for Fr {
    /// name
    const NAME: &'static str = "bn254 scalar field";

    /// size required to store the data
    const SIZE: usize = 32;

    const FIELD_SIZE: usize = 256;

    /// zero
    const ZERO: Self = Fr::zero();

    /// One
    const ONE: Self = Fr::one();

    /// Inverse of 2
    const INV_2: Self = Fr::TWO_INV;

    /// MODULUS in [u64; 4]
    const MODULUS: U256 = MODULUS;

    // ====================================
    // constants
    // ====================================
    /// Zero element
    #[inline(always)]
    fn zero() -> Self {
        Fr::zero()
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        *self == Fr::zero()
    }

    /// Identity element
    #[inline(always)]
    fn one() -> Self {
        Fr::one()
    }

    // ====================================
    // generators
    // ====================================
    /// create a random element from rng.
    /// test only -- the output may not be uniformly random.
    #[inline(always)]
    fn random_unsafe(rng: impl RngCore) -> Self {
        Fr::random(rng)
    }

    /// create a random boolean element from rng
    #[inline(always)]
    fn random_bool(mut rng: impl RngCore) -> Self {
        Self::from((rng.next_u32() & 1) as u64)
    }

    /// expose the element as u32.
    fn as_u32_unchecked(&self) -> u32 {
        todo!()
    }

    #[inline(always)]
    // TODO: better implementation
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= 32);
        <Fr as FromUniformBytes<64>>::from_uniform_bytes(
            &[&bytes[..32], [0u8; 32].as_slice()]
                .concat()
                .try_into()
                .unwrap(),
        )
    }

    #[inline(always)]
    fn from_u256(x: ethnum::U256) -> Self {
        Fr::from_bytes(&(x % MODULUS).to_le_bytes()).unwrap()
    }

    #[inline(always)]
    fn to_u256(&self) -> ethnum::U256 {
        ethnum::U256::from_le_bytes(self.to_bytes())
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
        self.pow_vartime(exp_limbs)
    }

    /// find the inverse of the element; return None if not exist
    #[inline(always)]
    fn inv(&self) -> Option<Self> {
        self.invert().into()
    }
}

impl SimdField for Fr {
    type Scalar = Self;

    #[inline(always)]
    fn scale(&self, challenge: &Self::Scalar) -> Self {
        self * challenge
    }

    #[inline(always)]
    fn pack_full(base: &Self::Scalar) -> Self {
        *base
    }

    #[inline(always)]
    fn pack(base_vec: &[Self::Scalar]) -> Self {
        assert!(base_vec.len() == 1);
        base_vec[0]
    }

    #[inline(always)]
    fn unpack(&self) -> Vec<Self::Scalar> {
        vec![*self]
    }

    const PACK_SIZE: usize = 1;
}

impl ExtensionField for Fr {
    const DEGREE: usize = 1;

    /// Extension Field over X-1 which is self
    const W: u32 = 1;

    // placeholder, doesn't make sense for Fr
    const X: Self = Fr::zero();

    /// Base field for the extension
    type BaseField = Self;

    /// Multiply the extension field with the base field
    #[inline(always)]
    fn mul_by_base_field(&self, base: &Self::BaseField) -> Self {
        self * base
    }

    /// Add the extension field with the base field
    #[inline(always)]
    fn add_by_base_field(&self, base: &Self::BaseField) -> Self {
        self + base
    }

    /// Multiply the extension field by x, i.e, 0 + x + 0 x^2 + 0 x^3 + ...
    #[inline(always)]
    fn mul_by_x(&self) -> Self {
        unimplemented!("mul_by_x for Fr doesn't make sense")
    }

    /// Construct a new instance of extension field from coefficients
    #[inline(always)]
    fn from_limbs(limbs: &[Self::BaseField]) -> Self {
        if limbs.len() < Self::DEGREE {
            Self::zero()
        } else {
            limbs[0]
        }
    }

    /// Extract polynomial field coefficients from the extension field instance
    #[inline(always)]
    fn to_limbs(&self) -> Vec<Self::BaseField> {
        vec![*self]
    }
}

impl FFTField for Fr {
    const TWO_ADICITY: usize = <Self as PrimeField>::S as usize;

    fn root_of_unity() -> Self {
        <Self as PrimeField>::ROOT_OF_UNITY
    }
}
