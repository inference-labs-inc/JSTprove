// Galois field over 2^128
// credit to intel for the original implementation
// https://www.intel.com/content/dam/develop/external/us/en/documents/clmul-wp-rev-2-02-2014-04-20.pdf

use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use arith::field_common;
use arith::Field;
use ethnum::U256;
use serdes::{ExpSerde, SerdeResult};

pub const MOD: u32 = 2;

#[derive(Debug, Clone, Copy, Default, PartialOrd, Ord)]
pub struct GF2 {
    pub v: u8,
}

impl PartialEq for GF2 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.v & 1 == other.v & 1
    }
}

impl Eq for GF2 {}

field_common!(GF2);

impl ExpSerde for GF2 {
    #[inline(always)]
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        writer.write_all(self.v.to_le_bytes().as_ref())?;
        Ok(())
    }

    #[inline(always)]
    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let mut u = [0u8; 1];
        reader.read_exact(&mut u)?;
        Ok(GF2 { v: u[0] % 2 })
    }
}

impl Field for GF2 {
    // still will pack 8 bits into a u8

    const NAME: &'static str = "Galois Field 2";

    const SIZE: usize = 1;

    const FIELD_SIZE: usize = 1; // in bits

    const ZERO: Self = GF2 { v: 0 };

    const ONE: Self = GF2 { v: 1 };

    const INV_2: Self = GF2 { v: 0 };

    const MODULUS: U256 = U256([MOD as u128, 0]);

    #[inline(always)]
    fn zero() -> Self {
        GF2 { v: 0 }
    }

    #[inline(always)]
    fn one() -> Self {
        GF2 { v: 1 }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.v == 0
    }

    #[inline(always)]
    fn random_unsafe(mut rng: impl rand::RngCore) -> Self {
        GF2 {
            v: (rng.next_u32() % 2) as u8,
        }
    }

    #[inline(always)]
    fn random_bool(mut rng: impl rand::RngCore) -> Self {
        GF2 {
            v: (rng.next_u32() % 2) as u8,
        }
    }

    #[inline(always)]
    fn exp(&self, exponent: u128) -> Self {
        if exponent % 2 == 0 {
            Self::one()
        } else {
            *self
        }
    }

    #[inline(always)]
    fn inv(&self) -> Option<Self> {
        if self.v == 0 {
            None
        } else {
            Some(Self::one())
        }
    }

    #[inline(always)]
    fn as_u32_unchecked(&self) -> u32 {
        self.v as u32 % 2
    }

    #[inline(always)]
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        GF2 { v: bytes[0] % 2 }
    }

    #[inline(always)]
    fn mul_by_5(&self) -> Self {
        *self
    }

    #[inline(always)]
    fn mul_by_6(&self) -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn to_u256(&self) -> U256 {
        U256([self.v as u128, 0])
    }

    #[inline(always)]
    fn from_u256(value: U256) -> Self {
        // unsafe -- we assume the value is in the field
        GF2 {
            v: (value.0[0] % 2) as u8,
        }
    }
}

impl Neg for GF2 {
    type Output = GF2;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn neg(self) -> GF2 {
        GF2 { v: self.v }
    }
}

impl From<u32> for GF2 {
    #[inline(always)]
    fn from(v: u32) -> Self {
        GF2 { v: (v % 2) as u8 }
    }
}

impl From<bool> for GF2 {
    #[inline(always)]
    fn from(value: bool) -> Self {
        GF2 { v: value.into() }
    }
}

#[inline(always)]
fn add_internal(a: &GF2, b: &GF2) -> GF2 {
    GF2 { v: a.v ^ b.v }
}

#[inline(always)]
fn sub_internal(a: &GF2, b: &GF2) -> GF2 {
    GF2 { v: a.v ^ b.v }
}

#[inline(always)]
fn mul_internal(a: &GF2, b: &GF2) -> GF2 {
    GF2 { v: a.v & b.v }
}

impl std::hash::Hash for GF2 {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u8(self.v);
    }
}
