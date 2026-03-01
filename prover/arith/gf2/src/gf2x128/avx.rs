use std::{
    arch::x86_64::*,
    hash::Hasher,
    mem::{transmute, zeroed},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::Field;
use ethnum::U256;
use serdes::{ExpSerde, SerdeResult};

use crate::GF2;

#[derive(Debug, Clone, Copy)]
pub struct AVXGF2x128 {
    pub v: __m128i,
}

impl ExpSerde for AVXGF2x128 {
    #[inline(always)]
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        unsafe { writer.write_all(transmute::<__m128i, [u8; 16]>(self.v).as_ref())? };
        Ok(())
    }

    #[inline(always)]
    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let mut u = [0u8; 16];
        reader.read_exact(&mut u)?;
        unsafe {
            Ok(AVXGF2x128 {
                v: transmute::<[u8; 16], __m128i>(u),
            })
        }
    }
}

impl Field for AVXGF2x128 {
    const NAME: &'static str = "AVX Galois Field 2 SIMD 128";

    const SIZE: usize = 128 / 8;

    const FIELD_SIZE: usize = 1; // in bits

    const ZERO: Self = AVXGF2x128 {
        v: unsafe { zeroed() },
    };

    const ONE: Self = AVXGF2x128 {
        v: unsafe { transmute::<[u64; 2], __m128i>([!0u64, !0u64]) },
    };

    const INV_2: Self = AVXGF2x128 {
        v: unsafe { zeroed() },
    };

    const MODULUS: U256 = unimplemented!(); // should not be used

    #[inline(always)]
    fn zero() -> Self {
        AVXGF2x128 {
            v: unsafe { zeroed() },
        }
    }

    #[inline(always)]
    fn one() -> Self {
        AVXGF2x128 {
            v: unsafe { transmute::<[u64; 2], __m128i>([!0u64, !0u64]) },
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        unsafe { transmute::<__m128i, [u8; 16]>(self.v) == [0; 16] }
    }

    #[inline(always)]
    fn random_unsafe(mut rng: impl rand::RngCore) -> Self {
        let mut u = [0u8; 16];
        rng.fill_bytes(&mut u);
        unsafe {
            AVXGF2x128 {
                v: *(u.as_ptr() as *const __m128i),
            }
        }
    }

    #[inline(always)]
    fn random_bool(mut rng: impl rand::RngCore) -> Self {
        let mut u = [0u8; 16];
        rng.fill_bytes(&mut u);
        unsafe {
            AVXGF2x128 {
                v: *(u.as_ptr() as *const __m128i),
            }
        }
    }

    #[inline(always)]
    fn exp(&self, exponent: u128) -> Self {
        if exponent == 0 {
            return Self::one();
        }
        *self
    }

    #[inline(always)]
    fn inv(&self) -> Option<Self> {
        unimplemented!()
    }

    #[inline(always)]
    fn as_u32_unchecked(&self) -> u32 {
        unimplemented!("u32 for GF2x128 doesn't make sense")
    }

    #[inline(always)]
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        unsafe {
            AVXGF2x128 {
                v: transmute::<[u8; 16], __m128i>(bytes[..16].try_into().unwrap()),
            }
        }
    }
}

impl Default for AVXGF2x128 {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

impl PartialEq for AVXGF2x128 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe { _mm_test_all_ones(_mm_cmpeq_epi8(self.v, other.v)) == 1 }
    }
}

impl Mul<&AVXGF2x128> for AVXGF2x128 {
    type Output = AVXGF2x128;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: &AVXGF2x128) -> AVXGF2x128 {
        AVXGF2x128 {
            v: unsafe { _mm_and_si128(self.v, rhs.v) },
        }
    }
}

impl Mul<AVXGF2x128> for AVXGF2x128 {
    type Output = AVXGF2x128;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: AVXGF2x128) -> AVXGF2x128 {
        AVXGF2x128 {
            v: unsafe { _mm_and_si128(self.v, rhs.v) },
        }
    }
}

impl MulAssign<&AVXGF2x128> for AVXGF2x128 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn mul_assign(&mut self, rhs: &AVXGF2x128) {
        self.v = unsafe { _mm_and_si128(self.v, rhs.v) };
    }
}

impl MulAssign<AVXGF2x128> for AVXGF2x128 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn mul_assign(&mut self, rhs: AVXGF2x128) {
        self.v = unsafe { _mm_and_si128(self.v, rhs.v) };
    }
}

impl Sub for AVXGF2x128 {
    type Output = AVXGF2x128;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: AVXGF2x128) -> AVXGF2x128 {
        AVXGF2x128 {
            v: unsafe { _mm_xor_si128(self.v, rhs.v) },
        }
    }
}

impl SubAssign for AVXGF2x128 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn sub_assign(&mut self, rhs: AVXGF2x128) {
        self.v = unsafe { _mm_xor_si128(self.v, rhs.v) };
    }
}

impl Add for AVXGF2x128 {
    type Output = AVXGF2x128;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: AVXGF2x128) -> AVXGF2x128 {
        AVXGF2x128 {
            v: unsafe { _mm_xor_si128(self.v, rhs.v) },
        }
    }
}

impl AddAssign for AVXGF2x128 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, rhs: AVXGF2x128) {
        self.v = unsafe { _mm_xor_si128(self.v, rhs.v) };
    }
}

impl Add<&AVXGF2x128> for AVXGF2x128 {
    type Output = AVXGF2x128;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: &AVXGF2x128) -> AVXGF2x128 {
        AVXGF2x128 {
            v: unsafe { _mm_xor_si128(self.v, rhs.v) },
        }
    }
}

impl AddAssign<&AVXGF2x128> for AVXGF2x128 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, rhs: &AVXGF2x128) {
        self.v = unsafe { _mm_xor_si128(self.v, rhs.v) };
    }
}

impl Sub<&AVXGF2x128> for AVXGF2x128 {
    type Output = AVXGF2x128;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: &AVXGF2x128) -> AVXGF2x128 {
        AVXGF2x128 {
            v: unsafe { _mm_xor_si128(self.v, rhs.v) },
        }
    }
}

impl SubAssign<&AVXGF2x128> for AVXGF2x128 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn sub_assign(&mut self, rhs: &AVXGF2x128) {
        self.v = unsafe { _mm_xor_si128(self.v, rhs.v) };
    }
}

impl<T: std::borrow::Borrow<AVXGF2x128>> std::iter::Sum<T> for AVXGF2x128 {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, item| acc + item.borrow())
    }
}

impl<T: std::borrow::Borrow<AVXGF2x128>> std::iter::Product<T> for AVXGF2x128 {
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, item| acc * item.borrow())
    }
}

impl Neg for AVXGF2x128 {
    type Output = AVXGF2x128;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn neg(self) -> AVXGF2x128 {
        AVXGF2x128 { v: self.v }
    }
}

impl From<u32> for AVXGF2x128 {
    #[inline(always)]
    fn from(v: u32) -> Self {
        assert!(v < 2);
        if v == 0 {
            AVXGF2x128::ZERO
        } else {
            AVXGF2x128::ONE
        }
    }
}

impl From<GF2> for AVXGF2x128 {
    #[inline(always)]
    fn from(v: GF2) -> Self {
        assert!(v.v < 2);
        if v.v == 0 {
            AVXGF2x128::ZERO
        } else {
            AVXGF2x128::ONE
        }
    }
}

impl std::hash::Hash for AVXGF2x128 {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe {
            state.write(transmute::<__m128i, [u8; 16]>(self.v).as_ref());
        }
    }
}
