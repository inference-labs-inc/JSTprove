use std::{
    arch::aarch64::*,
    mem::{transmute, zeroed},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::Field;
use ethnum::U256;
use serdes::{ExpSerde, SerdeResult};

use crate::GF2;

#[derive(Clone, Copy, Debug)]
pub struct NeonGF2x128 {
    pub(crate) v: uint32x4_t,
}

impl ExpSerde for NeonGF2x128 {
    #[inline(always)]
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        unsafe { writer.write_all(transmute::<uint32x4_t, [u8; 16]>(self.v).as_ref())? };
        Ok(())
    }

    #[inline(always)]
    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let mut u = [0u8; 16];
        reader.read_exact(&mut u)?;
        unsafe {
            Ok(NeonGF2x128 {
                v: transmute::<[u8; 16], uint32x4_t>(u),
            })
        }
    }
}

impl Field for NeonGF2x128 {
    const NAME: &'static str = "Neon Galois Field 2 SIMD 128";

    const SIZE: usize = 128 / 8;

    const FIELD_SIZE: usize = 1; // in bits

    const ZERO: Self = NeonGF2x128 {
        v: unsafe { zeroed() },
    };

    const ONE: Self = NeonGF2x128 {
        v: unsafe { transmute::<[u64; 2], uint32x4_t>([!0u64, !0u64]) },
    };

    const INV_2: Self = NeonGF2x128 {
        v: unsafe { zeroed() },
    };

    const MODULUS: U256 = unimplemented!(); // should not be used

    #[inline(always)]
    fn zero() -> Self {
        NeonGF2x128 {
            v: unsafe { zeroed() },
        }
    }

    #[inline(always)]
    fn one() -> Self {
        NeonGF2x128 {
            v: unsafe { transmute::<[u64; 2], uint32x4_t>([!0u64, !0u64]) },
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        unsafe { transmute::<uint32x4_t, [u8; 16]>(self.v) == [0; 16] }
    }

    #[inline(always)]
    fn random_unsafe(mut rng: impl rand::RngCore) -> Self {
        let mut u = [0u8; 16];
        rng.fill_bytes(&mut u);
        unsafe {
            NeonGF2x128 {
                v: *(u.as_ptr() as *const uint32x4_t),
            }
        }
    }

    #[inline(always)]
    fn random_bool(mut rng: impl rand::RngCore) -> Self {
        let mut u = [0u8; 16];
        rng.fill_bytes(&mut u);
        unsafe {
            NeonGF2x128 {
                v: *(u.as_ptr() as *const uint32x4_t),
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
        unimplemented!("u32 for GFx128 doesn't make sense")
    }

    #[inline(always)]
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        unsafe {
            NeonGF2x128 {
                v: transmute::<[u8; 16], uint32x4_t>(bytes[..16].try_into().unwrap()),
            }
        }
    }
}

impl Default for NeonGF2x128 {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

impl PartialEq for NeonGF2x128 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            transmute::<uint32x4_t, [u8; 16]>(self.v) == transmute::<uint32x4_t, [u8; 16]>(other.v)
        }
    }
}

impl Mul<&NeonGF2x128> for NeonGF2x128 {
    type Output = NeonGF2x128;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: &NeonGF2x128) -> NeonGF2x128 {
        NeonGF2x128 {
            v: unsafe { vandq_u32(self.v, rhs.v) },
        }
    }
}

impl Mul<NeonGF2x128> for NeonGF2x128 {
    type Output = NeonGF2x128;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: NeonGF2x128) -> NeonGF2x128 {
        NeonGF2x128 {
            v: unsafe { vandq_u32(self.v, rhs.v) },
        }
    }
}

impl MulAssign<&NeonGF2x128> for NeonGF2x128 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn mul_assign(&mut self, rhs: &NeonGF2x128) {
        self.v = unsafe { vandq_u32(self.v, rhs.v) };
    }
}

impl MulAssign<NeonGF2x128> for NeonGF2x128 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn mul_assign(&mut self, rhs: NeonGF2x128) {
        self.v = unsafe { vandq_u32(self.v, rhs.v) };
    }
}

impl Sub for NeonGF2x128 {
    type Output = NeonGF2x128;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: NeonGF2x128) -> NeonGF2x128 {
        NeonGF2x128 {
            v: unsafe { veorq_u32(self.v, rhs.v) },
        }
    }
}

impl SubAssign for NeonGF2x128 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn sub_assign(&mut self, rhs: NeonGF2x128) {
        self.v = unsafe { veorq_u32(self.v, rhs.v) };
    }
}

impl Add for NeonGF2x128 {
    type Output = NeonGF2x128;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: NeonGF2x128) -> NeonGF2x128 {
        NeonGF2x128 {
            v: unsafe { veorq_u32(self.v, rhs.v) },
        }
    }
}

impl AddAssign for NeonGF2x128 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, rhs: NeonGF2x128) {
        self.v = unsafe { veorq_u32(self.v, rhs.v) };
    }
}

impl Add<&NeonGF2x128> for NeonGF2x128 {
    type Output = NeonGF2x128;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: &NeonGF2x128) -> NeonGF2x128 {
        NeonGF2x128 {
            v: unsafe { veorq_u32(self.v, rhs.v) },
        }
    }
}

impl AddAssign<&NeonGF2x128> for NeonGF2x128 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, rhs: &NeonGF2x128) {
        self.v = unsafe { veorq_u32(self.v, rhs.v) };
    }
}

impl Sub<&NeonGF2x128> for NeonGF2x128 {
    type Output = NeonGF2x128;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: &NeonGF2x128) -> NeonGF2x128 {
        NeonGF2x128 {
            v: unsafe { veorq_u32(self.v, rhs.v) },
        }
    }
}

impl SubAssign<&NeonGF2x128> for NeonGF2x128 {
    #[inline(always)]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn sub_assign(&mut self, rhs: &NeonGF2x128) {
        self.v = unsafe { veorq_u32(self.v, rhs.v) };
    }
}

impl<T: std::borrow::Borrow<NeonGF2x128>> std::iter::Sum<T> for NeonGF2x128 {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, item| acc + item.borrow())
    }
}

impl<T: std::borrow::Borrow<NeonGF2x128>> std::iter::Product<T> for NeonGF2x128 {
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, item| acc * item.borrow())
    }
}

impl Neg for NeonGF2x128 {
    type Output = NeonGF2x128;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn neg(self) -> NeonGF2x128 {
        NeonGF2x128 { v: self.v }
    }
}

impl From<u32> for NeonGF2x128 {
    #[inline(always)]
    fn from(v: u32) -> Self {
        assert!(v < 2);
        if v == 0 {
            NeonGF2x128::ZERO
        } else {
            NeonGF2x128::ONE
        }
    }
}

impl From<GF2> for NeonGF2x128 {
    #[inline(always)]
    fn from(v: GF2) -> Self {
        assert!(v.v < 2);
        if v.v == 0 {
            NeonGF2x128::ZERO
        } else {
            NeonGF2x128::ONE
        }
    }
}

impl std::hash::Hash for NeonGF2x128 {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        unsafe {
            state.write(transmute::<uint32x4_t, [u8; 16]>(self.v).as_ref());
        }
    }
}
