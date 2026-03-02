use std::{
    io::{Read, Write},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::{field_common, Field, SimdField};
use ethnum::U256;
use rand::RngCore;
use serdes::{ExpSerde, SerdeResult};

pub const M31_MOD: u32 = 2147483647;

#[inline]
// if x = MOD this will return MOD instead of 0
// for absolute reduction, use mod_reduce_safe
pub(crate) fn mod_reduce_u32(x: u32) -> u32 {
    (x & M31_MOD) + (x >> 31)
}

pub(crate) fn mod_reduce_u32_safe(x: u32) -> u32 {
    let x = (x & M31_MOD) + (x >> 31);
    if x == M31_MOD {
        0
    } else {
        x
    }
}

#[inline]
fn mod_reduce_i64(x: i64) -> i64 {
    (x & M31_MOD as i64) + (x >> 31)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct M31 {
    pub v: u32,
}

impl PartialEq for M31 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        mod_reduce_u32_safe(self.v) == mod_reduce_u32_safe(other.v)
    }
}

impl Eq for M31 {}

impl PartialOrd for M31 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for M31 {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        mod_reduce_u32_safe(self.v).cmp(&mod_reduce_u32_safe(other.v))
    }
}

field_common!(M31);

impl ExpSerde for M31 {
    #[inline(always)]
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        writer.write_all(self.v.to_le_bytes().as_ref())?;
        Ok(())
    }

    // FIXME: this deserialization function auto corrects invalid inputs.
    // We should use separate APIs for this and for the actual deserialization.
    #[inline(always)]
    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let mut u = [0u8; 4];
        reader.read_exact(&mut u)?;
        let mut v = u32::from_le_bytes(u);
        v = mod_reduce_u32(v);
        Ok(M31 { v })
    }
}

impl M31 {
    // Add two M31 without mod reduction
    #[inline(always)]
    pub fn unsafe_add(&self, rhs: &Self) -> Self {
        Self { v: self.v + rhs.v }
    }

    // Double an M31 without mod reduction
    #[inline(always)]
    pub fn unsafe_double(&self) -> Self {
        Self { v: self.v << 1 }
    }
}

impl Field for M31 {
    const NAME: &'static str = "Mersenne 31";

    const SIZE: usize = 32 / 8;

    const ZERO: Self = M31 { v: 0 };

    const ONE: Self = M31 { v: 1 };

    const INV_2: M31 = M31 { v: 1 << 30 };

    const FIELD_SIZE: usize = 32;

    const MODULUS: U256 = U256([M31_MOD as u128, 0]);

    #[inline(always)]
    fn zero() -> Self {
        M31 { v: 0 }
    }

    #[inline(always)]
    fn one() -> Self {
        M31 { v: 1 }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.v == 0 || self.v == M31_MOD
    }

    #[inline(always)]
    fn random_unsafe(mut rng: impl RngCore) -> Self {
        rng.next_u32().into()
    }

    #[inline(always)]
    fn random_bool(mut rng: impl RngCore) -> Self {
        (rng.next_u32() & 1).into()
    }

    #[inline(always)]
    fn to_u256(&self) -> U256 {
        U256([mod_reduce_u32_safe(self.v) as u128, 0])
    }

    #[inline(always)]
    fn from_u256(value: U256) -> Self {
        // Extract the words from the U256
        let (high, low) = value.into_words();

        // Convert to i64 for safer arithmetic operations
        let mut accumulator: i64 = 0;

        // Process low 128 bits in 31-bit chunks
        for i in 0..5 {
            let shift = i * 31;
            if shift < 128 {
                let mask = if shift + 31 <= 128 {
                    (1u128 << 31) - 1
                } else {
                    (1u128 << (128 - shift)) - 1
                };
                let chunk = ((low >> shift) & mask) as i64;
                accumulator = mod_reduce_i64(accumulator + chunk);
            }
        }

        // Process high 128 bits in 31-bit chunks
        for i in 0..5 {
            let shift = i * 31;
            if shift < 128 {
                let mask = if shift + 31 <= 128 {
                    (1u128 << 31) - 1
                } else {
                    (1u128 << (128 - shift)) - 1
                };
                let chunk = ((high >> shift) & mask) as i64;
                // The high word chunks need to be multiplied by 2^128 mod M31_MOD, which is 16
                accumulator = mod_reduce_i64(accumulator + (chunk * 16));
            }
        }

        // Final reduction to ensure the result is in the correct range
        let result = accumulator as u32;

        Self {
            v: mod_reduce_u32_safe(result),
        }
    }

    #[inline(always)]
    fn inv(&self) -> Option<Self> {
        self.try_inverse()
    }

    #[inline(always)]
    fn as_u32_unchecked(&self) -> u32 {
        self.v
    }

    #[inline(always)]
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        let mut v = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        v = mod_reduce_u32_safe(v);
        M31 { v }
    }

    #[inline(always)]
    fn mul_by_5(&self) -> Self {
        *self * Self { v: 5 }
    }

    #[inline(always)]
    fn mul_by_6(&self) -> Self {
        *self * Self { v: 6 }
    }
}

impl Neg for M31 {
    type Output = M31;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        M31 {
            v: if self.v == 0 { 0 } else { M31_MOD - self.v },
        }
    }
}

impl From<u32> for M31 {
    #[inline(always)]
    fn from(x: u32) -> Self {
        M31 {
            v: if x < M31_MOD { x } else { x % M31_MOD },
        }
    }
}

impl M31 {
    #[inline(always)]
    fn exp_power_of_2(&self, power_log: usize) -> Self {
        let mut res = *self;
        for _ in 0..power_log {
            res = res.square();
        }
        res
    }

    /// credit: https://github.com/Plonky3/Plonky3/blob/ed21a5e11cb20effadaab606598ccad4e70e1a3e/mersenne-31/src/mersenne_31.rs#L235
    #[inline(always)]
    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        // From Fermat's little theorem, in a prime field `F_p`, the inverse of `a` is `a^(p-2)`.
        // Here p-2 = 2147483646 = 1111111111111111111111111111101_2.
        // Uses 30 Squares + 7 Multiplications => 37 Operations total.

        let p1 = *self;
        let p101 = p1.exp_power_of_2(2) * p1;
        let p1111 = p101.square() * p101;
        let p11111111 = p1111.exp_power_of_2(4) * p1111;
        let p111111110000 = p11111111.exp_power_of_2(4);
        let p111111111111 = p111111110000 * p1111;
        let p1111111111111111 = p111111110000.exp_power_of_2(4) * p11111111;
        let p1111111111111111111111111111 = p1111111111111111.exp_power_of_2(12) * p111111111111;
        let p1111111111111111111111111111101 =
            p1111111111111111111111111111.exp_power_of_2(3) * p101;
        Some(p1111111111111111111111111111101)
    }
}

#[inline(always)]
fn add_internal(a: &M31, b: &M31) -> M31 {
    let mut vv = a.v + b.v;
    if vv >= M31_MOD {
        vv -= M31_MOD;
    }
    M31 { v: vv }
}

#[inline(always)]
fn sub_internal(a: &M31, b: &M31) -> M31 {
    let mut vv = a.v + M31_MOD - b.v;
    if vv >= M31_MOD {
        vv -= M31_MOD;
    }
    M31 { v: vv }
}

#[inline(always)]
fn mul_internal(a: &M31, b: &M31) -> M31 {
    let mut vv = a.v as i64 * b.v as i64;
    vv = mod_reduce_i64(vv);

    if vv >= M31_MOD as i64 {
        vv -= M31_MOD as i64;
    }
    M31 { v: vv as u32 }
}

impl std::hash::Hash for M31 {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u32(mod_reduce_u32_safe(self.v));
    }
}

impl SimdField for M31 {
    type Scalar = Self;

    const PACK_SIZE: usize = 1;

    #[inline(always)]
    fn scale(&self, challenge: &Self::Scalar) -> Self {
        *self * challenge
    }

    #[inline(always)]
    fn pack_full(base: &Self::Scalar) -> Self {
        *base
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
