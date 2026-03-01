use std::{
    arch::x86_64::*,
    hash::Hash,
    io::{Read, Write},
    iter::{Product, Sum},
    mem::transmute,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::{field_common, FFTField, Field, SimdField};
use ethnum::U256;
use rand::Rng;
use rand::RngCore;
use serdes::{ExpSerde, SerdeResult};

use crate::{goldilocks::p2_instructions, Goldilocks, EPSILON, GOLDILOCKS_MOD};

/// Number of Goldilocks elements in each __m512i element
const GOLDILOCKS_PACK_SIZE: usize = 8;

/// Packed field order
const PACKED_GOLDILOCKS_MOD: __m512i = unsafe { transmute([GOLDILOCKS_MOD; GOLDILOCKS_PACK_SIZE]) };

/// Packed epsilon (i.e., 2^64 % modulus)
const PACKED_EPSILON: __m512i = unsafe { transmute([EPSILON; GOLDILOCKS_PACK_SIZE]) };

/// Packed zero
const PACKED_0: __m512i = unsafe { transmute([0u64; GOLDILOCKS_PACK_SIZE]) };

/// Packed inverse of 2
const PACKED_INV_2: __m512i = unsafe { transmute([0x7FFFFFFF80000001u64; GOLDILOCKS_PACK_SIZE]) };

#[derive(Debug, Clone, Copy)]
pub struct AVXGoldilocks {
    // each __m512i element contains 8 Goldilocks elements
    pub v: __m512i,
}

field_common!(AVXGoldilocks);

impl ExpSerde for AVXGoldilocks {
    #[inline(always)]
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        let data = unsafe { transmute::<__m512i, [u8; 64]>(mod_reduce_epi64(self.v)) };
        writer.write_all(&data)?;
        Ok(())
    }

    #[inline(always)]
    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let mut data = [0; 64];
        reader.read_exact(&mut data)?;
        unsafe {
            let value = transmute::<[u8; 64], __m512i>(data);
            Ok(Self { v: value })
        }
    }
}

impl Field for AVXGoldilocks {
    const NAME: &'static str = "AVXGoldilocks";

    const SIZE: usize = 512 / 8;

    const ZERO: Self = Self { v: PACKED_0 };

    const ONE: Self = Self {
        v: unsafe { transmute::<[u64; 8], __m512i>([1; GOLDILOCKS_PACK_SIZE]) },
    };

    const INV_2: Self = Self { v: PACKED_INV_2 };

    const FIELD_SIZE: usize = 64;

    const MODULUS: U256 = U256([GOLDILOCKS_MOD as u128, 0]);

    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn one() -> Self {
        Self::ONE
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        // value is either zero or 0x7FFFFFFF
        unsafe {
            let pcmp = _mm512_cmpeq_epi64_mask(self.v, PACKED_0);
            let pcmp2 = _mm512_cmpeq_epi64_mask(self.v, PACKED_GOLDILOCKS_MOD);
            (pcmp | pcmp2) == 0xFF
        }
    }

    #[inline(always)]
    fn random_unsafe(mut rng: impl RngCore) -> Self {
        // Caution: this may not produce uniformly random elements
        unsafe {
            let mut v = _mm512_setr_epi64(
                rng.gen::<i64>(),
                rng.gen::<i64>(),
                rng.gen::<i64>(),
                rng.gen::<i64>(),
                rng.gen::<i64>(),
                rng.gen::<i64>(),
                rng.gen::<i64>(),
                rng.gen::<i64>(),
            );
            v = mod_reduce_epi64(v);
            Self { v }
        }
    }

    #[inline(always)]
    fn random_bool(mut rng: impl RngCore) -> Self {
        // Caution: this may not produce uniformly random elements
        unsafe {
            let v = _mm512_setr_epi64(
                rng.gen::<bool>() as i64,
                rng.gen::<bool>() as i64,
                rng.gen::<bool>() as i64,
                rng.gen::<bool>() as i64,
                rng.gen::<bool>() as i64,
                rng.gen::<bool>() as i64,
                rng.gen::<bool>() as i64,
                rng.gen::<bool>() as i64,
            );
            Self { v }
        }
    }

    #[inline(always)]
    fn square(&self) -> Self {
        unsafe {
            let (hi, lo) = p3_instructions::square64(self.v);
            AVXGoldilocks {
                v: p3_instructions::reduce128((hi, lo)),
            }
        }
    }

    #[inline(always)]
    fn inv(&self) -> Option<Self> {
        // slow, should not be used in production
        let mut goldilocks_vec =
            unsafe { transmute::<__m512i, [Goldilocks; GOLDILOCKS_PACK_SIZE]>(self.v) };
        let is_non_zero = goldilocks_vec.iter().all(|x| !x.is_zero());
        if !is_non_zero {
            return None;
        }

        goldilocks_vec
            .iter_mut()
            .for_each(|x| *x = x.inv().unwrap()); // safe unwrap
        Some(Self {
            v: unsafe { transmute::<[Goldilocks; GOLDILOCKS_PACK_SIZE], __m512i>(goldilocks_vec) },
        })
    }

    #[inline(always)]
    fn as_u32_unchecked(&self) -> u32 {
        unimplemented!("self is a vector, cannot convert to u32")
    }

    #[inline(always)]
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        let m = Goldilocks::from_uniform_bytes(bytes);
        Self {
            v: unsafe { _mm512_set1_epi64(m.v as i64) },
        }
    }
}

impl SimdField for AVXGoldilocks {
    type Scalar = Goldilocks;

    #[inline]
    fn scale(&self, challenge: &Self::Scalar) -> Self {
        *self * *challenge
    }

    const PACK_SIZE: usize = GOLDILOCKS_PACK_SIZE;

    #[inline(always)]
    fn pack_full(x: &Goldilocks) -> Self {
        unsafe {
            Self {
                v: _mm512_set1_epi64(x.v as i64),
            }
        }
    }

    #[inline(always)]
    fn pack(base_vec: &[Self::Scalar]) -> Self {
        assert_eq!(base_vec.len(), Self::PACK_SIZE);
        let ret: [Self::Scalar; Self::PACK_SIZE] = base_vec.try_into().unwrap();
        unsafe { transmute(ret) }
    }

    #[inline(always)]
    fn unpack(&self) -> Vec<Self::Scalar> {
        let ret = unsafe { transmute::<__m512i, [Self::Scalar; Self::PACK_SIZE]>(self.v) };
        ret.to_vec()
    }

    #[inline(always)]
    fn horizontal_sum(&self) -> Self::Scalar {
        let mut temp: u128 = 0;
        let vars = unsafe { transmute::<__m512i, [Self::Scalar; Self::PACK_SIZE]>(self.v) };
        vars.iter().for_each(|c| temp += c.v as u128);

        p2_instructions::reduce128(temp)
    }
}

impl From<Goldilocks> for AVXGoldilocks {
    #[inline(always)]
    fn from(x: Goldilocks) -> Self {
        Self {
            v: unsafe { _mm512_set1_epi64(x.v as i64) },
        }
    }
}

impl Default for AVXGoldilocks {
    fn default() -> Self {
        Self::zero()
    }
}

impl PartialEq for AVXGoldilocks {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let pcmp = _mm512_cmpeq_epi64_mask(mod_reduce_epi64(self.v), mod_reduce_epi64(other.v));
            pcmp == 0xFF
        }
    }
}

impl Eq for AVXGoldilocks {}

impl Mul<&Goldilocks> for AVXGoldilocks {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: &Goldilocks) -> Self::Output {
        // ZZ: better implementation?
        let rhs_packed = Self::from(*rhs);
        self * rhs_packed
    }
}

impl Mul<Goldilocks> for AVXGoldilocks {
    type Output = Self;

    #[inline(always)]
    #[allow(clippy::op_ref)]
    fn mul(self, rhs: Goldilocks) -> Self::Output {
        self * &rhs
    }
}

impl Add<Goldilocks> for AVXGoldilocks {
    type Output = AVXGoldilocks;
    #[inline(always)]
    fn add(self, rhs: Goldilocks) -> Self::Output {
        self + AVXGoldilocks::pack_full(&rhs)
    }
}

impl Hash for AVXGoldilocks {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        unsafe {
            state.write(transmute::<__m512i, [u8; 64]>(self.v).as_ref());
        }
    }
}

impl Neg for AVXGoldilocks {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        if self.is_zero() {
            self
        } else {
            Self {
                v: unsafe { _mm512_sub_epi64(PACKED_GOLDILOCKS_MOD, self.v) },
            }
        }
    }
}

impl From<u32> for AVXGoldilocks {
    #[inline(always)]
    fn from(x: u32) -> Self {
        Self {
            v: unsafe { _mm512_set1_epi64(x as i64) },
        }
    }
}

impl From<u64> for AVXGoldilocks {
    #[inline(always)]
    fn from(x: u64) -> Self {
        Self::pack_full(&Goldilocks::from(x))
    }
}

impl FFTField for AVXGoldilocks {
    const TWO_ADICITY: usize = 32;

    /// The `2^s` root of unity.
    ///
    /// It can be calculated by exponentiating `Self::MULTIPLICATIVE_GENERATOR` by `t`,
    /// where `t = (modulus - 1) >> Self::S`.
    fn root_of_unity() -> Self {
        Self {
            v: unsafe { _mm512_set1_epi64(0x185629dcda58878c) },
        }
    }
}

#[inline]
unsafe fn mod_reduce_epi64(x: __m512i) -> __m512i {
    // Compare each element with modulus
    let mask = _mm512_cmpge_epu64_mask(x, PACKED_GOLDILOCKS_MOD);
    // If element >= modulus, subtract modulus
    _mm512_mask_sub_epi64(x, mask, x, PACKED_GOLDILOCKS_MOD)
}

#[inline(always)]
fn add_internal(a: &AVXGoldilocks, b: &AVXGoldilocks) -> AVXGoldilocks {
    AVXGoldilocks {
        v: unsafe {
            p3_instructions::add_no_double_overflow_64_64(a.v, p3_instructions::canonicalize(b.v))
        },
    }
}

#[inline(always)]
fn sub_internal(a: &AVXGoldilocks, b: &AVXGoldilocks) -> AVXGoldilocks {
    AVXGoldilocks {
        v: unsafe {
            p3_instructions::sub_no_double_overflow_64_64(a.v, p3_instructions::canonicalize(b.v))
        },
    }
}

#[inline]
fn mul_internal(x: &AVXGoldilocks, y: &AVXGoldilocks) -> AVXGoldilocks {
    unsafe {
        let (hi0, lo0) = p3_instructions::mul64_64(x.v, y.v);
        let res = p3_instructions::reduce128((hi0, lo0));
        AVXGoldilocks { v: res }
    }
}

/// instructions adopted from Plonky3 https://github.com/Plonky3/Plonky3/blob/main/goldilocks/src/x86_64_avx512/packing.rs
mod p3_instructions {
    use super::*;

    #[allow(clippy::useless_transmute)]
    const LO_32_BITS_MASK: __mmask16 = unsafe { transmute(0b0101010101010101u16) };

    #[inline]
    pub(super) unsafe fn canonicalize(x: __m512i) -> __m512i {
        let mask = _mm512_cmpge_epu64_mask(x, PACKED_GOLDILOCKS_MOD);
        _mm512_mask_sub_epi64(x, mask, x, PACKED_GOLDILOCKS_MOD)
    }

    #[inline]
    pub(super) unsafe fn add_no_double_overflow_64_64(x: __m512i, y: __m512i) -> __m512i {
        let res_wrapped = _mm512_add_epi64(x, y);
        let mask = _mm512_cmplt_epu64_mask(res_wrapped, y); // mask set if add overflowed
        _mm512_mask_sub_epi64(res_wrapped, mask, res_wrapped, PACKED_GOLDILOCKS_MOD)
    }

    #[inline]
    pub(super) unsafe fn sub_no_double_overflow_64_64(x: __m512i, y: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epu64_mask(x, y); // mask set if sub will underflow (x < y)
        let res_wrapped = _mm512_sub_epi64(x, y);
        _mm512_mask_add_epi64(res_wrapped, mask, res_wrapped, PACKED_GOLDILOCKS_MOD)
    }

    #[inline]
    pub(super) unsafe fn mul64_64(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
        let x_hi = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(x)));
        let y_hi = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(y)));

        let mul_ll = _mm512_mul_epu32(x, y);
        let mul_lh = _mm512_mul_epu32(x, y_hi);
        let mul_hl = _mm512_mul_epu32(x_hi, y);
        let mul_hh = _mm512_mul_epu32(x_hi, y_hi);

        let mul_ll_hi = _mm512_srli_epi64::<32>(mul_ll);
        let t0 = _mm512_add_epi64(mul_hl, mul_ll_hi);
        let t0_lo = _mm512_and_si512(t0, PACKED_EPSILON);
        let t0_hi = _mm512_srli_epi64::<32>(t0);
        let t1 = _mm512_add_epi64(mul_lh, t0_lo);
        let t2 = _mm512_add_epi64(mul_hh, t0_hi);
        let t1_hi = _mm512_srli_epi64::<32>(t1);
        let res_hi = _mm512_add_epi64(t2, t1_hi);

        let t1_lo = _mm512_castps_si512(_mm512_moveldup_ps(_mm512_castsi512_ps(t1)));
        let res_lo = _mm512_mask_blend_epi32(LO_32_BITS_MASK, t1_lo, mul_ll);

        (res_hi, res_lo)
    }

    #[inline]
    pub(super) unsafe fn square64(x: __m512i) -> (__m512i, __m512i) {
        let x_hi = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(x)));

        let mul_ll = _mm512_mul_epu32(x, x);
        let mul_lh = _mm512_mul_epu32(x, x_hi);
        let mul_hh = _mm512_mul_epu32(x_hi, x_hi);

        let mul_ll_hi = _mm512_srli_epi64::<33>(mul_ll);
        let t0 = _mm512_add_epi64(mul_lh, mul_ll_hi);
        let t0_hi = _mm512_srli_epi64::<31>(t0);
        let res_hi = _mm512_add_epi64(mul_hh, t0_hi);

        let mul_lh_lo = _mm512_slli_epi64::<33>(mul_lh);
        let res_lo = _mm512_add_epi64(mul_ll, mul_lh_lo);

        (res_hi, res_lo)
    }

    #[inline]
    pub(super) unsafe fn reduce128(x: (__m512i, __m512i)) -> __m512i {
        let (hi0, lo0) = x;
        let hi_hi0 = _mm512_srli_epi64::<32>(hi0);
        let lo1 = sub_no_double_overflow_64_64(lo0, hi_hi0);
        let t1 = _mm512_mul_epu32(hi0, PACKED_EPSILON);
        add_no_double_overflow_64_64(lo1, t1)
    }
}
