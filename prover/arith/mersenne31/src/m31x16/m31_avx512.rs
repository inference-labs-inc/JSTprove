use std::{
    arch::x86_64::*,
    fmt::Debug,
    hash::Hash,
    io::{Read, Write},
    iter::{Product, Sum},
    mem::transmute,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::{field_common, Field, SimdField};
use ethnum::U256;
use rand::{Rng, RngCore};
use serdes::{ExpSerde, SerdeResult};

use crate::m31::{M31, M31_MOD};

const M31_PACK_SIZE: usize = 16;
const PACKED_MOD: __m512i = unsafe { transmute([M31_MOD; M31_PACK_SIZE]) };
const PACKED_0: __m512i = unsafe { transmute([0; M31_PACK_SIZE]) };
const PACKED_INV_2: __m512i = unsafe { transmute([1 << 30; M31_PACK_SIZE]) };

#[inline(always)]
unsafe fn mod_reduce_epi32(x: __m512i) -> __m512i {
    let subx = _mm512_sub_epi32(x, PACKED_MOD);
    _mm512_min_epu32(x, subx)
}

#[derive(Clone, Copy)]
pub struct AVXM31 {
    pub v: __m512i,
}

field_common!(AVXM31);

impl ExpSerde for AVXM31 {
    #[inline(always)]
    /// serialize self into bytes
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        let data = unsafe { transmute::<__m512i, [u8; 64]>(mod_reduce_epi32(self.v)) };
        writer.write_all(&data)?;
        Ok(())
    }

    /// deserialize bytes into field
    #[inline(always)]
    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let mut data = [0; 64];
        reader.read_exact(&mut data)?;
        unsafe {
            let mut value = transmute::<[u8; 64], __m512i>(data);
            value = mod_reduce_epi32(value); // ZZ: this step seems unnecessary
            Ok(AVXM31 { v: value })
        }
    }
}

impl Field for AVXM31 {
    const NAME: &'static str = "AVX Packed Mersenne 31";

    // size in bytes
    const SIZE: usize = 512 / 8;

    const ZERO: Self = Self { v: PACKED_0 };

    const ONE: Self = Self {
        v: unsafe { transmute::<[u32; 16], __m512i>([1; M31_PACK_SIZE]) },
    };

    const INV_2: Self = Self { v: PACKED_INV_2 };

    const FIELD_SIZE: usize = 32;

    const MODULUS: U256 = U256([M31_MOD as u128, 0]);

    #[inline(always)]
    fn zero() -> Self {
        AVXM31 { v: PACKED_0 }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        // value is either zero or 0x7FFFFFFF
        unsafe {
            let pcmp = _mm512_cmpeq_epi32_mask(self.v, PACKED_0);
            let pcmp2 = _mm512_cmpeq_epi32_mask(self.v, PACKED_MOD);
            (pcmp | pcmp2) == 0xFFFF
        }
    }

    #[inline(always)]
    fn one() -> Self {
        AVXM31 {
            v: unsafe { _mm512_set1_epi32(1) },
        }
    }

    #[inline(always)]
    // this function is for internal testing only. it is not
    // a source for uniformly random field elements and
    // should not be used in production.
    fn random_unsafe(mut rng: impl RngCore) -> Self {
        // Caution: this may not produce uniformly random elements
        unsafe {
            let mut v = _mm512_setr_epi32(
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
            );
            v = mod_reduce_epi32(v);
            v = mod_reduce_epi32(v);
            AVXM31 { v }
        }
    }

    #[inline(always)]
    fn random_bool(mut rng: impl RngCore) -> Self {
        // TODO: optimize this code
        AVXM31 {
            v: unsafe {
                _mm512_setr_epi32(
                    rng.gen::<bool>() as i32,
                    rng.gen::<bool>() as i32,
                    rng.gen::<bool>() as i32,
                    rng.gen::<bool>() as i32,
                    rng.gen::<bool>() as i32,
                    rng.gen::<bool>() as i32,
                    rng.gen::<bool>() as i32,
                    rng.gen::<bool>() as i32,
                    rng.gen::<bool>() as i32,
                    rng.gen::<bool>() as i32,
                    rng.gen::<bool>() as i32,
                    rng.gen::<bool>() as i32,
                    rng.gen::<bool>() as i32,
                    rng.gen::<bool>() as i32,
                    rng.gen::<bool>() as i32,
                    rng.gen::<bool>() as i32,
                )
            },
        }
    }

    #[inline(always)]
    fn double(&self) -> Self {
        self.mul_by_2()
    }

    #[inline(always)]
    fn mul_by_5(&self) -> AVXM31 {
        let double = unsafe { mod_reduce_epi32(_mm512_slli_epi32::<1>(self.v)) };
        let quad = unsafe { mod_reduce_epi32(_mm512_slli_epi32::<1>(double)) };
        let res = unsafe { mod_reduce_epi32(_mm512_add_epi32(self.v, quad)) };
        Self { v: res }
    }

    #[inline(always)]
    fn inv(&self) -> Option<Self> {
        // slow, should not be used in production
        let mut m31_vec = unsafe { transmute::<__m512i, [M31; 16]>(self.v) };
        let is_non_zero = m31_vec.iter().all(|x| !x.is_zero());
        if !is_non_zero {
            return None;
        }

        m31_vec.iter_mut().for_each(|x| *x = x.inv().unwrap()); // safe unwrap
        Some(Self {
            v: unsafe { transmute::<[M31; 16], __m512i>(m31_vec) },
        })
    }

    fn as_u32_unchecked(&self) -> u32 {
        unimplemented!("self is a vector, cannot convert to u32")
    }

    #[inline]
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        let m = M31::from_uniform_bytes(bytes);
        Self {
            v: unsafe { _mm512_set1_epi32(m.v as i32) },
        }
    }

    #[inline(always)]
    fn mul_by_3(&self) -> AVXM31 {
        let double = unsafe { mod_reduce_epi32(_mm512_slli_epi32::<1>(self.v)) };
        let res = unsafe { mod_reduce_epi32(_mm512_add_epi32(self.v, double)) };
        Self { v: res }
    }
}

impl SimdField for AVXM31 {
    type Scalar = M31;

    #[inline]
    fn scale(&self, challenge: &Self::Scalar) -> Self {
        *self * *challenge
    }

    const PACK_SIZE: usize = M31_PACK_SIZE;

    #[inline(always)]
    fn pack_full(x: &M31) -> AVXM31 {
        AVXM31 {
            v: unsafe { _mm512_set1_epi32(x.v as i32) },
        }
    }

    #[inline(always)]
    fn pack(base_vec: &[Self::Scalar]) -> Self {
        assert!(base_vec.len() == M31_PACK_SIZE);
        let ret: [Self::Scalar; M31_PACK_SIZE] = base_vec.try_into().unwrap();
        unsafe { transmute(ret) }
    }

    #[inline(always)]
    fn unpack(&self) -> Vec<Self::Scalar> {
        let ret = unsafe { transmute::<__m512i, [Self::Scalar; M31_PACK_SIZE]>(self.v) };
        ret.to_vec()
    }

    #[inline(always)]
    fn horizontal_sum(&self) -> Self::Scalar {
        let mut buffer = unsafe {
            let lo_32x8: __m256i = _mm512_castsi512_si256(self.v);
            let hi_32x8: __m256i = _mm512_extracti64x4_epi64(self.v, 1);
            let lo_64x8: __m512i = _mm512_cvtepu32_epi64(lo_32x8);
            let hi_64x8: __m512i = _mm512_cvtepu32_epi64(hi_32x8);
            let sum_64x8 = _mm512_add_epi64(lo_64x8, hi_64x8);
            _mm512_reduce_add_epi64(sum_64x8) as u64
        };

        buffer = (buffer & M31_MOD as u64) + (buffer >> 31);
        if buffer == M31_MOD as u64 {
            Self::Scalar::ZERO
        } else {
            Self::Scalar { v: buffer as u32 }
        }
    }
}

impl From<M31> for AVXM31 {
    #[inline(always)]
    fn from(x: M31) -> Self {
        AVXM31::pack_full(&x)
    }
}

impl Debug for AVXM31 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut data = [0; M31_PACK_SIZE];
        unsafe {
            _mm512_storeu_si512(data.as_mut_ptr() as *mut __m512i, self.v);
        }
        // if all data is the same, print only one
        if data.iter().all(|&x| x == data[0]) {
            write!(
                f,
                "mm512i<8 x {}>",
                if M31_MOD - data[0] > 1024 {
                    format!("{}", data[0])
                } else {
                    format!("-{}", M31_MOD - data[0])
                }
            )
        } else {
            write!(f, "mm512i<{data:?}>")
        }
    }
}

impl Default for AVXM31 {
    fn default() -> Self {
        AVXM31::zero()
    }
}

impl PartialEq for AVXM31 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            // probablisitic -- mod_reduce_epi32 only reduces one mod;
            // i32::MAX = 2*MOD + 1 so there is a small probability that the reduced result
            // does not lie in the field
            let pcmp = _mm512_cmpeq_epi32_mask(mod_reduce_epi32(self.v), mod_reduce_epi32(other.v));
            pcmp == 0xFFFF
        }
    }
}

impl Eq for AVXM31 {}

#[inline]
#[must_use]
fn mask_movehdup_epi32(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
    // The instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters. We cast to floats, do the thing, and cast back.
    unsafe {
        let src = _mm512_castsi512_ps(src);
        let a = _mm512_castsi512_ps(a);
        _mm512_castps_si512(_mm512_mask_movehdup_ps(src, k, a))
    }
}

#[inline]
#[must_use]
fn mask_moveldup_epi32(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
    // The instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters. We cast to floats, do the thing, and cast back.
    unsafe {
        let src = _mm512_castsi512_ps(src);
        let a = _mm512_castsi512_ps(a);
        _mm512_castps_si512(_mm512_mask_moveldup_ps(src, k, a))
    }
}

#[inline]
#[must_use]
fn add(lhs: __m512i, rhs: __m512i) -> __m512i {
    unsafe {
        let t = _mm512_add_epi32(lhs, rhs);
        let u = _mm512_sub_epi32(t, PACKED_MOD);
        _mm512_min_epu32(t, u)
    }
}

const EVENS: __mmask16 = 0b0101010101010101;
const ODDS: __mmask16 = 0b1010101010101010;

impl Mul<&M31> for AVXM31 {
    type Output = AVXM31;

    #[inline(always)]
    fn mul(self, rhs: &M31) -> Self::Output {
        let rhsv = AVXM31::pack_full(rhs);
        mul_internal(&self, &rhsv)
    }
}

impl Mul<M31> for AVXM31 {
    type Output = AVXM31;
    #[inline(always)]
    #[allow(clippy::op_ref)]
    fn mul(self, rhs: M31) -> Self::Output {
        self * &rhs
    }
}

impl Add<M31> for AVXM31 {
    type Output = AVXM31;
    #[inline(always)]
    fn add(self, rhs: M31) -> Self::Output {
        self + AVXM31::pack_full(&rhs)
    }
}

impl From<u32> for AVXM31 {
    #[inline(always)]
    fn from(x: u32) -> Self {
        AVXM31::pack_full(&M31::from(x))
    }
}

impl Neg for AVXM31 {
    type Output = AVXM31;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        AVXM31 {
            v: unsafe { _mm512_xor_epi32(self.v, PACKED_MOD) },
        }
    }
}

#[inline]
#[must_use]
fn movehdup_epi32(x: __m512i) -> __m512i {
    // The instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters. We cast to floats, duplicate, and cast back.
    unsafe { _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(x))) }
}

#[inline(always)]
fn add_internal(a: &AVXM31, b: &AVXM31) -> AVXM31 {
    unsafe {
        let mut result = _mm512_add_epi32(a.v, b.v);
        let subx = _mm512_sub_epi32(result, PACKED_MOD);
        result = _mm512_min_epu32(result, subx);

        AVXM31 { v: result }
    }
}

#[inline(always)]
fn sub_internal(a: &AVXM31, b: &AVXM31) -> AVXM31 {
    AVXM31 {
        v: unsafe {
            let t = _mm512_sub_epi32(a.v, b.v);
            let subx = _mm512_add_epi32(t, PACKED_MOD);
            _mm512_min_epu32(t, subx)
        },
    }
}

#[inline]
fn mul_internal(a: &AVXM31, b: &AVXM31) -> AVXM31 {
    // credit: https://github.com/Plonky3/Plonky3/blob/eeb4e37b20127c4daa871b2bad0df30a7c7380db/mersenne-31/src/x86_64_avx2/packing.rs#L154
    unsafe {
        let rhs_evn = b.v;
        let lhs_odd_dbl = _mm512_srli_epi64(a.v, 31);
        let lhs_evn_dbl = _mm512_add_epi32(a.v, a.v);
        let rhs_odd = movehdup_epi32(b.v);

        let prod_odd_dbl = _mm512_mul_epu32(lhs_odd_dbl, rhs_odd);
        let prod_evn_dbl = _mm512_mul_epu32(lhs_evn_dbl, rhs_evn);

        let prod_lo_dbl = mask_moveldup_epi32(prod_evn_dbl, ODDS, prod_odd_dbl);
        let prod_hi = mask_movehdup_epi32(prod_odd_dbl, EVENS, prod_evn_dbl);
        // Right shift to undo the doubling.
        let prod_lo = _mm512_srli_epi32::<1>(prod_lo_dbl);

        // Standard addition of two 31-bit values.
        let res = add(prod_lo, prod_hi);
        AVXM31 { v: res }
    }
}

impl Hash for AVXM31 {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        unsafe {
            state.write(transmute::<__m512i, [u8; 64]>(self.v).as_ref());
        }
    }
}
