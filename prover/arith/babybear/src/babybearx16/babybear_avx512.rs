use arith::{field_common, Field, SimdField};
use ark_std::iterable::Iterable;
use ethnum::U256;
use rand::{Rng, RngCore};
use serdes::{ExpSerde, SerdeResult};
use std::{
    arch::x86_64::*,
    fmt::Debug,
    hash::Hash,
    io::{Read, Write},
    iter::{Product, Sum},
    mem::transmute,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{babybear::BABY_BEAR_MOD, BabyBear};

#[inline]
unsafe fn mod_reduce_epi32(x: __m512i) -> __m512i {
    // Compare each element with modulus
    let mask = _mm512_cmpgt_epu32_mask(x, PACKED_MOD);
    // If element > modulus, subtract modulus
    _mm512_mask_sub_epi32(x, mask, x, PACKED_MOD)
}

const BABY_BEAR_PACK_SIZE: usize = 16;

const PACKED_0: __m512i = unsafe { transmute([0; BABY_BEAR_PACK_SIZE]) };

// 1 in Montgomery form
const PACKED_1: __m512i = unsafe { transmute([0xffffffe; BABY_BEAR_PACK_SIZE]) };

// 2^-1 Montgomery form
const PACKED_INV_2: __m512i = unsafe { transmute([0x7ffffff; BABY_BEAR_PACK_SIZE]) };

const PACKED_MOD: __m512i = unsafe { transmute([BABY_BEAR_MOD; BABY_BEAR_PACK_SIZE]) };

const PACKED_MU: __m512i =
    unsafe { transmute::<[u32; BABY_BEAR_PACK_SIZE], _>([0x88000001; BABY_BEAR_PACK_SIZE]) };

#[derive(Clone, Copy)]
pub struct AVXBabyBear {
    pub v: __m512i,
}

field_common!(AVXBabyBear);

impl ExpSerde for AVXBabyBear {
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
            let value = transmute::<[u8; 64], __m512i>(data);
            Ok(AVXBabyBear { v: value })
        }
    }
}

impl Field for AVXBabyBear {
    const NAME: &'static str = "AVXBabyBear";

    const SIZE: usize = 512 / 8;

    const ZERO: Self = Self { v: PACKED_0 };

    const ONE: Self = Self { v: PACKED_1 };

    const INV_2: Self = Self { v: PACKED_INV_2 };

    const FIELD_SIZE: usize = 32;

    const MODULUS: U256 = U256([BABY_BEAR_MOD as u128, 0]);

    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn one() -> Self {
        Self { v: PACKED_1 }
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
            Self { v }
        }
    }

    #[inline(always)]
    fn random_bool(mut rng: impl RngCore) -> Self {
        // Caution: this may not produce uniformly random elements
        unsafe {
            let v = _mm512_setr_epi32(
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
            );
            Self { v }
        }
    }

    #[inline(always)]
    fn inv(&self) -> Option<Self> {
        // slow, should not be used in production
        let values = unsafe { transmute::<__m512i, [BabyBear; BABY_BEAR_PACK_SIZE]>(self.v) };
        let is_non_zero = values.iter().all(|x| !x.is_zero());
        if !is_non_zero {
            return None;
        }
        let inv = values.iter().map(|x| x.inv().unwrap()).collect::<Vec<_>>();
        Some(Self {
            v: unsafe {
                transmute::<[BabyBear; BABY_BEAR_PACK_SIZE], __m512i>(inv.try_into().unwrap())
            },
        })
    }

    #[inline(always)]
    fn as_u32_unchecked(&self) -> u32 {
        unimplemented!("self is a vector, cannot convert to u32")
    }

    #[inline(always)]
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        let m = BabyBear::from_uniform_bytes(bytes);
        Self {
            v: unsafe { _mm512_set1_epi32(m.value as i32) },
        }
    }
}

impl SimdField for AVXBabyBear {
    type Scalar = BabyBear;

    #[inline]
    fn scale(&self, challenge: &Self::Scalar) -> Self {
        *self * *challenge
    }

    const PACK_SIZE: usize = BABY_BEAR_PACK_SIZE;

    #[inline(always)]
    fn pack_full(x: &BabyBear) -> AVXBabyBear {
        AVXBabyBear {
            v: unsafe { _mm512_set1_epi32(x.value as i32) },
        }
    }

    #[inline(always)]
    fn pack(base_vec: &[Self::Scalar]) -> Self {
        assert!(base_vec.len() == BABY_BEAR_PACK_SIZE);
        let ret: [Self::Scalar; BABY_BEAR_PACK_SIZE] = base_vec.try_into().unwrap();
        unsafe { transmute(ret) }
    }

    #[inline(always)]
    fn unpack(&self) -> Vec<Self::Scalar> {
        let ret = unsafe { transmute::<__m512i, [Self::Scalar; BABY_BEAR_PACK_SIZE]>(self.v) };
        ret.to_vec()
    }
}

impl From<BabyBear> for AVXBabyBear {
    #[inline(always)]
    fn from(x: BabyBear) -> Self {
        AVXBabyBear::pack_full(&x)
    }
}

impl Debug for AVXBabyBear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut data = [0; BABY_BEAR_PACK_SIZE];
        unsafe {
            _mm512_storeu_si512(data.as_mut_ptr() as *mut __m512i, self.v);
        }
        // if all data is the same, print only one
        if data.iter().all(|x| x == data[0]) {
            write!(
                f,
                "mm512i<8 x {}>",
                if BABY_BEAR_MOD - data[0] > 1024 {
                    format!("{}", data[0])
                } else {
                    format!("-{}", BABY_BEAR_MOD - data[0])
                }
            )
        } else {
            write!(f, "mm512i<{data:?}>")
        }
    }
}

impl Default for AVXBabyBear {
    fn default() -> Self {
        AVXBabyBear::zero()
    }
}

impl PartialEq for AVXBabyBear {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let pcmp = _mm512_cmpeq_epi32_mask(mod_reduce_epi32(self.v), mod_reduce_epi32(other.v));
            pcmp == 0xFFFF
        }
    }
}

impl Eq for AVXBabyBear {}

#[inline]
#[must_use]
fn add_internal(a: &AVXBabyBear, b: &AVXBabyBear) -> AVXBabyBear {
    AVXBabyBear {
        v: p3_instructions::add(a.v, b.v),
    }
}

#[inline]
#[must_use]
fn sub_internal(a: &AVXBabyBear, b: &AVXBabyBear) -> AVXBabyBear {
    AVXBabyBear {
        v: p3_instructions::sub(a.v, b.v),
    }
}

#[inline]
#[must_use]
fn mul_internal(a: &AVXBabyBear, b: &AVXBabyBear) -> AVXBabyBear {
    AVXBabyBear {
        v: p3_instructions::mul(a.v, b.v),
    }
}

impl Mul<&BabyBear> for AVXBabyBear {
    type Output = AVXBabyBear;

    #[inline(always)]
    fn mul(self, rhs: &BabyBear) -> Self::Output {
        let rhsv = AVXBabyBear::pack_full(rhs);
        mul_internal(&self, &rhsv)
    }
}

impl Mul<BabyBear> for AVXBabyBear {
    type Output = AVXBabyBear;
    #[inline(always)]
    #[allow(clippy::op_ref)]
    fn mul(self, rhs: BabyBear) -> Self::Output {
        self * &rhs
    }
}

impl Add<BabyBear> for AVXBabyBear {
    type Output = AVXBabyBear;
    #[inline(always)]
    fn add(self, rhs: BabyBear) -> Self::Output {
        self + AVXBabyBear::pack_full(&rhs)
    }
}

impl From<u32> for AVXBabyBear {
    #[inline(always)]
    fn from(x: u32) -> Self {
        AVXBabyBear::pack_full(&BabyBear::from(x))
    }
}

impl Neg for AVXBabyBear {
    type Output = AVXBabyBear;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        AVXBabyBear {
            v: p3_instructions::neg(self.v),
        }
    }
}

impl Hash for AVXBabyBear {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        unsafe {
            state.write(transmute::<__m512i, [u8; 64]>(self.v).as_ref());
        }
    }
}

mod p3_instructions {

    use core::arch::asm;
    use std::{arch::x86_64::*, hint::unreachable_unchecked, mem::transmute};

    use super::{PACKED_MOD, PACKED_MU};

    const EVENS: __mmask16 = 0b0101010101010101;

    /// Add two vectors of MontyField31 elements in canonical form.
    ///
    /// We allow a slight loosening of the canonical form requirement. One of this inputs
    /// must be in canonical form [0, P) but the other is also allowed to equal P.
    /// If the inputs do not conform to this representation, the result is undefined.
    #[inline]
    #[must_use]
    pub(super) fn add(lhs: __m512i, rhs: __m512i) -> __m512i {
        // We want this to compile to:
        //      vpaddd   t, lhs, rhs
        //      vpsubd   u, t, P
        //      vpminud  res, t, u
        // throughput: 1.5 cyc/vec (10.67 els/cyc)
        // latency: 3 cyc

        // Let t := lhs + rhs. We want to return t mod P. Recall that lhs and rhs are in [0, P]
        //   with at most one of them equal to P. Hence t is in [0, 2P - 1] and so it suffices
        //   to return t if t < P and t - P otherwise.
        // Let u := (t - P) mod 2^32 and r := unsigned_min(t, u).
        // If t is in [0, P - 1], then u is in (P - 1 <) 2^32 - P, ..., 2^32 - 1 and r = t.
        // Otherwise, t is in [P, 2P - 1], and u is in [0, P - 1] (< P) and r = u. Hence, r is t if
        //   t < P and t - P otherwise, as desired.
        unsafe {
            // Safety: If this code got compiled then AVX-512F intrinsics are available.
            let t = _mm512_add_epi32(lhs, rhs);
            let u = _mm512_sub_epi32(t, PACKED_MOD);
            _mm512_min_epu32(t, u)
        }
    }

    /// Subtract vectors of MontyField31 elements in canonical form.
    ///
    /// We allow a slight loosening of the canonical form requirement. The
    /// rhs input is additionally allowed to be P.
    /// If the inputs do not conform to this representation, the result is undefined.
    #[inline]
    #[must_use]
    pub(super) fn sub(lhs: __m512i, rhs: __m512i) -> __m512i {
        // We want this to compile to:
        //      vpsubd   t, lhs, rhs
        //      vpaddd   u, t, P
        //      vpminud  res, t, u
        // throughput: 1.5 cyc/vec (10.67 els/cyc)
        // latency: 3 cyc

        // Let t := lhs - rhs. We want to return t mod P. Recall that lhs is in [0, P - 1]
        //   and rhs is in [0, P] so t is in (-2^31 <) -P, ..., P - 1 (< 2^31). It suffices to
        // return t if   t >= 0 and t + P otherwise.
        // Let u := (t + P) mod 2^32 and r := unsigned_min(t, u).
        // If t is in [0, P - 1], then u is in P, ..., 2 P - 1 and r = t.
        // Otherwise, t is in [-P, -1], u is in [0, P - 1] (< P) and r = u. Hence, r is t if
        //   t < P and t - P otherwise, as desired.
        unsafe {
            // Safety: If this code got compiled then AVX-512F intrinsics are available.
            let t = _mm512_sub_epi32(lhs, rhs);
            let u = _mm512_add_epi32(t, PACKED_MOD);
            _mm512_min_epu32(t, u)
        }
    }

    /// Viewing the input as a vector of 16 `u32`s, copy the odd elements into the even elements
    /// below them. In other words, for all `0 <= i < 8`, set the even elements according to
    /// `res[2 * i] := a[2 * i + 1]`, and the odd elements according to
    /// `res[2 * i + 1] := a[2 * i + 1]`.
    #[inline]
    #[must_use]
    fn movehdup_epi32(a: __m512i) -> __m512i {
        // The instruction is only available in the floating-point flavor; this distinction is only
        // for historical reasons and no longer matters. We cast to floats, do the thing,
        // and cast back.
        unsafe { _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(a))) }
    }

    /// Viewing `a` as a vector of 16 `u32`s, copy the odd elements into the even elements below
    /// them, then merge with `src` according to the mask provided. In other words, for all `0
    /// <= i < 8`, set the even elements according to `res[2 * i] := if k[2 * i] { a[2 * i + 1]
    /// } else { src[2 * i] }`, and the odd elements according to
    /// `res[2 * i + 1] := if k[2 * i + 1] { a[2 * i + 1] } else { src[2 * i + 1] }`.
    #[inline]
    #[must_use]
    fn mask_movehdup_epi32(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
        // The instruction is only available in the floating-point flavor; this distinction is only
        // for historical reasons and no longer matters.

        // While we can write this using intrinsics, when inlined, the intrinsic often compiles
        // to a vpermt2ps which has worse latency, see https://godbolt.org/z/489aaPhz3.
        // Hence we use inline assembly to force the compiler to do the right thing.
        unsafe {
            let dst: __m512i;
            asm!(
                "vmovshdup {src_dst}{{{k}}}, {a}",
                src_dst = inlateout(zmm_reg) src => dst,
                k = in(kreg) k,
                a = in(zmm_reg) a,
                options(nomem, nostack, preserves_flags, pure),
            );
            dst
        }
    }

    /// Multiply a vector of unsigned field elements return a vector of unsigned field elements
    /// lying in [0, P).
    ///
    /// Note that the input does not need to be in canonical form but must satisfy
    /// the bound `lhs * rhs < 2^32 * P`. If this bound is not satisfied, the result
    /// is undefined.
    #[inline]
    #[must_use]
    pub(super) fn mul(lhs: __m512i, rhs: __m512i) -> __m512i {
        // We want this to compile to:
        //      vmovshdup  lhs_odd, lhs
        //      vmovshdup  rhs_odd, rhs
        //      vpmuludq   prod_evn, lhs, rhs
        //      vpmuludq   prod_hi, lhs_odd, rhs_odd
        //      vpmuludq   q_evn, prod_evn, MU
        //      vpmuludq   q_odd, prod_hi, MU
        //      vmovshdup  prod_hi{EVENS}, prod_evn
        //      vpmuludq   q_p_evn, q_evn, P
        //      vpmuludq   q_p_hi, q_odd, P
        //      vmovshdup  q_p_hi{EVENS}, q_p_evn
        //      vpcmpltud  underflow, prod_hi, q_p_hi
        //      vpsubd     res, prod_hi, q_p_hi
        //      vpaddd     res{underflow}, res, P
        // throughput: 6.5 cyc/vec (2.46 els/cyc)
        // latency: 21 cyc
        unsafe {
            // `vpmuludq` only reads the even doublewords, so when we pass `lhs` and `rhs` directly
            // we get the eight products at even positions.
            let lhs_evn = lhs;
            let rhs_evn = rhs;

            // Copy the odd doublewords into even positions to compute the eight products at odd
            // positions.
            // NB: The odd doublewords are ignored by `vpmuludq`, so we have a lot of choices for
            // how to do this; `vmovshdup` is nice because it runs on a memory port if
            // the operand is in memory, thus improving our throughput.
            let lhs_odd = movehdup_epi32(lhs);
            let rhs_odd = movehdup_epi32(rhs);

            let prod_evn = _mm512_mul_epu32(lhs_evn, rhs_evn);
            let prod_odd = _mm512_mul_epu32(lhs_odd, rhs_odd);

            // We throw a confuse compiler here to prevent the compiler from
            // using vpmullq instead of vpmuludq in the computations for q_p.
            // vpmullq has both higher latency and lower throughput.
            let q_evn = confuse_compiler(_mm512_mul_epu32(prod_evn, PACKED_MU));
            let q_odd = confuse_compiler(_mm512_mul_epu32(prod_odd, PACKED_MU));

            // Get all the high halves as one vector: this is `(lhs * rhs) >> 32`.
            // NB: `vpermt2d` may feel like a more intuitive choice here, but it has much higher
            // latency.
            let prod_hi = mask_movehdup_epi32(prod_odd, EVENS, prod_evn);

            // Normally we'd want to mask to perform % 2**32, but the instruction below only reads
            // the low 32 bits anyway.
            let q_p_evn = _mm512_mul_epu32(q_evn, PACKED_MOD);
            let q_p_odd = _mm512_mul_epu32(q_odd, PACKED_MOD);

            // We can ignore all the low halves of `q_p` as they cancel out. Get all the high halves
            // as one vector.
            let q_p_hi = mask_movehdup_epi32(q_p_odd, EVENS, q_p_evn);

            // Subtraction `prod_hi - q_p_hi` modulo `P`.
            // NB: Normally we'd `vpaddd P` and take the `vpminud`, but `vpminud` runs on port 0,
            // which is already under a lot of pressure performing multiplications. To
            // relieve this pressure, we check for underflow to generate a mask, and
            // then conditionally add `P`. The underflow check runs on port 5,
            // increasing our throughput, although it does cost us an additional
            // cycle of latency.
            let underflow = _mm512_cmplt_epu32_mask(prod_hi, q_p_hi);
            let t = _mm512_sub_epi32(prod_hi, q_p_hi);
            _mm512_mask_add_epi32(t, underflow, t, PACKED_MOD)
        }
    }

    /// No-op. Prevents the compiler from deducing the value of the vector.
    ///
    /// Similar to `std::hint::black_box`, it can be used to stop the compiler applying undesirable
    /// "optimizations". Unlike the built-in `black_box`, it does not force the value to be written
    /// to and then read from the stack.
    #[inline]
    #[must_use]
    fn confuse_compiler(x: __m512i) -> __m512i {
        let y;
        unsafe {
            asm!(
                "/*{0}*/",
                inlateout(zmm_reg) x => y,
                options(nomem, nostack, preserves_flags, pure),
            );
            // Below tells the compiler the semantics of this so it can still do constant folding,
            // etc. You may ask, doesn't it defeat the point of the inline asm block to
            // tell the compiler what it does? The answer is that we still inhibit the
            // transform we want to avoid, so apparently not. Idk, LLVM works in
            // mysterious ways.
            if transmute::<__m512i, [u32; 16]>(x) != transmute::<__m512i, [u32; 16]>(y) {
                unreachable_unchecked();
            }
        }
        y
    }

    /// Negate a vector of MontyField31 elements in canonical form.
    /// If the inputs are not in canonical form, the result is undefined.
    #[inline]
    #[must_use]
    pub(super) fn neg(val: __m512i) -> __m512i {
        // We want this to compile to:
        //      vptestmd  nonzero, val, val
        //      vpsubd    res{nonzero}{z}, P, val
        // throughput: 1 cyc/vec (16 els/cyc)
        // latency: 4 cyc

        // NB: This routine prioritizes throughput over latency. An alternative method would be to
        // do sub(0, val), which would result in shorter latency, but also lower throughput.

        //   If val is nonzero, then val is in {1, ..., P - 1} and P - val is in the same range. If
        // val is zero, then the result is zeroed by masking.
        unsafe {
            // Safety: If this code got compiled then AVX-512F intrinsics are available.
            let nonzero = _mm512_test_epi32_mask(val, val);
            _mm512_maskz_sub_epi32(nonzero, PACKED_MOD, val)
        }
    }
}
