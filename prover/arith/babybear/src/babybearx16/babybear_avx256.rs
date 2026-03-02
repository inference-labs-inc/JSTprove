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
unsafe fn mod_reduce_epi32(x: __m256i) -> __m256i {
    // If element >= modulus, subtract modulus
    let sub_mod = _mm256_sub_epi32(x, PACKED_MOD);
    _mm256_min_epu32(x, sub_mod)
}

const BABY_BEAR_PACK_SIZE: usize = 16;

const PACKED_0: __m256i = unsafe { transmute([0; 8]) };

// 1 in Montgomery form
const PACKED_1: __m256i = unsafe { transmute([0xffffffe; 8]) };

// 2^-1 Montgomery form
const PACKED_INV_2: __m256i = unsafe { transmute([0x7ffffff; 8]) };

const PACKED_MOD: __m256i = unsafe { transmute([BABY_BEAR_MOD; 8]) };

const PACKED_MU: __m256i = unsafe { transmute::<[u32; 8], __m256i>([0x88000001; 8]) };

#[derive(Clone, Copy)]
pub struct AVXBabyBear {
    pub v: [__m256i; 2],
}

field_common!(AVXBabyBear);

impl ExpSerde for AVXBabyBear {
    #[inline(always)]
    /// serialize self into bytes
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        let data = unsafe { transmute::<[__m256i; 2], [u8; 64]>(self.v) };
        writer.write_all(&data)?;
        Ok(())
    }

    /// deserialize bytes into field
    #[inline(always)]
    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let mut data = [0; 64];
        reader.read_exact(&mut data)?;
        unsafe {
            let value = transmute::<[u8; 64], [__m256i; 2]>(data);
            let v0 = mod_reduce_epi32(value[0]);
            let v1 = mod_reduce_epi32(value[1]);
            Ok(AVXBabyBear { v: [v0, v1] })
        }
    }
}

impl Field for AVXBabyBear {
    const NAME: &'static str = "AVXBabyBear";

    const SIZE: usize = 512 / 8;

    const ZERO: Self = Self {
        v: [PACKED_0, PACKED_0],
    };

    const ONE: Self = Self {
        v: [PACKED_1, PACKED_1],
    };

    const INV_2: Self = Self {
        v: [PACKED_INV_2, PACKED_INV_2],
    };

    const FIELD_SIZE: usize = 32;

    const MODULUS: U256 = U256([BABY_BEAR_MOD as u128, 0]);

    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn one() -> Self {
        Self {
            v: [PACKED_1, PACKED_1],
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        // value is either zero or 0x7FFFFFFF
        unsafe {
            let pcmp0 =
                _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(self.v[0], PACKED_0)));
            let pcmp1 =
                _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(self.v[1], PACKED_0)));
            let pcmp2_0 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(
                self.v[0], PACKED_MOD,
            )));
            let pcmp2_1 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(
                self.v[1], PACKED_MOD,
            )));
            (pcmp0 | pcmp1 | pcmp2_0 | pcmp2_1) == 0xFF
        }
    }

    #[inline(always)]
    fn random_unsafe(mut rng: impl RngCore) -> Self {
        // Caution: this may not produce uniformly random elements
        unsafe {
            let mut v0 = _mm256_setr_epi32(
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
            );
            let mut v1 = _mm256_setr_epi32(
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
                rng.gen::<i32>(),
            );
            v0 = mod_reduce_epi32(v0);
            v0 = mod_reduce_epi32(v0);
            v1 = mod_reduce_epi32(v1);
            v1 = mod_reduce_epi32(v1);
            Self { v: [v0, v1] }
        }
    }

    #[inline(always)]
    fn random_bool(mut rng: impl RngCore) -> Self {
        // Caution: this may not produce uniformly random elements
        unsafe {
            let v0 = _mm256_setr_epi32(
                rng.gen::<bool>() as i32,
                rng.gen::<bool>() as i32,
                rng.gen::<bool>() as i32,
                rng.gen::<bool>() as i32,
                rng.gen::<bool>() as i32,
                rng.gen::<bool>() as i32,
                rng.gen::<bool>() as i32,
                rng.gen::<bool>() as i32,
            );
            let v1 = _mm256_setr_epi32(
                rng.gen::<bool>() as i32,
                rng.gen::<bool>() as i32,
                rng.gen::<bool>() as i32,
                rng.gen::<bool>() as i32,
                rng.gen::<bool>() as i32,
                rng.gen::<bool>() as i32,
                rng.gen::<bool>() as i32,
                rng.gen::<bool>() as i32,
            );
            Self { v: [v0, v1] }
        }
    }

    #[inline(always)]
    fn inv(&self) -> Option<Self> {
        // slow, should not be used in production
        let values0 = unsafe { transmute::<__m256i, [BabyBear; 8]>(self.v[0]) };
        let values1 = unsafe { transmute::<__m256i, [BabyBear; 8]>(self.v[1]) };
        let is_non_zero = values0.iter().chain(values1.iter()).all(|x| !x.is_zero());
        if !is_non_zero {
            return None;
        }
        let inv0 = values0.iter().map(|x| x.inv().unwrap()).collect::<Vec<_>>();
        let inv1 = values1.iter().map(|x| x.inv().unwrap()).collect::<Vec<_>>();
        Some(Self {
            v: unsafe {
                [
                    transmute::<[BabyBear; 8], __m256i>(inv0.try_into().unwrap()),
                    transmute::<[BabyBear; 8], __m256i>(inv1.try_into().unwrap()),
                ]
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
            v: unsafe {
                [
                    _mm256_set1_epi32(m.value as i32),
                    _mm256_set1_epi32(m.value as i32),
                ]
            },
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
            v: unsafe {
                [
                    _mm256_set1_epi32(x.value as i32),
                    _mm256_set1_epi32(x.value as i32),
                ]
            },
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
        let ret = unsafe { transmute::<[__m256i; 2], [Self::Scalar; BABY_BEAR_PACK_SIZE]>(self.v) };
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
            _mm256_storeu_si256(data[..8].as_mut_ptr() as *mut __m256i, self.v[0]);
            _mm256_storeu_si256(data[8..].as_mut_ptr() as *mut __m256i, self.v[1]);
        }
        // if all data is the same, print only one
        if data.iter().all(|x| x == data[0]) {
            write!(
                f,
                "mm256i<16 x {}>",
                if BABY_BEAR_MOD - data[0] > 1024 {
                    format!("{}", data[0])
                } else {
                    format!("-{}", BABY_BEAR_MOD - data[0])
                }
            )
        } else {
            write!(f, "mm256i<{data:?}>")
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
            let pcmp0 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(
                mod_reduce_epi32(self.v[0]),
                mod_reduce_epi32(other.v[0]),
            )));
            let pcmp1 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(
                mod_reduce_epi32(self.v[1]),
                mod_reduce_epi32(other.v[1]),
            )));
            (pcmp0 & pcmp1) == 0xFF
        }
    }
}

impl Eq for AVXBabyBear {}

#[inline]
#[must_use]
fn add_internal(a: &AVXBabyBear, b: &AVXBabyBear) -> AVXBabyBear {
    AVXBabyBear {
        v: [
            p3_instructions::add(a.v[0], b.v[0]),
            p3_instructions::add(a.v[1], b.v[1]),
        ],
    }
}

#[inline]
#[must_use]
fn sub_internal(a: &AVXBabyBear, b: &AVXBabyBear) -> AVXBabyBear {
    AVXBabyBear {
        v: [
            p3_instructions::sub(a.v[0], b.v[0]),
            p3_instructions::sub(a.v[1], b.v[1]),
        ],
    }
}

#[inline]
#[must_use]
fn mul_internal(a: &AVXBabyBear, b: &AVXBabyBear) -> AVXBabyBear {
    let v1 = p3_instructions::mul(a.v[0], b.v[0]);
    let v1 = p3_instructions::red_signed_to_canonical(v1);
    let v2 = p3_instructions::mul(a.v[1], b.v[1]);
    let v2 = p3_instructions::red_signed_to_canonical(v2);

    AVXBabyBear { v: [v1, v2] }
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
            v: [
                p3_instructions::neg(self.v[0]),
                p3_instructions::neg(self.v[1]),
            ],
        }
    }
}

impl Hash for AVXBabyBear {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        unsafe {
            let mut data = [0; BABY_BEAR_PACK_SIZE];
            _mm256_storeu_si256(data[..8].as_mut_ptr() as *mut __m256i, self.v[0]);
            _mm256_storeu_si256(data[8..].as_mut_ptr() as *mut __m256i, self.v[1]);
            state.write(data.as_ref());
        }
    }
}

mod p3_instructions {

    use std::arch::x86_64::*;

    use super::{PACKED_MOD, PACKED_MU};

    /// Add two vectors of Monty31 field elements in canonical form.
    /// If the inputs are not in canonical form, the result is undefined.
    #[inline]
    #[must_use]
    pub(super) fn add(lhs: __m256i, rhs: __m256i) -> __m256i {
        // We want this to compile to:
        //      vpaddd   t, lhs, rhs
        //      vpsubd   u, t, P
        //      vpminud  res, t, u
        // throughput: 1 cyc/vec (8 els/cyc)
        // latency: 3 cyc

        //   Let t := lhs + rhs. We want to return t mod P. Recall that lhs and rhs are in
        // 0, ..., P - 1, so t is in 0, ..., 2 P - 2 (< 2^32). It suffices to return t if t < P and
        // t - P otherwise.
        //   Let u := (t - P) mod 2^32 and r := unsigned_min(t, u).
        //   If t is in 0, ..., P - 1, then u is in (P - 1 <) 2^32 - P, ..., 2^32 - 1 and r = t.
        // Otherwise, t is in P, ..., 2 P - 2, u is in 0, ..., P - 2 (< P) and r = u. Hence, r is t
        // if t < P and t - P otherwise, as desired.

        unsafe {
            // Safety: If this code got compiled then AVX2 intrinsics are available.
            let t = _mm256_add_epi32(lhs, rhs);
            let u = _mm256_sub_epi32(t, PACKED_MOD);
            _mm256_min_epu32(t, u)
        }
    }

    /// Subtract vectors of MontyField31 field elements in canonical form.
    /// If the inputs are not in canonical form, the result is undefined.
    #[inline]
    #[must_use]
    pub(super) fn sub(lhs: __m256i, rhs: __m256i) -> __m256i {
        // We want this to compile to:
        //      vpsubd   t, lhs, rhs
        //      vpaddd   u, t, P
        //      vpminud  res, t, u
        // throughput: 1 cyc/vec (8 els/cyc)
        // latency: 3 cyc

        unsafe {
            // Safety: If this code got compiled then AVX2 intrinsics are available.
            let t = _mm256_sub_epi32(lhs, rhs);
            red_signed_to_canonical(t)
        }
    }

    #[inline]
    #[must_use]
    fn movehdup_epi32(x: __m256i) -> __m256i {
        // This instruction is only available in the floating-point flavor; this distinction is only
        // for historical reasons and no longer matters. We cast to floats, duplicate, and
        // cast back.
        unsafe { _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(x))) }
    }

    /// Multiply unsigned vectors of field elements returning a vector of signed integers lying in
    /// (-P, P).
    ///
    /// Inputs are allowed to not be in canonical form however they must obey the bound `lhs*rhs <
    /// 2^32P`. If this bound is broken, the output is undefined.
    #[inline]
    #[must_use]
    pub(super) fn mul(lhs: __m256i, rhs: __m256i) -> __m256i {
        // We want this to compile to:
        //      vmovshdup  lhs_odd, lhs
        //      vmovshdup  rhs_odd, rhs
        //      vpmuludq   prod_evn, lhs, rhs
        //      vpmuludq   prod_odd, lhs_odd, rhs_odd
        //      vpmuludq   q_evn, prod_evn, MU
        //      vpmuludq   q_odd, prod_odd, MU
        //      vpmuludq   q_P_evn, q_evn, P
        //      vpmuludq   q_P_odd, q_odd, P
        //      vpsubq     d_evn, prod_evn, q_P_evn
        //      vpsubq     d_odd, prod_odd, q_P_odd
        //      vmovshdup  d_evn_hi, d_evn
        //      vpblendd   t, d_evn_hi, d_odd, aah
        // throughput: 4 cyc/vec (2 els/cyc)
        // latency: 19 cyc
        let lhs_evn = lhs;
        let rhs_evn = rhs;
        let lhs_odd = movehdup_epi32(lhs);
        let rhs_odd = movehdup_epi32(rhs);

        let d_evn = monty_mul(lhs_evn, rhs_evn);
        let d_odd = monty_mul(lhs_odd, rhs_odd);

        blend_evn_odd(d_evn, d_odd)
    }

    /// Negate a vector of MontyField31 elements in canonical form.
    /// If the inputs are not in canonical form, the result is undefined.
    #[inline]
    #[must_use]
    pub(super) fn neg(val: __m256i) -> __m256i {
        // We want this to compile to:
        //      vpsubd   t, P, val
        //      vpsignd  res, t, val
        // throughput: .67 cyc/vec (12 els/cyc)
        // latency: 2 cyc

        //   The vpsignd instruction is poorly named, because it doesn't _return_ or _copy_ the sign
        // of anything, but _multiplies_ x by the sign of y (treating both as signed
        // integers). In other words,
        //                       { x            if y >s 0,
        //      vpsignd(x, y) := { 0            if y = 0,
        //                       { -x mod 2^32  if y <s 0.
        //   We define t := P - val and note that t = -val (mod P). When val is in {1, ..., P - 1},
        // t is similarly in {1, ..., P - 1}, so it's in canonical form. Otherwise, val = 0
        // and t = P.   This is where we define res := vpsignd(t, val). The sign bit of val
        // is never set so either val = 0 or val >s 0. If val = 0, then res = vpsignd(t, 0)
        // = 0, as desired. Otherwise, res = vpsignd(t, val) = t passes t through.
        unsafe {
            // Safety: If this code got compiled then AVX2 intrinsics are available.
            let t = _mm256_sub_epi32(PACKED_MOD, val);
            _mm256_sign_epi32(t, val)
        }
    }

    /// Blend together in two vectors interleaving the 32-bit elements stored in the odd components.
    ///
    /// This ignores whatever is stored in even positions.
    #[inline(always)]
    #[must_use]
    fn blend_evn_odd(evn: __m256i, odd: __m256i) -> __m256i {
        // We want this to compile to:
        //      vmovshdup  evn_hi, evn
        //      vpblendd   t, evn_hi, odd, aah
        // throughput: 0.67 cyc/vec (12 els/cyc)
        // latency: 2 cyc
        unsafe {
            // We start with:
            //   evn = [ e0  e1  e2  e3  e4  e5  e6  e7 ],
            //   odd = [ o0  o1  o2  o3  o4  o5  o6  o7 ].
            let evn_hi = movehdup_epi32(evn);
            _mm256_blend_epi32(evn_hi, odd, 0b10101010)
            // res = [e1, o1, e3, o3, e5, o5, e7, o7]
        }
    }

    /// Multiply the MontyField31 field elements in the even index entries.
    /// lhs[2i], rhs[2i] must be unsigned 32-bit integers such that
    /// lhs[2i] * rhs[2i] lies in {0, ..., 2^32P}.
    /// The output will lie in {-P, ..., P} and be stored in output[2i + 1].
    #[inline]
    #[must_use]
    fn monty_mul(lhs: __m256i, rhs: __m256i) -> __m256i {
        unsafe {
            let prod = _mm256_mul_epu32(lhs, rhs);
            partial_monty_red_unsigned_to_signed(prod)
        }
    }

    /// Given a vector of signed field elements, return a vector of elements in canonical form.
    ///
    /// Inputs must be signed 32-bit integers lying in (-P, ..., P). If they do not lie in
    /// this range, the output is undefined.
    #[inline(always)]
    #[must_use]
    pub(super) fn red_signed_to_canonical(input: __m256i) -> __m256i {
        unsafe {
            // We want this to compile to:
            //      vpaddd     corr, input, P
            //      vpminud    res, input, corr
            // throughput: 0.67 cyc/vec (12 els/cyc)
            // latency: 2 cyc

            // We want to return input mod P where input lies in (-2^31 <) -P + 1, ..., P - 1 (<
            // 2^31). It suffices to return input if input >= 0 and input + P otherwise.
            //
            // Let corr := (input + P) mod 2^32 and res := unsigned_min(input, corr).
            // If input is in 0, ..., P - 1, then corr is in P, ..., 2 P - 1 and res = input.
            // Otherwise, input is in -P + 1, ..., -1; corr is in 1, ..., P - 1 (< P) and res =
            // corr. Hence, res is input if input < P and input + P otherwise, as
            // desired.
            let corr = _mm256_add_epi32(input, PACKED_MOD);
            _mm256_min_epu32(input, corr)
        }
    }

    // MONTGOMERY MULTIPLICATION
    //   This implementation is based on [1] but with minor changes. The reduction is as follows:
    //
    // Constants: P < 2^31, prime
    //            B = 2^32
    //            μ = P^-1 mod B
    // Input: 0 <= C < P B
    // Output: 0 <= R < P such that R = C B^-1 (mod P)
    //   1. Q := μ C mod B
    //   2. D := (C - Q P) / B
    //   3. R := if D < 0 then D + P else D
    //
    // We first show that the division in step 2. is exact. It suffices to show that C = Q P (mod
    // B). By definition of Q and μ, we have Q P = μ C P = P^-1 C P = C (mod B). We also have
    // C - Q P = C (mod P), so thus D = C B^-1 (mod P).
    //
    // It remains to show that R is in the correct range. It suffices to show that -P < D < P. We
    // know that 0 <= C < P B and 0 <= Q P < P B. Then -P B < C - QP < P B and -P < D < P, as
    // desired.
    //
    // [1] Modern Computer Arithmetic, Richard Brent and Paul Zimmermann, Cambridge University
    // Press,     2010, algorithm 2.7.

    // We provide 2 variants of Montgomery reduction depending on if the inputs are unsigned or
    // signed. The unsigned variant follows steps 1 and 2 in the above protocol to produce D in
    // (-P, ..., P). For the signed variant we assume -PB/2 < C < PB/2 and let Q := μ C mod B be
    // the unique representative in [-B/2, ..., B/2 - 1]. The division in step 2 is clearly
    // still exact and |C - Q P| <= |C| + |Q||P| < PB so D still lies in (-P, ..., P).

    /// Perform a partial Montgomery reduction on each 64 bit element.
    /// Input must lie in {0, ..., 2^32P}.
    /// The output will lie in {-P, ..., P} and be stored in the upper 32 bits.
    #[inline]
    #[must_use]
    fn partial_monty_red_unsigned_to_signed(input: __m256i) -> __m256i {
        unsafe {
            let q = _mm256_mul_epu32(input, PACKED_MU);
            let q_p = _mm256_mul_epu32(q, PACKED_MOD);

            // By construction, the bottom 32 bits of input and q_p are equal.
            // Thus _mm256_sub_epi32 and _mm256_sub_epi64 should act identically.
            // However for some reason, the compiler gets confused if we use _mm256_sub_epi64
            // and outputs a load of nonsense, see: https://godbolt.org/z/3W8M7Tv84.
            _mm256_sub_epi32(input, q_p)
        }
    }
}
