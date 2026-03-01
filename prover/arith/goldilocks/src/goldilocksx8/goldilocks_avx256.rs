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

/// Number of Goldilocks elements in [__m256i; 2] elements
const GOLDILOCKS_PACK_SIZE: usize = 8;

/// Packed field order
const PACKED_GOLDILOCKS_MOD: __m256i = unsafe { transmute([GOLDILOCKS_MOD; 4]) };

const SHIFTED_PACKED_GOLDILOCKS_MOD: __m256i =
    unsafe { transmute([GOLDILOCKS_MOD ^ (i64::MIN as u64); 4]) };

/// Packed epsilon (i.e., 2^64 % modulus)
const PACKED_EPSILON: __m256i = unsafe { transmute([EPSILON; 4]) };

/// Packed zero
const PACKED_0: __m256i = unsafe { transmute([0u64; 4]) };

/// Packed inverse of 2
const PACKED_INV_2: __m256i = unsafe { transmute([0x7FFFFFFF80000001u64; 4]) };

#[derive(Debug, Clone, Copy)]
pub struct AVXGoldilocks {
    // using two __m256i to simulate a __m512i
    pub v: [__m256i; 2],
}

field_common!(AVXGoldilocks);

impl ExpSerde for AVXGoldilocks {
    #[inline(always)]
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        let data0 = unsafe { transmute::<__m256i, [u8; 32]>(mod_reduce_epi64(self.v[0])) };
        let data1 = unsafe { transmute::<__m256i, [u8; 32]>(mod_reduce_epi64(self.v[1])) };
        writer.write_all(&data0)?;
        writer.write_all(&data1)?;
        Ok(())
    }

    #[inline(always)]
    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let mut data = [0u8; 64];
        reader.read_exact(&mut data)?;
        Ok(AVXGoldilocks {
            v: unsafe { transmute::<[u8; 64], [__m256i; 2]>(data) },
        })
    }
}

impl Field for AVXGoldilocks {
    const NAME: &'static str = "AVXGoldilocks";

    const SIZE: usize = 512 / 8;

    const ZERO: Self = Self {
        v: [PACKED_0, PACKED_0],
    };

    const ONE: Self = Self {
        v: [unsafe { transmute::<[u64; 4], __m256i>([1; 4]) }, unsafe {
            transmute::<[u64; 4], __m256i>([1; 4])
        }],
    };

    const INV_2: Self = Self {
        v: [PACKED_INV_2, PACKED_INV_2],
    };

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
        let eq1 = unsafe {
            // Replace _mm256_cmpeq_epi64_mask with _mm256_cmpeq_epi64
            let pcmp_zero = _mm256_cmpeq_epi64(self.v[0], PACKED_0);
            let pcmp_mod = _mm256_cmpeq_epi64(self.v[0], PACKED_GOLDILOCKS_MOD);
            // Combine the results using OR
            let result = _mm256_or_si256(pcmp_zero, pcmp_mod);
            // Check if all bits are set (equivalent to mask == 0xF)
            _mm256_testc_si256(result, _mm256_set1_epi64x(-1)) != 0
        };

        let eq2 = unsafe {
            let pcmp_zero = _mm256_cmpeq_epi64(self.v[1], PACKED_0);
            let pcmp_mod = _mm256_cmpeq_epi64(self.v[1], PACKED_GOLDILOCKS_MOD);
            let result = _mm256_or_si256(pcmp_zero, pcmp_mod);
            _mm256_testc_si256(result, _mm256_set1_epi64x(-1)) != 0
        };

        eq1 && eq2
    }

    #[inline(always)]
    fn random_unsafe(mut rng: impl RngCore) -> Self {
        unsafe {
            let mut v0 = _mm256_setr_epi64x(
                rng.gen::<i64>(),
                rng.gen::<i64>(),
                rng.gen::<i64>(),
                rng.gen::<i64>(),
            );
            let mut v1 = _mm256_setr_epi64x(
                rng.gen::<i64>(),
                rng.gen::<i64>(),
                rng.gen::<i64>(),
                rng.gen::<i64>(),
            );
            v0 = mod_reduce_epi64(v0);
            v1 = mod_reduce_epi64(v1);
            Self { v: [v0, v1] }
        }
    }

    #[inline(always)]
    fn random_bool(mut rng: impl RngCore) -> Self {
        unsafe {
            let v0 = _mm256_setr_epi64x(
                rng.gen::<bool>() as i64,
                rng.gen::<bool>() as i64,
                rng.gen::<bool>() as i64,
                rng.gen::<bool>() as i64,
            );
            let v1 = _mm256_setr_epi64x(
                rng.gen::<bool>() as i64,
                rng.gen::<bool>() as i64,
                rng.gen::<bool>() as i64,
                rng.gen::<bool>() as i64,
            );
            Self { v: [v0, v1] }
        }
    }

    #[inline(always)]
    fn square(&self) -> Self {
        let (hi0, lo0) = unsafe { p3_instructions::square64(self.v[0]) };
        let (hi1, lo1) = unsafe { p3_instructions::square64(self.v[1]) };
        AVXGoldilocks {
            v: [unsafe { p3_instructions::reduce128((hi0, lo0)) }, unsafe {
                p3_instructions::reduce128((hi1, lo1))
            }],
        }
    }

    #[inline(always)]
    fn inv(&self) -> Option<Self> {
        // slow, should not be used in production
        let mut goldilocks_vec1 = unsafe { transmute::<__m256i, [Goldilocks; 4]>(self.v[0]) };
        let is_non_zero = goldilocks_vec1.iter().all(|x| !x.is_zero());
        if !is_non_zero {
            return None;
        }
        let mut goldilocks_vec2 = unsafe { transmute::<__m256i, [Goldilocks; 4]>(self.v[1]) };
        let is_non_zero = goldilocks_vec2.iter().all(|x| !x.is_zero());
        if !is_non_zero {
            return None;
        }

        goldilocks_vec1
            .iter_mut()
            .for_each(|x| *x = x.inv().unwrap()); // safe unwrap
        goldilocks_vec2
            .iter_mut()
            .for_each(|x| *x = x.inv().unwrap()); // safe unwrap

        let v0 = unsafe { transmute::<[Goldilocks; 4], __m256i>(goldilocks_vec1) };
        let v1 = unsafe { transmute::<[Goldilocks; 4], __m256i>(goldilocks_vec2) };
        Some(Self { v: [v0, v1] })
    }

    #[inline(always)]
    fn as_u32_unchecked(&self) -> u32 {
        unimplemented!("self is a vector, cannot convert to u32")
    }

    #[inline(always)]
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        let m = Goldilocks::from_uniform_bytes(bytes);
        Self {
            v: [unsafe { _mm256_set1_epi64x(m.v as i64) }, unsafe {
                _mm256_set1_epi64x(m.v as i64)
            }],
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
                v: [
                    _mm256_set1_epi64x(x.v as i64),
                    _mm256_set1_epi64x(x.v as i64),
                ],
            }
        }
    }

    #[inline(always)]
    fn pack(base_vec: &[Self::Scalar]) -> Self {
        assert_eq!(base_vec.len(), Self::PACK_SIZE);
        let ret: [Self::Scalar; Self::PACK_SIZE] = base_vec.try_into().unwrap();
        let v0 = unsafe { transmute::<[Self::Scalar; 4], __m256i>(ret[..4].try_into().unwrap()) };
        let v1 = unsafe { transmute::<[Self::Scalar; 4], __m256i>(ret[4..].try_into().unwrap()) };
        Self { v: [v0, v1] }
    }

    #[inline(always)]
    fn unpack(&self) -> Vec<Self::Scalar> {
        let ret0 = unsafe { transmute::<__m256i, [Self::Scalar; 4]>(self.v[0]) };
        let ret1 = unsafe { transmute::<__m256i, [Self::Scalar; 4]>(self.v[1]) };
        let mut ret = Vec::with_capacity(Self::PACK_SIZE);
        ret.extend_from_slice(&ret0);
        ret.extend_from_slice(&ret1);
        ret
    }

    #[inline(always)]
    fn horizontal_sum(&self) -> Self::Scalar {
        let mut temp: u128 = 0;
        let vars0 = unsafe { transmute::<__m256i, [Self::Scalar; 4]>(self.v[0]) };
        let vars1 = unsafe { transmute::<__m256i, [Self::Scalar; 4]>(self.v[1]) };
        vars0.iter().for_each(|c| temp += c.v as u128);
        vars1.iter().for_each(|c| temp += c.v as u128);

        p2_instructions::reduce128(temp)
    }
}

impl From<Goldilocks> for AVXGoldilocks {
    #[inline(always)]
    fn from(x: Goldilocks) -> Self {
        Self {
            v: [unsafe { _mm256_set1_epi64x(x.v as i64) }, unsafe {
                _mm256_set1_epi64x(x.v as i64)
            }],
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
            let v0_eq =
                _mm256_cmpeq_epi64(mod_reduce_epi64(self.v[0]), mod_reduce_epi64(other.v[0]));

            let v1_eq =
                _mm256_cmpeq_epi64(mod_reduce_epi64(self.v[1]), mod_reduce_epi64(other.v[1]));

            // Check if all elements are equal
            _mm256_testc_si256(v0_eq, _mm256_set1_epi64x(-1)) != 0
                && _mm256_testc_si256(v1_eq, _mm256_set1_epi64x(-1)) != 0
        }
    }
}

impl Eq for AVXGoldilocks {}

impl Mul<&Goldilocks> for AVXGoldilocks {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: &Goldilocks) -> Self::Output {
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
            state.write(transmute::<__m256i, [u8; 32]>(self.v[0]).as_ref());
            state.write(transmute::<__m256i, [u8; 32]>(self.v[1]).as_ref());
        }
    }
}

impl Neg for AVXGoldilocks {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self {
            v: [unsafe { p3_instructions::neg(self.v[0]) }, unsafe {
                p3_instructions::neg(self.v[1])
            }],
        }
    }
}

impl From<u32> for AVXGoldilocks {
    #[inline(always)]
    fn from(x: u32) -> Self {
        Self {
            v: [
                unsafe { p3_instructions::shift(_mm256_set1_epi64x(x as i64)) },
                unsafe { p3_instructions::shift(_mm256_set1_epi64x(x as i64)) },
            ],
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

    fn root_of_unity() -> Self {
        Self {
            v: [unsafe { _mm256_set1_epi64x(0x185629dcda58878c) }, unsafe {
                _mm256_set1_epi64x(0x185629dcda58878c)
            }],
        }
    }
}

#[inline]
unsafe fn mod_reduce_epi64(x: __m256i) -> __m256i {
    unsafe {
        // Use the shift trick to convert unsigned comparison to signed comparison
        let x_s = p3_instructions::shift(x);
        let mod_s = SHIFTED_PACKED_GOLDILOCKS_MOD;
        // Use _mm256_cmpgt_epi64 instead of mask-returning version
        let mask = _mm256_cmpgt_epi64(mod_s, x_s);
        // If mask is 0 (x >= mod), subtract mod; if mask is -1 (x < mod), subtract 0
        let to_sub = _mm256_andnot_si256(mask, PACKED_GOLDILOCKS_MOD);
        _mm256_sub_epi64(x, to_sub)
    }
}

#[inline(always)]
fn add_internal(a: &AVXGoldilocks, b: &AVXGoldilocks) -> AVXGoldilocks {
    let v0 = unsafe { p3_instructions::add(a.v[0], b.v[0]) };
    let v1 = unsafe { p3_instructions::add(a.v[1], b.v[1]) };
    AVXGoldilocks { v: [v0, v1] }
}

#[inline(always)]
fn sub_internal(a: &AVXGoldilocks, b: &AVXGoldilocks) -> AVXGoldilocks {
    let v0 = unsafe { p3_instructions::sub(a.v[0], b.v[0]) };
    let v1 = unsafe { p3_instructions::sub(a.v[1], b.v[1]) };
    AVXGoldilocks { v: [v0, v1] }
}

#[inline]
fn mul_internal(x: &AVXGoldilocks, y: &AVXGoldilocks) -> AVXGoldilocks {
    let (hi0, lo0) = unsafe { p3_instructions::mul64_64(x.v[0], y.v[0]) };
    let (hi1, lo1) = unsafe { p3_instructions::mul64_64(x.v[1], y.v[1]) };
    AVXGoldilocks {
        v: [unsafe { p3_instructions::reduce128((hi0, lo0)) }, unsafe {
            p3_instructions::reduce128((hi1, lo1))
        }],
    }
}

/// instructions adopted from Plonky3 https://github.com/Plonky3/Plonky3/blob/main/goldilocks/src/x86_64_avx2/packing.rs
mod p3_instructions {
    use super::*;

    // Resources:
    // 1. Intel Intrinsics Guide for explanation of each intrinsic: https://software.intel.com/sites/landingpage/IntrinsicsGuide/
    // 2. uops.info lists micro-ops for each instruction: https://uops.info/table.html
    // 3. Intel optimization manual for introduction to x86 vector extensions and best practices: https://software.intel.com/content/www/us/en/develop/download/intel-64-and-ia-32-architectures-optimization-reference-manual.html

    // Preliminary knowledge:
    // 1. Vector code usually avoids branching. Instead of branches, we can do input selection with
    //    _mm256_blendv_epi8 or similar instruction. If all we're doing is conditionally zeroing a
    //    vector element then _mm256_and_si256 or _mm256_andnot_si256 may be used and are cheaper.
    //
    // 2. AVX does not support addition with carry but 128-bit (2-word) addition can be easily
    //    emulated. The method recognizes that for a + b overflowed iff (a + b) < a: i. res_lo =
    //    a_lo + b_lo ii. carry_mask = res_lo < a_lo iii. res_hi = a_hi + b_hi - carry_mask Notice
    //    that carry_mask is subtracted, not added. This is because AVX comparison instructions
    //    return -1 (all bits 1) for true and 0 for false.
    //
    // 3. AVX does not have unsigned 64-bit comparisons. Those can be emulated with signed
    //    comparisons by recognizing that a <u b iff a + (1 << 63) <s b + (1 << 63), where the
    //    addition wraps around and the comparisons are unsigned and signed respectively. The shift
    //    function adds/subtracts 1 << 63 to enable this trick. Example: addition with carry. i.
    //    a_lo_s = shift(a_lo) ii. res_lo_s = a_lo_s + b_lo iii. carry_mask = res_lo_s <s a_lo_s iv.
    //    res_lo = shift(res_lo_s) v. res_hi = a_hi + b_hi - carry_mask The suffix _s denotes a
    //    value that has been shifted by 1 << 63. The result of addition is shifted if exactly one
    //    of the operands is shifted, as is the case on line ii. Line iii. performs a signed
    //    comparison res_lo_s <s a_lo_s on shifted values to emulate unsigned comparison res_lo <u
    //    a_lo on unshifted values. Finally, line iv. reverses the shift so the result can be
    //    returned. When performing a chain of calculations, we can often save instructions by
    //    letting the shift propagate through and only undoing it when necessary. For example, to
    //    compute the addition of three two-word (128-bit) numbers we can do: i. a_lo_s =
    //    shift(a_lo) ii. tmp_lo_s = a_lo_s + b_lo iii. tmp_carry_mask = tmp_lo_s <s a_lo_s iv.
    //    tmp_hi = a_hi + b_hi - tmp_carry_mask v. res_lo_s = tmp_lo_s + c_lo vi. res_carry_mask =
    //    res_lo_s <s tmp_lo_s vii. res_lo = shift(res_lo_s) viii. res_hi = tmp_hi + c_hi -
    //    res_carry_mask Notice that the above 3-value addition still only requires two calls to
    //    shift, just like our 2-value addition.

    const SIGN_BIT: __m256i = unsafe { transmute([i64::MIN; 4]) };

    /// Add 2^63 with overflow. Needed to emulate unsigned comparisons (see point 3. in
    /// packed_prime_field.rs).
    ///  # Safety
    /// TODO
    #[inline]
    pub(super) unsafe fn shift(x: __m256i) -> __m256i {
        unsafe { _mm256_xor_si256(x, SIGN_BIT) }
    }

    /// Convert to canonical representation.
    /// The argument is assumed to be shifted by 1 << 63 (i.e. x_s = x + 1<<63, where x is the field
    ///   value). The returned value is similarly shifted by 1 << 63 (i.e. we return y_s = y +
    /// (1<<63),   where 0 <= y < FIELD_ORDER).
    #[inline]
    pub(super) unsafe fn canonicalize_s(x_s: __m256i) -> __m256i {
        unsafe {
            // If x >= FIELD_ORDER then corresponding mask bits are all 0; otherwise all 1.
            let mask = _mm256_cmpgt_epi64(SHIFTED_PACKED_GOLDILOCKS_MOD, x_s);
            // wrapback_amt is -FIELD_ORDER if mask is 0; otherwise 0.
            let wrapback_amt = _mm256_andnot_si256(mask, PACKED_EPSILON);
            _mm256_add_epi64(x_s, wrapback_amt)
        }
    }

    /// Addition u64 + u64 -> u64. Assumes that x + y < 2^64 + FIELD_ORDER. The second argument is
    /// pre-shifted by 1 << 63. The result is similarly shifted.
    #[inline]
    unsafe fn add_no_double_overflow_64_64s_s(x: __m256i, y_s: __m256i) -> __m256i {
        unsafe {
            let res_wrapped_s = _mm256_add_epi64(x, y_s);
            let mask = _mm256_cmpgt_epi64(y_s, res_wrapped_s); // -1 if overflowed else 0.
            let wrapback_amt = _mm256_srli_epi64::<32>(mask); // -FIELD_ORDER if overflowed else 0.
            _mm256_add_epi64(res_wrapped_s, wrapback_amt)
        }
    }

    #[inline]
    pub(super) unsafe fn add(x: __m256i, y: __m256i) -> __m256i {
        unsafe {
            let y_s = shift(y);
            let res_s = add_no_double_overflow_64_64s_s(x, canonicalize_s(y_s));
            shift(res_s)
        }
    }

    #[inline]
    pub(super) unsafe fn sub(x: __m256i, y: __m256i) -> __m256i {
        unsafe {
            let mut y_s = shift(y);
            y_s = canonicalize_s(y_s);
            let x_s = shift(x);
            let mask = _mm256_cmpgt_epi64(y_s, x_s); // -1 if sub will underflow (y > x) else 0.
            let wrapback_amt = _mm256_srli_epi64::<32>(mask); // -FIELD_ORDER if underflow else 0.
            let res_wrapped = _mm256_sub_epi64(x_s, y_s);
            _mm256_sub_epi64(res_wrapped, wrapback_amt)
        }
    }

    #[inline]
    pub(super) unsafe fn neg(y: __m256i) -> __m256i {
        unsafe {
            let y_s = shift(y);
            _mm256_sub_epi64(SHIFTED_PACKED_GOLDILOCKS_MOD, canonicalize_s(y_s))
        }
    }

    /// Full 64-bit by 64-bit multiplication. This emulated multiplication is 1.33x slower than the
    /// scalar instruction, but may be worth it if we want our data to live in vector registers.
    #[inline]
    pub(super) unsafe fn mul64_64(x: __m256i, y: __m256i) -> (__m256i, __m256i) {
        unsafe {
            // We want to move the high 32 bits to the low position. The multiplication instruction
            // ignores the high 32 bits, so it's ok to just duplicate it into the low
            // position. This duplication can be done on port 5; bitshifts run on ports
            // 0 and 1, competing with multiplication.   This instruction is only
            // provided for 32-bit floats, not integers. Idk why Intel makes the
            // distinction; the casts are free and it guarantees that the exact bit pattern is
            // preserved. Using a swizzle instruction of the wrong domain (float vs int)
            // does not increase latency since Haswell.
            let x_hi = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(x)));
            let y_hi = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(y)));

            // All four pairwise multiplications
            let mul_ll = _mm256_mul_epu32(x, y);
            let mul_lh = _mm256_mul_epu32(x, y_hi);
            let mul_hl = _mm256_mul_epu32(x_hi, y);
            let mul_hh = _mm256_mul_epu32(x_hi, y_hi);

            // Bignum addition
            // Extract high 32 bits of mul_ll and add to mul_hl. This cannot overflow.
            let mul_ll_hi = _mm256_srli_epi64::<32>(mul_ll);
            let t0 = _mm256_add_epi64(mul_hl, mul_ll_hi);
            // Extract low 32 bits of t0 and add to mul_lh. Again, this cannot overflow.
            // Also, extract high 32 bits of t0 and add to mul_hh.
            let t0_lo = _mm256_and_si256(t0, PACKED_EPSILON);
            let t0_hi = _mm256_srli_epi64::<32>(t0);
            let t1 = _mm256_add_epi64(mul_lh, t0_lo);
            let t2 = _mm256_add_epi64(mul_hh, t0_hi);
            // Lastly, extract the high 32 bits of t1 and add to t2.
            let t1_hi = _mm256_srli_epi64::<32>(t1);
            let res_hi = _mm256_add_epi64(t2, t1_hi);

            // Form res_lo by combining the low half of mul_ll with the low half of t1 (shifted into
            // high position).
            let t1_lo = _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(t1)));
            let res_lo = _mm256_blend_epi32::<0xaa>(mul_ll, t1_lo);

            (res_hi, res_lo)
        }
    }

    /// Full 64-bit squaring. This routine is 1.2x faster than the scalar instruction.
    #[inline]
    pub(super) unsafe fn square64(x: __m256i) -> (__m256i, __m256i) {
        unsafe {
            // Get high 32 bits of x. See comment in mul64_64_s.
            let x_hi = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(x)));

            // All pairwise multiplications.
            let mul_ll = _mm256_mul_epu32(x, x);
            let mul_lh = _mm256_mul_epu32(x, x_hi);
            let mul_hh = _mm256_mul_epu32(x_hi, x_hi);

            // Bignum addition, but mul_lh is shifted by 33 bits (not 32).
            let mul_ll_hi = _mm256_srli_epi64::<33>(mul_ll);
            let t0 = _mm256_add_epi64(mul_lh, mul_ll_hi);
            let t0_hi = _mm256_srli_epi64::<31>(t0);
            let res_hi = _mm256_add_epi64(mul_hh, t0_hi);

            // Form low result by adding the mul_ll and the low 31 bits of mul_lh (shifted to the
            // high position).
            let mul_lh_lo = _mm256_slli_epi64::<33>(mul_lh);
            let res_lo = _mm256_add_epi64(mul_ll, mul_lh_lo);

            (res_hi, res_lo)
        }
    }

    /// Goldilocks addition of a "small" number. `x_s` is pre-shifted by 2**63. `y` is assumed to be
    /// <= `0xffffffff00000000`. The result is shifted by 2**63.
    #[inline]
    unsafe fn add_small_64s_64_s(x_s: __m256i, y: __m256i) -> __m256i {
        unsafe {
            let res_wrapped_s = _mm256_add_epi64(x_s, y);
            // 32-bit compare is faster than 64-bit. It's safe as long as x > res_wrapped iff x >>
            // 32 > res_wrapped >> 32. The case of x >> 32 > res_wrapped >> 32 is
            // trivial and so is <. The case where x >> 32 = res_wrapped >> 32 remains.
            // If x >> 32 = res_wrapped >> 32, then y >> 32 = 0xffffffff and the
            // addition of the low 32 bits generated a carry. This can never occur if y
            // <= 0xffffffff00000000: if y >> 32 = 0xffffffff, then no carry can occur.
            let mask = _mm256_cmpgt_epi32(x_s, res_wrapped_s); // -1 if overflowed else 0.
                                                               // The mask contains 0xffffffff in the high 32 bits if wraparound occurred and 0
                                                               // otherwise.
            let wrapback_amt = _mm256_srli_epi64::<32>(mask); // -FIELD_ORDER if overflowed else 0.
            _mm256_add_epi64(res_wrapped_s, wrapback_amt)
        }
    }

    /// Goldilocks subtraction of a "small" number. `x_s` is pre-shifted by 2**63. `y` is assumed to
    /// be <= `0xffffffff00000000`. The result is shifted by 2**63.
    #[inline]
    unsafe fn sub_small_64s_64_s(x_s: __m256i, y: __m256i) -> __m256i {
        unsafe {
            let res_wrapped_s = _mm256_sub_epi64(x_s, y);
            // 32-bit compare is faster than 64-bit. It's safe as long as res_wrapped > x iff
            // res_wrapped >> 32 > x >> 32. The case of res_wrapped >> 32 > x >> 32 is
            // trivial and so is <. The case where res_wrapped >> 32 = x >> 32 remains.
            // If res_wrapped >> 32 = x >> 32, then y >> 32 = 0xffffffff and the
            // subtraction of the low 32 bits generated a borrow. This can never occur if
            // y <= 0xffffffff00000000: if y >> 32 = 0xffffffff, then no borrow can occur.
            let mask = _mm256_cmpgt_epi32(res_wrapped_s, x_s); // -1 if underflowed else 0.
                                                               // The mask contains 0xffffffff in the high 32 bits if wraparound occurred and 0
                                                               // otherwise.
            let wrapback_amt = _mm256_srli_epi64::<32>(mask); // -FIELD_ORDER if underflowed else 0.
            _mm256_sub_epi64(res_wrapped_s, wrapback_amt)
        }
    }

    #[inline]
    pub(super) unsafe fn reduce128(x: (__m256i, __m256i)) -> __m256i {
        unsafe {
            let (hi0, lo0) = x;
            let lo0_s = shift(lo0);
            let hi_hi0 = _mm256_srli_epi64::<32>(hi0);
            let lo1_s = sub_small_64s_64_s(lo0_s, hi_hi0);
            let t1 = _mm256_mul_epu32(hi0, PACKED_EPSILON);
            let lo2_s = add_small_64s_64_s(lo1_s, t1);
            shift(lo2_s)
        }
    }
}
