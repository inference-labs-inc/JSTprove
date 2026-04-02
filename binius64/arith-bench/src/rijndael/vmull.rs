// Copyright 2025 Irreducible Inc.
//! GF(2^8) Rijndael field multiplication using aarch64 NEON `vmull_p8` instructions.
//!
//! This implements the same algorithm as `binius-field`'s `packed_aes_16x8b_multiply`,
//! using polynomial multiplication followed by Barrett reduction.

use std::arch::aarch64::*;

/// Multiply packed GF(2^8) elements in the Rijndael field.
///
/// Performs 16 parallel multiplications of 8-bit elements using the AES reduction
/// polynomial x^8 + x^4 + x^3 + x + 1.
#[inline]
pub fn mul(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    unsafe {
        let a = vreinterpretq_p8_u64(a);
        let b = vreinterpretq_p8_u64(b);

        // Carryless multiply lower and upper halves
        let c0 = vreinterpretq_p8_p16(vmull_p8(vget_low_p8(a), vget_low_p8(b)));
        let c1 = vreinterpretq_p8_p16(vmull_p8(vget_high_p8(a), vget_high_p8(b)));

        // Barrett reduction using equation 22 from
        // https://www.intel.com/content/dam/develop/external/us/en/documents/clmul-wp-rev-2-02-2014-04-20.pdf

        // q+(x)/x = (x^8 + x^4 + x^3 + x)/x = 0x8d
        const QPLUS_RSH1: poly8x8_t = unsafe { std::mem::transmute(0x8d8d8d8d8d8d8d8d_u64) };
        // q*(x) = x^4 + x^3 + x + 1 = 0x1b
        const QSTAR: poly8x8_t = unsafe { std::mem::transmute(0x1b1b1b1b1b1b1b1b_u64) };

        // Deinterleave to get low and high bytes of each 16-bit product
        let cl = vuzp1q_p8(c0, c1);
        let ch = vuzp2q_p8(c0, c1);

        // First reduction step: multiply high bytes by q+(x)/x
        let tmp0 = vmull_p8(vget_low_p8(ch), QPLUS_RSH1);
        let tmp1 = vmull_p8(vget_high_p8(ch), QPLUS_RSH1);

        // Correct for q+(x) having been divided by x
        let tmp0 = vreinterpretq_p8_u16(vshlq_n_u16(vreinterpretq_u16_p16(tmp0), 1));
        let tmp1 = vreinterpretq_p8_u16(vshlq_n_u16(vreinterpretq_u16_p16(tmp1), 1));

        // Second reduction step: multiply by q*(x)
        let tmp_hi = vuzp2q_p8(tmp0, tmp1);
        let tmp0 = vreinterpretq_p8_p16(vmull_p8(vget_low_p8(tmp_hi), QSTAR));
        let tmp1 = vreinterpretq_p8_p16(vmull_p8(vget_high_p8(tmp_hi), QSTAR));
        let tmp_lo = vuzp1q_p8(tmp0, tmp1);

        // Final XOR to combine
        vreinterpretq_u64_p8(vaddq_p8(cl, tmp_lo))
    }
}
