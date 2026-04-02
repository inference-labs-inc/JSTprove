// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::arch::aarch64::*;

use super::m128::M128;

#[inline]
pub fn packed_aes_16x8b_invert_or_zero(x: M128) -> M128 {
    lookup_16x8b(AES_INVERT_OR_ZERO_LOOKUP_TABLE, x)
}

#[inline]
pub fn packed_aes_16x8b_square(x: M128) -> M128 {
    // Freshman's dream: in GF(2^n), (a0 + a1*x + ... + a7*x^7)^2
    // = a0 + a1*x^2 + a2*x^4 + ... + a7*x^14
    // This is just bit-spreading (interleave with zeros), then reducing mod the AES polynomial.
    //
    // Split each byte into low nibble (bits 3..0) and high nibble (bits 7..4):
    // - Low nibble spread gives degree < 8, no reduction needed
    // - High nibble maps to x^8..x^14 terms, reduced via a 16-entry table
    unsafe {
        let x: uint8x16_t = x.into();

        // Nibble-to-spread table: maps 4-bit nibble abcd -> 0a0b0c0d (8-bit spread)
        let spread_tbl = vld1q_u8(
            [
                0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15, 0x40, 0x41, 0x44, 0x45, 0x50, 0x51,
                0x54, 0x55,
            ]
            .as_ptr(),
        );

        // Reduction table: maps high nibble (a7,a6,a5,a4) to the reduction of
        // a4*x^8 + a5*x^10 + a6*x^12 + a7*x^14 mod (x^8 + x^4 + x^3 + x + 1)
        let reduce_tbl = vld1q_u8(
            [
                0x00, 0x1B, 0x6C, 0x77, 0xAB, 0xB0, 0xC7, 0xDC, 0x9A, 0x81, 0xF6, 0xED, 0x31, 0x2A,
                0x5D, 0x46,
            ]
            .as_ptr(),
        );

        let lo = vandq_u8(x, vdupq_n_u8(0x0F));

        let hi = vshrq_n_u8(x, 4);

        let spread_lo = vqtbl1q_u8(spread_tbl, lo);

        let reduced_hi = vqtbl1q_u8(reduce_tbl, hi);

        veorq_u8(spread_lo, reduced_hi).into()
    }
}

#[inline]
pub fn packed_aes_16x8b_mul_alpha(x: M128) -> M128 {
    // 0xD3 corresponds to 0x10 after isomorphism from BinaryField8b to AESField
    packed_aes_16x8b_multiply(x, M128::from_le_bytes([0xD3; 16]))
}

#[inline]
pub fn packed_aes_16x8b_multiply(a: M128, b: M128) -> M128 {
    //! Performs a multiplication in GF(2^8) on the packed bytes.
    //! See <https://doc.rust-lang.org/beta/core/arch/x86_64/fn._mm_gf2p8mul_epi8.html>
    unsafe {
        let a = vreinterpretq_p8_p128(a.into());
        let b = vreinterpretq_p8_p128(b.into());
        let c0 = vreinterpretq_p8_p16(vmull_p8(vget_low_p8(a), vget_low_p8(b)));
        let c1 = vreinterpretq_p8_p16(vmull_p8(vget_high_p8(a), vget_high_p8(b)));

        // Reduces the 16-bit output of a carryless multiplication to 8 bits using equation 22 in
        // https://www.intel.com/content/dam/develop/external/us/en/documents/clmul-wp-rev-2-02-2014-04-20.pdf

        // Since q+(x) doesn't fit into 8 bits, we right shift the polynomial (divide by x) and
        // correct for this later. This works because q+(x) is divisible by x/the last polynomial
        // bit is 0. q+(x)/x = (x^8 + x^4 + x^3 + x)/x = 0b100011010 >> 1 = 0b10001101 = 0x8d
        const QPLUS_RSH1: poly8x8_t = unsafe { std::mem::transmute(0x8d8d8d8d8d8d8d8d_u64) };

        // q*(x) = x^4 + x^3 + x + 1 = 0b00011011 = 0x1b
        const QSTAR: poly8x8_t = unsafe { std::mem::transmute(0x1b1b1b1b1b1b1b1b_u64) };

        let cl = vuzp1q_p8(c0, c1);
        let ch = vuzp2q_p8(c0, c1);

        let tmp0 = vmull_p8(vget_low_p8(ch), QPLUS_RSH1);
        let tmp1 = vmull_p8(vget_high_p8(ch), QPLUS_RSH1);

        // Correct for q+(x) having been divided by x
        let tmp0 = vreinterpretq_p8_u16(vshlq_n_u16(vreinterpretq_u16_p16(tmp0), 1));
        let tmp1 = vreinterpretq_p8_u16(vshlq_n_u16(vreinterpretq_u16_p16(tmp1), 1));

        let tmp_hi = vuzp2q_p8(tmp0, tmp1);
        let tmp0 = vreinterpretq_p8_p16(vmull_p8(vget_low_p8(tmp_hi), QSTAR));
        let tmp1 = vreinterpretq_p8_p16(vmull_p8(vget_high_p8(tmp_hi), QSTAR));
        let tmp_lo = vuzp1q_p8(tmp0, tmp1);

        vreinterpretq_p128_p8(vaddq_p8(cl, tmp_lo)).into()
    }
}

#[inline]
pub fn lookup_16x8b(table: [u8; 256], x: M128) -> M128 {
    unsafe {
        let table: [uint8x16x4_t; 4] = std::mem::transmute(table);
        let x = x.into();
        let y0 = vqtbl4q_u8(table[0], x);
        let y1 = vqtbl4q_u8(table[1], veorq_u8(x, vdupq_n_u8(0x40)));
        let y2 = vqtbl4q_u8(table[2], veorq_u8(x, vdupq_n_u8(0x80)));
        let y3 = vqtbl4q_u8(table[3], veorq_u8(x, vdupq_n_u8(0xC0)));
        veorq_u8(veorq_u8(y0, y1), veorq_u8(y2, y3)).into()
    }
}

pub const AES_INVERT_OR_ZERO_LOOKUP_TABLE: [u8; 256] = [
    0x00, 0x01, 0x8D, 0xF6, 0xCB, 0x52, 0x7B, 0xD1, 0xE8, 0x4F, 0x29, 0xC0, 0xB0, 0xE1, 0xE5, 0xC7,
    0x74, 0xB4, 0xAA, 0x4B, 0x99, 0x2B, 0x60, 0x5F, 0x58, 0x3F, 0xFD, 0xCC, 0xFF, 0x40, 0xEE, 0xB2,
    0x3A, 0x6E, 0x5A, 0xF1, 0x55, 0x4D, 0xA8, 0xC9, 0xC1, 0x0A, 0x98, 0x15, 0x30, 0x44, 0xA2, 0xC2,
    0x2C, 0x45, 0x92, 0x6C, 0xF3, 0x39, 0x66, 0x42, 0xF2, 0x35, 0x20, 0x6F, 0x77, 0xBB, 0x59, 0x19,
    0x1D, 0xFE, 0x37, 0x67, 0x2D, 0x31, 0xF5, 0x69, 0xA7, 0x64, 0xAB, 0x13, 0x54, 0x25, 0xE9, 0x09,
    0xED, 0x5C, 0x05, 0xCA, 0x4C, 0x24, 0x87, 0xBF, 0x18, 0x3E, 0x22, 0xF0, 0x51, 0xEC, 0x61, 0x17,
    0x16, 0x5E, 0xAF, 0xD3, 0x49, 0xA6, 0x36, 0x43, 0xF4, 0x47, 0x91, 0xDF, 0x33, 0x93, 0x21, 0x3B,
    0x79, 0xB7, 0x97, 0x85, 0x10, 0xB5, 0xBA, 0x3C, 0xB6, 0x70, 0xD0, 0x06, 0xA1, 0xFA, 0x81, 0x82,
    0x83, 0x7E, 0x7F, 0x80, 0x96, 0x73, 0xBE, 0x56, 0x9B, 0x9E, 0x95, 0xD9, 0xF7, 0x02, 0xB9, 0xA4,
    0xDE, 0x6A, 0x32, 0x6D, 0xD8, 0x8A, 0x84, 0x72, 0x2A, 0x14, 0x9F, 0x88, 0xF9, 0xDC, 0x89, 0x9A,
    0xFB, 0x7C, 0x2E, 0xC3, 0x8F, 0xB8, 0x65, 0x48, 0x26, 0xC8, 0x12, 0x4A, 0xCE, 0xE7, 0xD2, 0x62,
    0x0C, 0xE0, 0x1F, 0xEF, 0x11, 0x75, 0x78, 0x71, 0xA5, 0x8E, 0x76, 0x3D, 0xBD, 0xBC, 0x86, 0x57,
    0x0B, 0x28, 0x2F, 0xA3, 0xDA, 0xD4, 0xE4, 0x0F, 0xA9, 0x27, 0x53, 0x04, 0x1B, 0xFC, 0xAC, 0xE6,
    0x7A, 0x07, 0xAE, 0x63, 0xC5, 0xDB, 0xE2, 0xEA, 0x94, 0x8B, 0xC4, 0xD5, 0x9D, 0xF8, 0x90, 0x6B,
    0xB1, 0x0D, 0xD6, 0xEB, 0xC6, 0x0E, 0xCF, 0xAD, 0x08, 0x4E, 0xD7, 0xE3, 0x5D, 0x50, 0x1E, 0xB3,
    0x5B, 0x23, 0x38, 0x34, 0x68, 0x46, 0x03, 0x8C, 0xDD, 0x9C, 0x7D, 0xA0, 0xCD, 0x1A, 0x41, 0x1C,
];
