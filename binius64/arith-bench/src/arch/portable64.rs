// Copyright (c) 2019-2025 The RustCrypto Project Developers
// Copyright (c) 2016 Thomas Pornin <pornin@bolet.org>
//
// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.

//! Constant-time software implementation of carryless multiplication for 64-bit architectures.
//!
//! This implementation is adapted from the RustCrypto/universal-hashes repository:
//! <https://github.com/RustCrypto/universal-hashes>
//!
//! Which in turn was adapted from BearSSL's `ghash_ctmul64.c`:
//! <https://bearssl.org/gitweb/?p=BearSSL;a=blob;f=src/hash/ghash_ctmul64.c;hb=4b6046412>

use std::num::Wrapping;

/// 2 x `u64` values
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct U64x2(pub u64, pub u64);

impl From<u128> for U64x2 {
    fn from(x: u128) -> Self {
        // Little-endian: low 64 bits first, then high 64 bits
        U64x2(x as u64, (x >> 64) as u64)
    }
}

impl From<U64x2> for u128 {
    fn from(x: U64x2) -> Self {
        // Little-endian: x.0 is low 64 bits, x.1 is high 64 bits
        (x.0 as u128) | ((x.1 as u128) << 64)
    }
}

/// Multiplication in GF(2)\[X\], truncated to the low 64-bits, with "holes"
/// (sequences of zeroes) to avoid carry spilling.
///
/// When carries do occur, they wind up in a "hole" and are subsequently masked
/// out of the result.
pub fn bmul64(x: u64, y: u64) -> u64 {
    let x0 = Wrapping(x & 0x1111_1111_1111_1111);
    let x1 = Wrapping(x & 0x2222_2222_2222_2222);
    let x2 = Wrapping(x & 0x4444_4444_4444_4444);
    let x3 = Wrapping(x & 0x8888_8888_8888_8888);
    let y0 = Wrapping(y & 0x1111_1111_1111_1111);
    let y1 = Wrapping(y & 0x2222_2222_2222_2222);
    let y2 = Wrapping(y & 0x4444_4444_4444_4444);
    let y3 = Wrapping(y & 0x8888_8888_8888_8888);

    let mut z0 = ((x0 * y0) ^ (x1 * y3) ^ (x2 * y2) ^ (x3 * y1)).0;
    let mut z1 = ((x0 * y1) ^ (x1 * y0) ^ (x2 * y3) ^ (x3 * y2)).0;
    let mut z2 = ((x0 * y2) ^ (x1 * y1) ^ (x2 * y0) ^ (x3 * y3)).0;
    let mut z3 = ((x0 * y3) ^ (x1 * y2) ^ (x2 * y1) ^ (x3 * y0)).0;

    z0 &= 0x1111_1111_1111_1111;
    z1 &= 0x2222_2222_2222_2222;
    z2 &= 0x4444_4444_4444_4444;
    z3 &= 0x8888_8888_8888_8888;

    z0 | z1 | z2 | z3
}

/// Bit-reverse a `u64` in constant time
pub fn rev64(mut x: u64) -> u64 {
    x = ((x & 0x5555_5555_5555_5555) << 1) | ((x >> 1) & 0x5555_5555_5555_5555);
    x = ((x & 0x3333_3333_3333_3333) << 2) | ((x >> 2) & 0x3333_3333_3333_3333);
    x = ((x & 0x0f0f_0f0f_0f0f_0f0f) << 4) | ((x >> 4) & 0x0f0f_0f0f_0f0f_0f0f);
    x = ((x & 0x00ff_00ff_00ff_00ff) << 8) | ((x >> 8) & 0x00ff_00ff_00ff_00ff);
    x = ((x & 0xffff_0000_ffff) << 16) | ((x >> 16) & 0xffff_0000_ffff);
    x.rotate_right(32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u64x2_conversion() {
        // Test round-trip conversion
        let test_values = [
            0u128,
            1u128,
            u128::MAX,
            0x0123456789abcdef_fedcba9876543210u128,
        ];

        for &val in &test_values {
            let u64x2 = U64x2::from(val);
            let back: u128 = u64x2.into();
            assert_eq!(val, back, "Round-trip conversion failed for 0x{val:032x}");
        }
    }

    #[test]
    fn test_rev64() {
        // Test bit reversal
        assert_eq!(rev64(0x0000000000000000), 0x0000000000000000);
        assert_eq!(rev64(0xffffffffffffffff), 0xffffffffffffffff);
        assert_eq!(rev64(0x0123456789abcdef), 0xf7b3d591e6a2c480);
        assert_eq!(rev64(0x8000000000000000), 0x0000000000000001);
        assert_eq!(rev64(0x0000000000000001), 0x8000000000000000);
    }

    #[test]
    fn test_bmul64_basic() {
        // Test basic cases
        assert_eq!(bmul64(0, 0), 0);
        assert_eq!(bmul64(1, 1), 1);
        assert_eq!(bmul64(2, 2), 4);
        assert_eq!(bmul64(3, 3), 5); // 11b * 11b = 101b in GF(2)[X]

        // Test that bmul64 is commutative
        let test_pairs = [
            (0x1234567890abcdef, 0xfedcba0987654321),
            (0x1111111111111111, 0x2222222222222222),
            (0xaaaaaaaaaaaaaaaa, 0x5555555555555555),
        ];

        for (a, b) in test_pairs {
            assert_eq!(
                bmul64(a, b),
                bmul64(b, a),
                "bmul64 not commutative for 0x{a:016x} and 0x{b:016x}",
            );
        }
    }
}
