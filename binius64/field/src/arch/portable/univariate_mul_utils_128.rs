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

use crate::underlier::UnderlierWithBitOps;

/// Trait for 64-bit underliers that can be used in carryless multiplication.
pub trait Underlier64bLanes: UnderlierWithBitOps {
    /// Reverse the bits of the 64-bit lanes.
    fn reverse_bits_64(self) -> Self;

    /// Broadcast a 64-bit value to all lanes.
    fn broadcast_64(val: u64) -> Self;
    /// Perform a wrapping integer multiplication for 64-bit lanes
    fn imul_64(self, other: Self) -> Self;

    /// Shift right by a specified number of bits for 64-bit lanes.
    fn shr_64(self, shift: u32) -> Self;
    /// Shift left by a specified number of bits for 64-bit lanes.
    fn shl_64(self, shift: u32) -> Self;
}

impl Underlier64bLanes for u64 {
    #[inline(always)]
    fn reverse_bits_64(self) -> Self {
        self.reverse_bits()
    }

    #[inline(always)]
    fn broadcast_64(val: u64) -> Self {
        val
    }

    #[inline(always)]
    fn imul_64(self, other: Self) -> Self {
        self.wrapping_mul(other)
    }

    #[inline(always)]
    fn shr_64(self, shift: u32) -> Self {
        self >> shift
    }

    #[inline(always)]
    fn shl_64(self, shift: u32) -> Self {
        self << shift
    }
}

/// Trait for 128-bit underliers that can be used in carryless multiplication.
pub trait Underlier128bLanes: UnderlierWithBitOps {
    /// The twice smaller type containing the 64-bit lanes.
    type U64: Underlier64bLanes;

    /// Split the 128-bit lanes into two 64-bit lanes.
    fn split_hi_lo_64(self) -> (Self::U64, Self::U64);
    /// Join two 64-bit lanes into a 128-bit lane.
    fn join_u64s(high: Self::U64, low: Self::U64) -> Self;
    /// Broadcast a 64-bit value to all lanes.
    fn broadcast_64(val: u64) -> Self;

    /// Interleave the bits of the 128-bit lanes with zeroes doubling the size of the lanes.
    /// Eg. for 0b1110 returns (0b1010, 0b1000).
    fn spread_bits_128(self) -> (Self, Self);
}

impl Underlier128bLanes for u128 {
    type U64 = u64;

    #[inline(always)]
    fn split_hi_lo_64(self) -> (Self::U64, Self::U64) {
        ((self >> 64) as Self::U64, self as Self::U64)
    }

    #[inline(always)]
    fn join_u64s(high: Self::U64, low: Self::U64) -> Self {
        ((high as u128) << 64) | (low as u128)
    }

    #[inline(always)]
    fn broadcast_64(val: u64) -> Self {
        val as u128
    }

    #[inline(always)]
    fn spread_bits_128(self) -> (Self, Self) {
        let (hi, lo) = self.split_hi_lo_64();

        (spread_bits_64(hi), spread_bits_64(lo))
    }
}

/// Multiplication in GF(2)\[X\], truncated to the low 64-bits, with "holes"
/// (sequences of zeroes) to avoid carry spilling.
///
/// When carries do occur, they wind up in a "hole" and are subsequently masked
/// out of the result.
#[inline]
pub fn bmul64<U: Underlier64bLanes>(x: U, y: U) -> U {
    let x0 = x & U::broadcast_64(0x1111_1111_1111_1111);
    let x1 = x & U::broadcast_64(0x2222_2222_2222_2222);
    let x2 = x & U::broadcast_64(0x4444_4444_4444_4444);
    let x3 = x & U::broadcast_64(0x8888_8888_8888_8888);
    let y0 = y & U::broadcast_64(0x1111_1111_1111_1111);
    let y1 = y & U::broadcast_64(0x2222_2222_2222_2222);
    let y2 = y & U::broadcast_64(0x4444_4444_4444_4444);
    let y3 = y & U::broadcast_64(0x8888_8888_8888_8888);

    let mut z0 = U::imul_64(x0, y0) ^ U::imul_64(x1, y3) ^ U::imul_64(x2, y2) ^ U::imul_64(x3, y1);
    let mut z1 = U::imul_64(x0, y1) ^ U::imul_64(x1, y0) ^ U::imul_64(x2, y3) ^ U::imul_64(x3, y2);
    let mut z2 = U::imul_64(x0, y2) ^ U::imul_64(x1, y1) ^ U::imul_64(x2, y0) ^ U::imul_64(x3, y3);
    let mut z3 = U::imul_64(x0, y3) ^ U::imul_64(x1, y2) ^ U::imul_64(x2, y1) ^ U::imul_64(x3, y0);

    z0 &= U::broadcast_64(0x1111_1111_1111_1111);
    z1 &= U::broadcast_64(0x2222_2222_2222_2222);
    z2 &= U::broadcast_64(0x4444_4444_4444_4444);
    z3 &= U::broadcast_64(0x8888_8888_8888_8888);

    z0 | z1 | z2 | z3
}

/// Spread bits of a 64-bit value into a 128-bit value by interleaving zeroes.
#[inline]
pub fn spread_bits_64(val: u64) -> u128 {
    let mut x = val as u128;
    x = (x | (x << 32)) & 0x00000000FFFFFFFF00000000FFFFFFFF;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF0000FFFF0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF00FF00FF00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333333333333333333333333333;
    x = (x | (x << 1)) & 0x55555555555555555555555555555555;
    x
}
#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};

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
            let u64x2 = val.split_hi_lo_64();
            let back: u128 = u128::join_u64s(u64x2.0, u64x2.1);
            assert_eq!(val, back, "Round-trip conversion failed for 0x{val:032x}");
        }
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

    #[test]
    fn test_spread_bits_64() {
        let value: u64 = StdRng::seed_from_u64(0).random();
        let spread = spread_bits_64(value);

        let expected = (0..64)
            .map(|i| {
                if (value & (1 << i)) != 0 {
                    1u128 << (i * 2)
                } else {
                    0u128
                }
            })
            .sum::<u128>();

        assert_eq!(spread, expected, "spread bits failed for 0x{value:016x}");
    }

    #[test]
    fn test_spread_bits_128() {
        let value: u128 = StdRng::seed_from_u64(0).random();
        let (hi, lo) = value.spread_bits_128();

        let expected_hi = spread_bits_64((value >> 64) as u64);
        let expected_lo = spread_bits_64(value as u64);

        assert_eq!(
            hi, expected_hi,
            "spread bits failed for high part of 0x{value:032x}"
        );
        assert_eq!(
            lo, expected_lo,
            "spread bits failed for low part of 0x{value:032x}"
        );
    }
}
