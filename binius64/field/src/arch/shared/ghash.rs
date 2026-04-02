// Copyright 2025 Irreducible Inc.

use crate::{Divisible, underlier::UnderlierWithBitOps};

/// Trait for underliers that support CLMUL operations which are needed for the
/// GHASH multiplication algorithm.
pub trait ClMulUnderlier: UnderlierWithBitOps + Divisible<u128> {
    /// Performs CLMUL operation on two 64-bit values that are selected from 128-bit lanes
    /// by the bytes of the IMM8 parameter.
    fn clmulepi64<const IMM8: i32>(a: Self, b: Self) -> Self;

    /// For each 128-bit lane, shifts the lower 64 bits to the upper 64 bits and zeroes the lower
    /// 64-bit.
    fn move_64_to_hi(a: Self) -> Self;
}

#[inline]
#[allow(dead_code)]
pub fn mul_clmul<U: ClMulUnderlier>(x: U, y: U) -> U {
    // Based on the C++ reference implementation
    // The algorithm performs polynomial multiplication followed by reduction

    // t1a = x.lo * y.hi
    let t1a = U::clmulepi64::<0x01>(x, y);

    // t1b = x.hi * y.lo
    let t1b = U::clmulepi64::<0x10>(x, y);

    // t1 = t1a + t1b (XOR in binary field)
    let mut t1 = t1a ^ t1b;

    // t2 = x.hi * y.hi
    let t2 = U::clmulepi64::<0x11>(x, y);

    // Reduce t1 and t2
    t1 = gf2_128_reduce(t1, t2);

    // t0 = x.lo * y.lo
    let mut t0 = U::clmulepi64::<0x00>(x, y);

    // Final reduction
    t0 = gf2_128_reduce(t0, t1);

    t0
}

/// The version of the multiplication for optimized suqare operation.
#[inline]
#[allow(dead_code)]
pub fn square_clmul<U: ClMulUnderlier>(x: U) -> U {
    // t1 from the previous function is always zero for squaring
    // t2 = x.hi * x.hi
    let t2 = U::clmulepi64::<0x11>(x, x);

    // Calculate t1 * x^64
    let t1 = gf2_128_shift_reduce(t2);

    // t0 = x.lo * x.lo
    let mut t0 = U::clmulepi64::<0x00>(x, x);

    // Final reduction
    t0 = gf2_128_reduce(t0, t1);

    t0
}

// The reduction polynomial x^128 + x^7 + x^2 + x + 1 is represented as 0x87
const POLY: u128 = 0x87;

/// Performs reduction step: returns t0 + x^64 * t1
#[inline]
fn gf2_128_reduce<U: ClMulUnderlier>(mut t0: U, t1: U) -> U {
    let poly = <U as UnderlierWithBitOps>::broadcast_subvalue(POLY);

    // t0 = t0 XOR (t1 << 64)
    // In SIMD, left shift by 64 bits is shifting by 8 bytes
    t0 ^= U::move_64_to_hi(t1);

    // t0 = t0 XOR clmul(t1, poly, 0x01)
    // This multiplies the high 64 bits of t1 with the low 64 bits of poly
    t0 ^= U::clmulepi64::<0x01>(t1, poly);

    t0
}

/// Returns a `x^64 * t` after reduction.
fn gf2_128_shift_reduce<U: ClMulUnderlier>(t: U) -> U {
    let poly = <U as UnderlierWithBitOps>::broadcast_subvalue(POLY);
    let mut result = U::move_64_to_hi(t);

    result ^= U::clmulepi64::<0x01>(t, poly);

    result
}
