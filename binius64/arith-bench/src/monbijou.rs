// Copyright 2025 Irreducible Inc.
//! Arithmetic for the Monbijou field, GF(2)\[X\] / (X^64 + X^4 + X^3 + X + 1).
//!
//! This module implements arithmetic in the GF(2^64) binary field defined by the
//! reduction polynomial X^64 + X^4 + X^3 + X + 1, which is used in the ISO 3309
//! standard for CRC-64 error detection.
//!
//! The implementation uses carry-less multiplication (CLMUL) CPU instructions for
//! efficient field multiplication on modern x86_64 processors. The algorithm is
//! optimized for SIMD parallelism, processing multiple field elements simultaneously
//! when using vector types like __m128i or __m256i.

use crate::{PackedUnderlier, Underlier, underlier::OpsClmul};

/// The multiplicative identity in the Monbijou field
///
/// In this field, the standard representation of 1 is simply 0x01
pub const MONBIJOU_ONE: u64 = 0x01;

/// The multiplicative identity in the Monbijou 128-bit extension field
///
/// In the degree-2 extension GF(2^128), the standard representation of 1 is simply 0x01
pub const MONBIJOU_128B_ONE: u128 = 0x01;

/// Multiplies two elements in GF(2^64) using SIMD carry-less multiplication.
///
/// This function performs multiplication in the Monbijou field GF(2^64) defined by
/// the reduction polynomial X^64 + X^4 + X^3 + X + 1. The algorithm uses a two-stage
/// reduction process to efficiently handle the polynomial reduction after multiplication.
#[inline]
#[allow(dead_code)]
pub fn mul_clmul<U: Underlier + OpsClmul + PackedUnderlier<u64>>(a: U, b: U) -> U {
    // Step 1: Carry-less multiplication of 64-bit operands produces 128-bit results
    // For SIMD types, this processes multiple pairs in parallel
    let prod_0 = U::clmulepi64::<0x00>(a, b); // 128-bit pre-reduction product elements 0
    let prod_1 = U::clmulepi64::<0x11>(a, b); // 128-bit pre-reduction product elements 1
    reduce_pair(prod_0, prod_1)
}

/// Multiplies two elements in GF(2^128), represented as a degree-2 extension of GF(2^64).
///
/// This field is defined as GF(2)[X, Y] / (X^64 + X^4 + X^3 + X + 1) / (Y^2 + XY + 1).
#[inline]
pub fn mul_128b_clmul<U: Underlier + OpsClmul + PackedUnderlier<u64>>(x: U, y: U) -> U {
    // This is the bit representation of the lower-degree terms (X^4 + X^3 + X + 1)
    const POLY: u64 = 0x1B;
    let poly = <U as PackedUnderlier<u64>>::broadcast(POLY);

    // t0 = x.lo * y.lo
    let t0 = U::clmulepi64::<0x00>(x, y);
    // t2 = x.hi * y.hi
    let t2 = U::clmulepi64::<0x11>(x, y);

    // t1a = x.lo * y.hi
    let t1a = U::clmulepi64::<0x01>(x, y);
    // t1b = x.hi * y.lo
    let t1b = U::clmulepi64::<0x10>(x, y);
    // t1 = t1a + t1b (XOR in binary field)
    let t1 = U::xor(t1a, t1b);

    let mut t2_times_x = U::slli_epi64::<1>(t2);
    let t2_overflow_mask = U::movepi64_mask(t2);
    let t2_overflow_redc = U::and(poly, t2_overflow_mask);
    t2_times_x = U::xor(t2_overflow_redc, t2_times_x);

    let term0 = U::xor(t0, t2);
    let term1 = U::xor(t1, t2_times_x);

    reduce_pair(term0, term1)
}

#[inline]
fn reduce_pair<U: Underlier + OpsClmul + PackedUnderlier<u64>>(prod_0: U, prod_1: U) -> U {
    // The reduction polynomial X^64 + X^4 + X^3 + X + 1 is represented as 0x1B
    // This is the bit representation of the lower-degree terms (X^4 + X^3 + X + 1)
    const POLY: u64 = 0x1B;
    let poly = <U as PackedUnderlier<u64>>::broadcast(POLY);

    // Step 2: First reduction - multiply high 64 bits by reduction polynomial
    // This effectively computes: high_bits * (X^4 + X^3 + X + 1) mod X^128
    let first_reduction_0 = U::clmulepi64::<0x01>(prod_0, poly);
    let first_reduction_1 = U::clmulepi64::<0x01>(prod_1, poly);

    // Extract the low 64 bits from the original products and first reductions
    let prod_lo = U::unpacklo_epi64(prod_0, prod_1);
    let first_reduction_lo = U::unpacklo_epi64(first_reduction_0, first_reduction_1);
    let result = U::xor(prod_lo, first_reduction_lo);

    // Step 3: Second reduction - handle overflow from the first reduction
    // The first reduction can produce results up to 67 bits, so we need another reduction
    let second_reduction_0 = U::clmulepi64::<0x01>(first_reduction_0, poly);
    let second_reduction_1 = U::clmulepi64::<0x01>(first_reduction_1, poly);

    // Extract low 64 bits of the second reduction
    let second_reduction_lo = U::unpacklo_epi64(second_reduction_0, second_reduction_1);

    // Final result: XOR all three components together
    U::xor(result, second_reduction_lo)
}
