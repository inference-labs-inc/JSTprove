// Copyright (c) 2019-2025 The RustCrypto Project Developers
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

//! Hardware-accelerated POLYVAL multiplication using CLMUL instructions.
//!
//! This implementation is adapted from the RustCrypto/universal-hashes repository:
//! <https://github.com/RustCrypto/universal-hashes>.
//!
//! The RustCrypto implementation was based on work by Eric Lagergren.
//!
//! Modified by Irreducible Inc. (2024-2025): Adapted from x86-specific intrinsics
//! to use generic Underlier traits for hardware-agnostic implementation.

use crate::underlier::{OpsClmul, PackedUnderlier, Underlier};

/// Multiply two field elements using CLMUL instructions.
#[inline]
pub fn mul<U: Underlier + OpsClmul + PackedUnderlier<u128>>(x: U, y: U) -> U {
    let (h, m, l) = karatsuba1(x, y);
    let (h, l) = karatsuba2(h, m, l);
    mont_reduce(h, l)
}

/// Karatsuba decomposition for `x*y`.
#[inline]
fn karatsuba1<U: Underlier + OpsClmul>(x: U, y: U) -> (U, U, U) {
    // First Karatsuba step: decompose x and y.
    //
    // (x1*y0 + x0*y1) = (x1+x0) * (y1+x0) + (x1*y1) + (x0*y0)
    //        M                                 H         L
    //
    // m = x.hi^x.lo * y.hi^y.lo
    let m = pmull(
        U::xor(x, U::duplicate_hi_64(x)),
        U::xor(y, U::duplicate_hi_64(y)),
    );
    let h = pmull2(y, x); // h = x.hi * y.hi
    let l = pmull(y, x); // l = x.lo * y.lo
    (h, m, l)
}

/// Karatsuba combine.
#[inline]
fn karatsuba2<U: Underlier + OpsClmul>(h: U, m: U, l: U) -> (U, U) {
    // Second Karatsuba step: combine into a 2n-bit product.
    //
    // m0 ^= l0 ^ h0 // = m0^(l0^h0)
    // m1 ^= l1 ^ h1 // = m1^(l1^h1)
    // l1 ^= m0      // = l1^(m0^l0^h0)
    // h0 ^= l0 ^ m1 // = h0^(l0^m1^l1^h1)
    // h1 ^= l1      // = h1^(l1^m0^l0^h0)
    let t = {
        //   {m0, m1} ^ {l1, h0}
        // = {m0^l1, m1^h0}
        let t0 = { U::xor(m, U::extract_hi_lo_64(l, h)) };

        //   {h0, h1} ^ {l0, l1}
        // = {h0^l0, h1^l1}
        let t1 = U::xor(h, l);

        //   {m0^l1, m1^h0} ^ {h0^l0, h1^l1}
        // = {m0^l1^h0^l0, m1^h0^h1^l1}
        U::xor(t0, t1)
    };

    // {m0^l1^h0^l0, l0}
    let x01 = U::unpacklo_epi64(l, t);

    // {h1, m1^h0^h1^l1}
    let x23 = U::unpackhi_epi64(t, h);

    (x23, x01)
}

#[inline]
fn mont_reduce<U: Underlier + OpsClmul + PackedUnderlier<u128>>(x23: U, x01: U) -> U {
    // Perform the Montgomery reduction over the 256-bit X.
    //    [A1:A0] = X0 • poly
    //    [B1:B0] = [X0 ⊕ A1 : X1 ⊕ A0]
    //    [C1:C0] = B0 • poly
    //    [D1:D0] = [B0 ⊕ C1 : B1 ⊕ C0]
    // Output: [D1 ⊕ X3 : D0 ⊕ X2]
    static POLY: u128 = (1 << 127) | (1 << 126) | (1 << 121) | (1 << 63) | (1 << 62) | (1 << 57);
    let poly = <U as PackedUnderlier<u128>>::broadcast(POLY);
    let a = pmull(x01, poly);
    let b = U::xor(x01, U::swap_hi_lo_64(a));
    let c = pmull2(b, poly);
    U::xor(x23, U::xor(c, b))
}

/// Multiplies the low bits in `a` and `b`.
#[inline]
fn pmull<U: Underlier + OpsClmul>(a: U, b: U) -> U {
    U::clmulepi64::<0x00>(a, b)
}

/// Multiplies the high bits in `a` and `b`.
#[inline]
fn pmull2<U: Underlier + OpsClmul>(a: U, b: U) -> U {
    U::clmulepi64::<0x11>(a, b)
}
