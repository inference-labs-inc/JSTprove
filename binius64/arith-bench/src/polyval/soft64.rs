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

//! Constant-time software implementation of POLYVAL for 64-bit architectures.
//!
//! This implementation is adapted from the RustCrypto/universal-hashes repository:
//! <https://github.com/RustCrypto/universal-hashes>
//!
//! Which in turn was adapted from BearSSL's `ghash_ctmul64.c`:
//! <https://bearssl.org/gitweb/?p=BearSSL;a=blob;f=src/hash/ghash_ctmul64.c;hb=4b6046412>
//!
//! Copyright (c) 2016 Thomas Pornin <pornin@bolet.org>
//!
//! Modified by Irreducible Inc. (2024-2025): Extracted core multiplication
//! algorithm and adapted for standalone use in the Binius field framework.

use crate::arch::portable64::{U64x2, bmul64, rev64};

/// Multiply two POLYVAL field elements using software implementation.
///
/// Method described at:
/// * <https://www.bearssl.org/constanttime.html#ghash-for-gcm>
/// * <https://crypto.stackexchange.com/questions/66448/how-does-bearssls-gcm-modular-reduction-work/66462#66462>
///
/// This actually performs the mathematical operation $x \cdot y \cdot X^{-128} \mod P(X)$, where
/// $P(X) = X^128 + X^127 + X^126 + X^121 + 1$. This is a sort of binary polynomial Montgomery
/// multiplication.
pub fn mul(x: u128, y: u128) -> u128 {
    // Convert to U64x2 representation
    let U64x2(x0, x1) = U64x2::from(x);
    let U64x2(y0, y1) = U64x2::from(y);

    // Perform multiplication
    let x0r = rev64(x0);
    let x1r = rev64(x1);
    let x2 = x0 ^ x1;
    let x2r = x0r ^ x1r;

    let y0r = rev64(y0);
    let y1r = rev64(y1);
    let y2 = y0 ^ y1;
    let y2r = y0r ^ y1r;
    let z0 = bmul64(y0, x0);
    let z1 = bmul64(y1, x1);

    let mut z2 = bmul64(y2, x2);
    let mut z0h = bmul64(y0r, x0r);
    let mut z1h = bmul64(y1r, x1r);
    let mut z2h = bmul64(y2r, x2r);

    z2 ^= z0 ^ z1;
    z2h ^= z0h ^ z1h;
    z0h = rev64(z0h) >> 1;
    z1h = rev64(z1h) >> 1;
    z2h = rev64(z2h) >> 1;

    let v0 = z0;
    let mut v1 = z0h ^ z2;
    let mut v2 = z1 ^ z2h;
    let mut v3 = z1h;

    // Reduce modulo X^128 + X^127 + X^126 + X^121 + 1.
    v2 ^= v0 ^ (v0 >> 1) ^ (v0 >> 2) ^ (v0 >> 7);
    v1 ^= (v0 << 63) ^ (v0 << 62) ^ (v0 << 57);
    v3 ^= v1 ^ (v1 >> 1) ^ (v1 >> 2) ^ (v1 >> 7);
    v2 ^= (v1 << 63) ^ (v1 << 62) ^ (v1 << 57);

    // Convert back to u128
    U64x2(v2, v3).into()
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::polyval::MONTGOMERY_ONE;

    proptest! {
        #[test]
        fn test_polyval_soft64_mul_commutative(
            a in any::<u128>(),
            b in any::<u128>()
        ) {
            // Test that a * b = b * a
            let ab = mul(a, b);
            let ba = mul(b, a); // // spellchecker:disable-line
            prop_assert_eq!(ab, ba, "POLYVAL soft64 multiplication is not commutative"); // spellchecker:disable-line
        }

        #[test]
        fn test_polyval_soft64_mul_associative(
            a in any::<u128>(),
            b in any::<u128>(),
            c in any::<u128>()
        ) {
            // Test that (a * b) * c = a * (b * c)
            let ab_c = mul(mul(a, b), c);
            let a_bc = mul(a, mul(b, c));
            prop_assert_eq!(ab_c, a_bc, "POLYVAL soft64 multiplication is not associative");
        }

        #[test]
        fn test_polyval_soft64_mul_distributive(
            a in any::<u128>(),
            b in any::<u128>(),
            c in any::<u128>()
        ) {
            // Test that a * (b + c) = (a * b) + (a * c) where + is XOR
            let b_plus_c = b ^ c;
            let a_times_b_plus_c = mul(a, b_plus_c);

            let ab = mul(a, b);
            let ac = mul(a, c);
            let ab_plus_ac = ab ^ ac;

            prop_assert_eq!(a_times_b_plus_c, ab_plus_ac,
                "POLYVAL soft64 multiplication does not satisfy the distributive law");
        }

        #[test]
        fn test_polyval_soft64_mul_identity(
            a in any::<u128>()
        ) {
            // Test that a * MONTGOMERY_ONE = a
            // MONTGOMERY_ONE is the multiplicative identity in Montgomery form
            let result = mul(a, MONTGOMERY_ONE);
            prop_assert_eq!(result, a, "The provided identity is not the multiplicative identity in POLYVAL soft64");
        }
    }
}
