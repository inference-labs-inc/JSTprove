// Copyright 2025 Irreducible Inc.
//! Secp256k1 endomorphism split
//!
//! The curve has an endomorphism `λ (x, y) = (βx, y)` where `λ³=1 (mod n)`
//! and `β³=1 (mod p)` (`n` being the scalar field modulus and `p` coordinate field one).
//!
//! For a 256-bit scalar `k` it is possible to split it into `k1` and `k2` such that
//! `k1 + λ k2 = k (mod n)` and both `k1` and `k2` are no farther than `2^128` from zero.
//!
//! The `k` scalar is represented by four 64-bit limbs in little endian order. The return value is
//! quadruple of `(k1_neg, k2_neg, k1_abs, k2_abs)` where `k1_neg` and `k2_neg` are MSB-bools
//! indicating whether `k1_abs` or `k2_abs`, respectively, should be negated. `k1_abs` and `k2_abs`
//! are at most 128 bits and are represented with two 64-bit limbs. When `k` cannot be represented
//! in this way (any valid scalar can, so it has to be modulus or above) both  `k1_abs` and `k2_abs`
//! are assigned zero values.
//!
//! This is a hint - a deterministic computation that happens only on the prover side.
//! The result should be additionally constrained by using bignum circuits to check that
//! `k1 + λ k2 = k (mod n)`.
//!
//! The method used here comes straight from libsecp256k1, follow the link for derivation:
//! <https://github.com/bitcoin-core/secp256k1/blob/master/src/scalar_impl.h#L92-L141>

use binius_core::Word;
use hex_literal::hex;
use num_bigint::BigUint;

use super::Hint;
use crate::util::num_biguint_from_u64_limbs;

pub struct Secp256k1EndosplitHint {
    minus_b1: BigUint,
    minus_b2: BigUint,
    g1: BigUint,
    g2: BigUint,
    endomorphism_lambda: BigUint,
    scalar_modulus: BigUint,
    scalar_modulus_half: BigUint,
    k1_tight_bound: BigUint,
    k2_tight_bound: BigUint,
}

impl Secp256k1EndosplitHint {
    pub fn new() -> Self {
        let [
            minus_b1,
            minus_b2,
            g1,
            g2,
            endomorphism_lambda,
            scalar_modulus,
            k1_tight_bound,
            k2_tight_bound,
        ] = [
            hex!("e4437ed6010e88286f547fa90abfe4c3").as_slice(),
            hex!("fffffffffffffffffffffffffffffffe8a280ac50774346dd765cda83db1562c").as_slice(),
            hex!("3086d221a7d46bcde86c90e49284eb153daa8a1471e8ca7fe893209a45dbb031").as_slice(),
            hex!("e4437ed6010e88286f547fa90abfe4c4221208ac9df506c61571b4ae8ac47f71").as_slice(),
            hex!("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72").as_slice(),
            hex!("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141").as_slice(),
            hex!("a2a8918ca85bafe22016d0b917e4dd77").as_slice(),
            hex!("8a65287bd47179fb2be08846cea267ed").as_slice(),
        ]
        .map(num_bigint::BigUint::from_bytes_be);

        let scalar_modulus_half = &scalar_modulus >> 1;

        Self {
            minus_b1,
            minus_b2,
            g1,
            g2,
            endomorphism_lambda,
            scalar_modulus,
            scalar_modulus_half,
            k1_tight_bound,
            k2_tight_bound,
        }
    }

    fn scalar_abs(&self, scalar: BigUint) -> (Word, BigUint) {
        if scalar > self.scalar_modulus_half {
            (Word::ALL_ONE, &self.scalar_modulus - scalar)
        } else {
            (Word::ZERO, scalar)
        }
    }
}

impl Default for Secp256k1EndosplitHint {
    fn default() -> Self {
        Self::new()
    }
}

impl Hint for Secp256k1EndosplitHint {
    fn shape(&self, dimensions: &[usize]) -> (usize, usize) {
        assert!(
            dimensions.is_empty(),
            "Secp256k1EndosplitHint has constant shape"
        );
        (4, 6)
    }

    fn execute(&self, dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
        assert!(
            dimensions.is_empty(),
            "Secp256k1EndosplitHint has constant shape"
        );

        assert_eq!(inputs.len(), 4);
        assert_eq!(outputs.len(), 6);

        let k =
            num_biguint_from_u64_limbs(inputs.iter().map(|w| w.as_u64())) % &self.scalar_modulus;

        // https://github.com/bitcoin-core/secp256k1/blob/master/src/scalar_impl.h#L92-L141
        let c1 = div_pow2_round(&k * &self.g1, 384) * &self.minus_b1;
        let c2 = div_pow2_round(&k * &self.g2, 384) * &self.minus_b2;

        let k2 = (c1 + c2) % &self.scalar_modulus;
        let k2_lambda = (&k2 * &self.endomorphism_lambda) % &self.scalar_modulus;
        let k1 = (&self.scalar_modulus - k2_lambda + k) % &self.scalar_modulus;

        // bring the magnitude of k1 & k2 below 2^128 by conditional negation
        let (k1_neg, mut k1_abs) = self.scalar_abs(k1);
        let (k2_neg, mut k2_abs) = self.scalar_abs(k2);

        if k1_abs >= self.k1_tight_bound || k2_abs >= self.k2_tight_bound {
            k1_abs = BigUint::ZERO;
            k2_abs = BigUint::ZERO;
        }

        outputs.fill(Word::ZERO);

        outputs[0] = k1_neg;
        outputs[1] = k2_neg;

        for (output, limb) in outputs[2..4].iter_mut().zip(k1_abs.iter_u64_digits()) {
            *output = Word::from_u64(limb);
        }

        for (output, limb) in outputs[4..6].iter_mut().zip(k2_abs.iter_u64_digits()) {
            *output = Word::from_u64(limb);
        }
    }
}

fn div_pow2_round(value: BigUint, shift: u64) -> BigUint {
    let increment = if shift == 0 || !value.bit(shift - 1) {
        BigUint::ZERO
    } else {
        BigUint::from(1usize)
    };
    (value >> shift) + increment
}
