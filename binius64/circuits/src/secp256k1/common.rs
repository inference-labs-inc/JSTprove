// Copyright 2025 Irreducible Inc.
use binius_frontend::CircuitBuilder;
use hex_literal::hex;

use crate::bignum::{BigUint, PseudoMersennePrimeField};

pub const N_LIMBS: usize = 4;

// Generator X coordinate, big endian.
const GX_BE: [u8; 32] = hex!("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");

// Generator Y coordinate, big endian.
const GY_BE: [u8; 32] = hex!("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");

// ((2^256-2^32-977) + 1)/4 => exponent for finding quadratic residues.
const POW_SQRT: [u8; 32] = hex!("3FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBFFFFF0C");

// The value `λ` of the endomorphism `λ (x, y) = (βx, y)`, big endian.
const LAMBDA: [u8; 32] = hex!("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72");

// The value `β` of the endomorphism `λ (x, y) = (βx, y)`, big endian.
const BETA: [u8; 32] = hex!("7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE");

/// Padded zero.
pub fn coord_zero(b: &CircuitBuilder) -> BigUint {
    BigUint::new_constant(b, &num_bigint::BigUint::ZERO).zero_extend(b, N_LIMBS)
}

/// Zero extended b = 7 constant.
pub fn coord_b(b: &CircuitBuilder) -> BigUint {
    BigUint::new_constant(b, &num_bigint::BigUint::from(7usize)).zero_extend(b, N_LIMBS)
}

/// The value `λ` of the endomorphism `λ (x, y) = (βx, y)`.
pub fn coord_lambda(b: &CircuitBuilder) -> BigUint {
    BigUint::new_constant(b, &num_bigint::BigUint::from_bytes_be(&LAMBDA))
}

/// The value `β` of the endomorphism `λ (x, y) = (βx, y)`.
pub fn coord_beta(b: &CircuitBuilder) -> BigUint {
    BigUint::new_constant(b, &num_bigint::BigUint::from_bytes_be(&BETA))
}

/// Quadratic residue exponent.
///
/// Field modulus p = 3 (mod 4) allow raising to the power (p+1)/4 to compute one of quadratic
/// residues. Another quadratic residue is its additive inverse.
pub fn pow_sqrt(b: &CircuitBuilder) -> BigUint {
    BigUint::new_constant(b, &num_bigint::BigUint::from_bytes_be(&POW_SQRT))
}

/// Coordinates of the generator basepoint.
pub fn coords_gen(b: &CircuitBuilder) -> (BigUint, BigUint) {
    let x = BigUint::new_constant(b, &num_bigint::BigUint::from_bytes_be(&GX_BE));
    let y = BigUint::new_constant(b, &num_bigint::BigUint::from_bytes_be(&GY_BE));
    (x, y)
}

/// Coordinate prime field, of modulus `2^256 - 2^32 - 977`.
pub fn coord_field(b: &CircuitBuilder) -> PseudoMersennePrimeField {
    PseudoMersennePrimeField::new(b, 256, &[1 << 32 | 977])
}

/// Scalar prime field, of modulus equal to secp256k1 group size.
pub fn scalar_field(b: &CircuitBuilder) -> PseudoMersennePrimeField {
    PseudoMersennePrimeField::new(b, 256, &[0x402da1732fc9bebf, 0x4551231950b75fc4, 1])
}
