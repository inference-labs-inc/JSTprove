// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

use super::common::{coord_zero, coords_gen};
use crate::bignum::{BigUint, select as select_biguint};

/// Curve point in affine form - a tuple `(x, y)` that satisfies `y^2 = x^3 + 7`,
/// or `(0, 0)` for additive identity (point at infinity).
#[derive(Clone)]
pub struct Secp256k1Affine {
    pub x: BigUint,
    pub y: BigUint,
    pub is_point_at_infinity: Wire,
}

impl Secp256k1Affine {
    /// Point at infinity - the identity element.
    pub fn point_at_infinity(b: &CircuitBuilder) -> Self {
        let zero = coord_zero(b);
        Self {
            x: zero.clone(),
            y: zero,
            is_point_at_infinity: b.add_constant(Word::ALL_ONE),
        }
    }

    /// Generator basepoint.
    pub fn generator(b: &CircuitBuilder) -> Self {
        let (x, y) = coords_gen(b);
        Self {
            x,
            y,
            is_point_at_infinity: b.add_constant(Word::ZERO),
        }
    }

    /// Return point-at-infinity unless the MSB-boolean `cond` is true, then pass the point
    /// unchanged.
    pub fn pai_unless(&self, b: &CircuitBuilder, cond: Wire) -> Secp256k1Affine {
        let is_point_at_infinity = b.select(
            cond,
            self.is_point_at_infinity,
            b.add_constant(Word::ALL_ONE),
        );
        Secp256k1Affine {
            x: self.x.clone(),
            y: self.y.clone(),
            is_point_at_infinity,
        }
    }
}

/// Conditionally selects between two affine secp256k1 points.
///
/// # Arguments
/// * `builder` - Circuit builder for constraint generation
/// * `cond` - an MSB-boolean
/// * `t` - Value to select when cond is true (MSB=1)
/// * `f` - Value to select when cond is false (MSB=0)
///
/// # Return value
/// Selects `t` if `cond` is true, otherwise selects `f`.
pub fn select(
    b: &CircuitBuilder,
    cond: Wire,
    pt: &Secp256k1Affine,
    pf: &Secp256k1Affine,
) -> Secp256k1Affine {
    let x = select_biguint(b, cond, &pt.x, &pf.x);
    let y = select_biguint(b, cond, &pt.y, &pf.y);
    let is_point_at_infinity = b.select(cond, pt.is_point_at_infinity, pf.is_point_at_infinity);
    Secp256k1Affine {
        x,
        y,
        is_point_at_infinity,
    }
}
