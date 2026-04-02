// Copyright 2025 Irreducible Inc.
use binius_core::consts::WORD_SIZE_BITS;
use binius_frontend::{CircuitBuilder, Wire};

use super::{
    common::{coord_b, coord_beta, coord_field, coord_zero, pow_sqrt, scalar_field},
    point::Secp256k1Affine,
};
use crate::bignum::{BigUint, PseudoMersennePrimeField, assert_eq, biguint_eq, select};

/// Secp256k1 - a short Weierstrass elliptic curve of the form `y^2 = x^3 + 7` over
/// the prime field of modulus `2^256 - 2^32 - 977`.
pub struct Secp256k1 {
    f_p: PseudoMersennePrimeField,
    f_scalar: PseudoMersennePrimeField,
    b: BigUint,
}

impl Secp256k1 {
    /// Creates new curve struct with constants related to curve and its coordinate and scalar
    /// fields.
    pub fn new(b: &CircuitBuilder) -> Self {
        let f_p = coord_field(b);
        let f_scalar = scalar_field(b);
        let b = coord_b(b);
        Self { f_p, f_scalar, b }
    }

    /// Coordinate field.
    pub fn f_p(&self) -> &PseudoMersennePrimeField {
        &self.f_p
    }

    /// Scalar field.
    pub fn f_scalar(&self) -> &PseudoMersennePrimeField {
        &self.f_scalar
    }

    /// Assert that given affine point actually resides on curve.
    ///
    /// Considers point-at-infinity not on curve.
    pub fn assert_on_curve(&self, b: &CircuitBuilder, p: &Secp256k1Affine) {
        let f_p = &self.f_p;

        let x_pow2 = f_p.square(b, &p.x);
        let x_pow3 = f_p.mul(b, &x_pow2, &p.x);

        let y_pow2 = f_p.square(b, &p.y);
        assert_eq(
            b,
            "secp256k1_on_curve",
            &y_pow2,
            &f_p.add(b, &x_pow3, &self.b),
        );

        b.assert_false("not_point_at_infinity", p.is_point_at_infinity);
    }

    /// Recover the full affine point `(r, y)` by its x coordinate and y parity.
    /// Returns point-at-infinity in case recovery isn't possible.
    ///
    /// Note: we don't handle the case where r does not fit into the scalar field, thus
    /// recid is boolean, and not a 0-3 bitmask signifying both parity and scalar field overflow.
    pub fn recover(&self, b: &CircuitBuilder, r: &BigUint, recid_odd: Wire) -> Secp256k1Affine {
        let f_p = self.f_p();

        // y^2 = x^3 + 7, x = r
        let y_squared = f_p.add(b, &f_p.mul(b, &f_p.square(b, r), r), &self.b);

        // p = 2^256 - 2^32 - 977 = 3 (mod 4), compute the quadratic residue by raising to (p+1)/4
        let res_1_limbs =
            b.biguint_mod_pow_hint(&y_squared.limbs, &pow_sqrt(b).limbs, &f_p.modulus().limbs);
        let res_1 = BigUint { limbs: res_1_limbs };
        let res_2 = f_p.sub(b, &coord_zero(b), &res_1);

        // both residues differ in parity
        let res_1_low_limb = *res_1.limbs.first().expect("N_LIMBS > 0");
        let odd_1 = b.shl(res_1_low_limb, (WORD_SIZE_BITS - 1) as u32);
        let is_res_2 = b.bxor(odd_1, recid_odd);

        let y = select(b, is_res_2, &res_2, &res_1);
        let is_valid_point = biguint_eq(b, &f_p.square(b, &y), &y_squared);

        Secp256k1Affine {
            x: r.clone(),
            y,
            is_point_at_infinity: b.bnot(is_valid_point),
        }
    }

    /// Negate the curve point `p` if MSB-bool `cond` is true.
    pub fn negate_if(
        &self,
        b: &CircuitBuilder,
        cond: Wire,
        p: &Secp256k1Affine,
    ) -> Secp256k1Affine {
        let neg_y = self.f_p.sub(b, &coord_zero(b), &p.y);
        let y = select(b, cond, &neg_y, &p.y);
        Secp256k1Affine {
            x: p.x.clone(),
            y,
            is_point_at_infinity: p.is_point_at_infinity,
        }
    }

    /// Compute the endomorphism `λ (x, y) = (βx, y)`.
    pub fn endomorphism(&self, b: &CircuitBuilder, p: &Secp256k1Affine) -> Secp256k1Affine {
        let x = self.f_p.mul(b, &coord_beta(b), &p.x);
        Secp256k1Affine {
            x,
            y: p.y.clone(),
            is_point_at_infinity: p.is_point_at_infinity,
        }
    }

    /// Add two curve points.
    ///
    /// Requires both `p1` and `p2` to be either valid curve points or points at infinities.
    ///
    /// This implementation is complete - it handles the cases of either `p1` or `p2` being
    /// point-at-infinities, as well as being equal (which falls back to doubling).
    pub fn add(
        &self,
        b: &CircuitBuilder,
        p1: &Secp256k1Affine,
        p2: &Secp256k1Affine,
    ) -> Secp256k1Affine {
        let (addition_slope, x_diff_zero, y_diff_zero) = self.addition_slope(b, p1, p2);
        let doubling_slope = self.doubling_slope(b, p1);

        let pai_1 = p1.is_point_at_infinity;
        let pai_2 = p2.is_point_at_infinity;

        let slope = select(b, x_diff_zero, &doubling_slope, &addition_slope);
        let (add_x, add_y) = self.sloped_add(b, &slope, p1, p2);

        let x = select(b, pai_1, &p2.x, &select(b, pai_2, &p1.x, &add_x));
        let y = select(b, pai_1, &p2.y, &select(b, pai_2, &p1.y, &add_y));

        let pai_sum = b.band(x_diff_zero, b.bnot(y_diff_zero)); // adding negation
        let is_point_at_infinity = b.select(pai_1, pai_2, b.select(pai_2, pai_1, pai_sum));

        Secp256k1Affine {
            x,
            y,
            is_point_at_infinity,
        }
    }

    /// Add two curve points, incomplete.
    ///
    /// Requires both `p1` and `p2` to be either valid curve points or points at infinities.
    ///
    /// Unlike [`Secp256k1::add`] this implementation does not handle doubling, asserting false in
    /// that case.
    pub fn add_incomplete(
        &self,
        b: &CircuitBuilder,
        p1: &Secp256k1Affine,
        p2: &Secp256k1Affine,
    ) -> Secp256k1Affine {
        let (slope, x_diff_zero, y_diff_zero) = self.addition_slope(b, p1, p2);

        let pai_1 = p1.is_point_at_infinity;
        let pai_2 = p2.is_point_at_infinity;
        let any_pai = b.bor(pai_1, pai_2);

        let (add_x, add_y) = self.sloped_add(b, &slope, p1, p2);
        let x = select(b, pai_1, &p2.x, &select(b, pai_2, &p1.x, &add_x));
        let y = select(b, pai_1, &p2.y, &select(b, pai_2, &p1.y, &add_y));

        let pai_sum = b.band(x_diff_zero, b.bnot(y_diff_zero)); // adding negation
        let is_point_at_infinity = b.select(pai_1, pai_2, b.select(pai_2, pai_1, pai_sum));

        b.assert_false(
            "not_doubling",
            b.band(b.bnot(any_pai), b.band(x_diff_zero, y_diff_zero)),
        );

        Secp256k1Affine {
            x,
            y,
            is_point_at_infinity,
        }
    }

    /// Double a curve point.
    ///
    /// Requires both `p` to be either a valid curve point or point at infinity.
    pub fn double(&self, b: &CircuitBuilder, p: &Secp256k1Affine) -> Secp256k1Affine {
        let slope = self.doubling_slope(b, p);
        let (x, y) = self.sloped_add(b, &slope, p, p);
        // x ≠ 0 ∧ y = 0 is not possible on secp256k1 because -7 is not a cubic residue in Fp;
        // we can only get PAI result when p is PAI.
        let is_point_at_infinity = p.is_point_at_infinity;
        Secp256k1Affine {
            x,
            y,
            is_point_at_infinity,
        }
    }

    fn addition_slope(
        &self,
        b: &CircuitBuilder,
        p1: &Secp256k1Affine,
        p2: &Secp256k1Affine,
    ) -> (BigUint, Wire, Wire) {
        let f_p = &self.f_p;
        // y₂−y₁/x₂−x₁
        let y_diff = f_p.sub(b, &p2.y, &p1.y);
        let x_diff = f_p.sub(b, &p2.x, &p1.x);
        let y_diff_zero = y_diff.is_zero(b);
        let x_diff_zero = x_diff.is_zero(b);
        let x_diff_inverse = f_p.inverse(b, &x_diff, b.bnot(x_diff_zero));
        let slope = f_p.mul(b, &y_diff, &x_diff_inverse);
        (slope, x_diff_zero, y_diff_zero)
    }

    fn doubling_slope(&self, b: &CircuitBuilder, p: &Secp256k1Affine) -> BigUint {
        let f_p = &self.f_p;
        // λ=3x²/2y
        let x_sqr = f_p.square(b, &p.x);
        let x_sqr_by_3 = f_p.add(b, &f_p.add(b, &x_sqr, &x_sqr), &x_sqr);
        // secp256k1 does not allow y=0, but we still have to check to avoid overconstraining.
        let y_zero = p.y.is_zero(b);
        let y_by_2 = f_p.add(b, &p.y, &p.y);
        let y_by_2_inverse = f_p.inverse(b, &y_by_2, b.bnot(y_zero));
        f_p.mul(b, &x_sqr_by_3, &y_by_2_inverse)
    }

    fn sloped_add(
        &self,
        b: &CircuitBuilder,
        slope: &BigUint,
        p1: &Secp256k1Affine,
        p2: &Secp256k1Affine,
    ) -> (BigUint, BigUint) {
        let f_p = &self.f_p;
        // x₃ = λ² − x₁ − x₂
        // x₃ = λ (x₁ − x₃) − y₁
        let slope_sqr = f_p.square(b, slope);
        let x = f_p.sub(b, &f_p.sub(b, &slope_sqr, &p1.x), &p2.x);
        let y = f_p.sub(b, &f_p.mul(b, slope, &f_p.sub(b, &p1.x, &x)), &p1.y);
        (x, y)
    }
}
