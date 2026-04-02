// Copyright 2024-2025 Irreducible Inc.

use binius_field::{PackedField, field::FieldOps};

/// Extrapolates a line through two points.
///
/// Given two points (0, x0) and (1, x1), this function evaluates the line through these
/// points at parameter z using the formula: x0 + (x1 - x0) * z
///
/// # Properties
/// - When z = 0, returns x0
/// - When z = 1, returns x1
/// - The function is linear in z
#[inline]
pub fn extrapolate_line<F: FieldOps>(x0: F, x1: F, z: F) -> F {
    x0.clone() + (x1 - x0) * z
}

/// Extrapolates lines through a pair of packed fields at a packed vector of points.
///
/// Given two points (0, x0) and (1, x1), this function evaluates the line through these
/// points at parameter z using the formula: x0 + (x1 - x0) * z
///
/// # Properties
/// - When z = 0, returns x0
/// - When z = 1, returns x1
/// - The function is linear in z
/// - Operates on packed field elements for SIMD efficiency
///
/// # Arguments
/// * `x0` - The y-coordinate at z = 0
/// * `x1` - The y-coordinate at z = 1
/// * `z` - The evaluation point(s)
///
/// # Returns
/// The interpolated/extrapolated value(s) at point(s) z
#[inline]
pub fn extrapolate_line_packed<P>(x0: P, x1: P, z: P) -> P
where
    P: PackedField,
{
    x0 + (x1 - x0) * z
}

#[cfg(test)]
mod tests {
    use binius_field::{Field, Random, field::FieldOps};
    use rand::prelude::*;

    use super::*;
    use crate::test_utils::{B128, Packed128b};

    type P = Packed128b;
    type F = B128;

    #[test]
    fn test_extrapolate_line_packed_boundary_values() {
        let mut rng = StdRng::seed_from_u64(0);

        // Test with scalar field
        let x0 = F::random(&mut rng);
        let x1 = F::random(&mut rng);
        let zero = F::ZERO;
        let one = F::ONE;

        // When z = 0, should return x0
        assert_eq!(extrapolate_line_packed(x0, x1, zero), x0);

        // When z = 1, should return x1
        assert_eq!(extrapolate_line_packed(x0, x1, one), x1);

        // Test with packed field
        let x0_packed = P::random(&mut rng);
        let x1_packed = P::random(&mut rng);
        let zero_packed = P::zero();
        let one_packed = P::one();

        // When z = 0, should return x0
        assert_eq!(
            extrapolate_line_packed(x0_packed, x1_packed, zero_packed),
            x0_packed
        );

        // When z = 1, should return x1
        assert_eq!(
            extrapolate_line_packed(x0_packed, x1_packed, one_packed),
            x1_packed
        );
    }

    #[test]
    fn test_extrapolate_line_packed_linearity() {
        let mut rng = StdRng::seed_from_u64(0);

        // Generate random points and values
        let x0 = F::random(&mut rng);
        let x1 = F::random(&mut rng);
        let z0 = F::random(&mut rng);
        let z1 = F::random(&mut rng);
        let alpha = F::random(&mut rng);

        // Test linearity property: f(αz0 + (1-α)z1) = αf(z0) + (1-α)f(z1)
        let z_combined = alpha * z0 + (F::ONE - alpha) * z1;

        let f_z0 = extrapolate_line_packed(x0, x1, z0);
        let f_z1 = extrapolate_line_packed(x0, x1, z1);
        let f_combined = extrapolate_line_packed(x0, x1, z_combined);

        let expected = alpha * f_z0 + (F::ONE - alpha) * f_z1;

        assert_eq!(f_combined, expected);
    }
}
