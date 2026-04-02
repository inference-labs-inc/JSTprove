// Copyright 2025 Irreducible Inc.

use std::ops::{Deref, DerefMut};

use binius_field::{Field, PackedField, field::FieldOps};
use binius_utils::rayon::prelude::*;

use crate::{
    FieldBuffer,
    inner_product::inner_product_buffers,
    multilinear::{eq::eq_ind_partial_eval, fold::fold_highest_var_inplace},
};

/// Evaluates a multilinear polynomial at a given point using sqrt(n) memory.
///
/// This method computes the evaluation by splitting the computation into two phases:
/// 1. Expand an eq tensor for the first half of coordinates (or at least P::LOG_WIDTH)
/// 2. Take inner products of evaluation chunks with the eq tensor to reduce the problem size
/// 3. Evaluate the remaining coordinates using evaluate_inplace
///
/// This approach uses O(sqrt(2^n)) memory instead of O(2^n).
///
/// # Arguments
/// * `evals` - A FieldBuffer containing the 2^n evaluations over the boolean hypercube
/// * `point` - The n coordinates at which to evaluate the polynomial
///
/// # Returns
/// The evaluation of the multilinear polynomial at the given point
///
/// ## Preconditions
///
/// * `point.len()` must equal `evals.log_len()`
pub fn evaluate<F, P, Data>(evals: &FieldBuffer<P, Data>, point: &[F]) -> F
where
    F: Field,
    P: PackedField<Scalar = F>,
    Data: Deref<Target = [P]>,
{
    assert_eq!(
        point.len(),
        evals.log_len(),
        "precondition: point length must equal evals log length"
    );

    // Split coordinates: first half gets at least P::LOG_WIDTH coordinates
    let first_half_len = (point.len() / 2).max(P::LOG_WIDTH).min(point.len());
    let (first_coords, remaining_coords) = point.split_at(first_half_len);

    // Generate eq tensor for first half of coordinates
    let eq_tensor = eq_ind_partial_eval::<P>(first_coords);

    // If there is no second half, just return the inner product with the whole evals.
    if remaining_coords.is_empty() {
        return inner_product_buffers(evals, &eq_tensor);
    }

    // Calculate chunk size based on first half length
    let log_chunk_size = first_half_len;

    // Collect inner products of chunks into scalar values
    let scalars = evals
        .chunks_par(log_chunk_size)
        .map(|chunk| inner_product_buffers(&chunk, &eq_tensor))
        .collect::<Vec<_>>();

    // Create temporary buffer from collected scalar values
    let temp_buffer = FieldBuffer::<P>::from_values(&scalars);

    // Evaluate remaining coordinates using evaluate_inplace
    evaluate_inplace(temp_buffer, remaining_coords)
}

/// Evaluates a multilinear polynomial at a given point, modifying the buffer in-place.
///
/// This method computes the evaluation of a multilinear polynomial specified by it's evaluations
/// on the boolean hypercube. For an $n$-variate multilinear, this implementation performs
/// $2^n - 1$ field multiplications and allocates no additional memory. The `evals` buffer is
/// modified in-place.
///
/// # Arguments
/// * `evals` - A [`FieldBuffer`] containing the $2^n$ evaluations over the boolean hypercube
/// * `coords` - The $n$ coordinates at which to evaluate the polynomial
///
/// # Returns
/// The evaluation of the multilinear polynomial at the given point
///
/// ## Preconditions
///
/// * `coords.len()` must equal `evals.log_len()`
pub fn evaluate_inplace<F, P, Data>(mut evals: FieldBuffer<P, Data>, coords: &[F]) -> F
where
    F: Field,
    P: PackedField<Scalar = F>,
    Data: DerefMut<Target = [P]>,
{
    assert_eq!(
        coords.len(),
        evals.log_len(),
        "precondition: coords length must equal evals log length"
    );

    // Perform folding for each coordinate in reverse order
    for &coord in coords.iter().rev() {
        fold_highest_var_inplace(&mut evals, coord);
    }

    assert_eq!(evals.len(), 1);
    evals.get(0)
}

/// Evaluates a multilinear polynomial at a given point in-place using scalar operations.
///
/// This is a simple variant of multilinear evaluation that works directly on slices of scalars
/// with only a `FieldOps` bound. For each coordinate (highest to lowest), it folds the upper
/// half into the lower half: `evals[j] += r * (evals[j + half] - evals[j])`.
///
/// The final result is stored in `evals[0]` after all folds.
///
/// # Arguments
/// * `evals` - The 2^n evaluations over the boolean hypercube, modified in-place
/// * `point` - The n coordinates at which to evaluate the polynomial
///
/// # Panics
///
/// Panics if `evals.len() != 1 << point.len()`.
pub fn evaluate_inplace_scalars<F: FieldOps>(
    mut evals: impl DerefMut<Target = [F]>,
    point: &[F],
) -> F {
    assert_eq!(
        evals.len(),
        1 << point.len(),
        "precondition: evals length must be 2^point.len()"
    );

    for (log_half_len, point_i) in point.iter().enumerate().rev() {
        let half_len = 1 << log_half_len;
        for j in 0..half_len {
            let delta = evals[j + half_len].clone() - evals[j].clone();
            evals[j] += point_i.clone() * delta;
        }
    }
    evals[0].clone()
}

#[cfg(test)]
mod tests {
    use rand::{RngCore, SeedableRng, rngs::StdRng};

    use super::*;
    use crate::{
        inner_product::inner_product_par,
        test_utils::{
            B128, Packed128b, index_to_hypercube_point, random_field_buffer, random_scalars,
        },
    };

    type P = Packed128b;
    type F = B128;

    #[test]
    fn test_evaluate_consistency() {
        /// Simple reference function for multilinear polynomial evaluation.
        fn evaluate_with_inner_product<F, P, Data>(evals: &FieldBuffer<P, Data>, point: &[F]) -> F
        where
            F: Field,
            P: PackedField<Scalar = F>,
            Data: Deref<Target = [P]>,
        {
            assert_eq!(point.len(), evals.log_len());

            // Compute the equality indicator tensor expansion
            let eq_tensor = eq_ind_partial_eval::<P>(point);
            inner_product_par(evals, &eq_tensor)
        }

        let mut rng = StdRng::seed_from_u64(0);

        for log_n in [0, P::LOG_WIDTH - 1, P::LOG_WIDTH, 10] {
            // Generate random buffer and evaluation point
            let buffer = random_field_buffer::<P>(&mut rng, log_n);
            let point = random_scalars::<F>(&mut rng, log_n);

            // Evaluate using all three methods
            let result_inner_product = evaluate_with_inner_product(&buffer, &point);
            let result_inplace = evaluate_inplace(buffer.clone(), &point);
            let result_sqrt_memory = evaluate(&buffer, &point);

            // All results should be equal
            assert_eq!(result_inner_product, result_inplace);
            assert_eq!(result_inner_product, result_sqrt_memory);
        }
    }

    #[test]
    fn test_evaluate_at_hypercube_indices() {
        let mut rng = StdRng::seed_from_u64(0);

        // Generate random multilinear with 8 variables
        let log_n = 8;
        let buffer = random_field_buffer::<F>(&mut rng, log_n);

        // Test 16 random hypercube indices
        for _ in 0..16 {
            let index = (rng.next_u32() as usize) % (1 << log_n);
            let point = index_to_hypercube_point::<F>(log_n, index);

            // Evaluate at the hypercube point
            let eval_result = evaluate(&buffer, &point);

            // Get the value directly from the buffer
            let direct_value = buffer.get(index);

            // They should be equal
            assert_eq!(eval_result, direct_value);
        }
    }

    #[test]
    fn test_evaluate_inplace_scalars_consistency() {
        let mut rng = StdRng::seed_from_u64(0);

        for log_n in [0, P::LOG_WIDTH - 1, P::LOG_WIDTH, 10] {
            let buffer = random_field_buffer::<P>(&mut rng, log_n);
            let point = random_scalars::<F>(&mut rng, log_n);

            let result_inplace = evaluate_inplace(buffer.clone(), &point);

            let scalar_evals = buffer.iter_scalars().collect::<Vec<_>>();
            let result_scalar = evaluate_inplace_scalars(scalar_evals, &point);

            assert_eq!(result_inplace, result_scalar, "mismatch at log_n={log_n}");
        }
    }

    #[test]
    fn test_linearity() {
        let mut rng = StdRng::seed_from_u64(0);

        // Generate random 8-variable multilinear and evaluation point
        let log_n = 8;
        let buffer = random_field_buffer::<F>(&mut rng, log_n);
        let mut point = random_scalars::<F>(&mut rng, log_n);

        // Test linearity for each coordinate
        for coord_idx in 0..log_n {
            // Choose three coordinate values
            let coord_vals = random_scalars::<F>(&mut rng, 3);

            // Evaluate at three points differing only in coordinate coord_idx
            let evals: Vec<_> = coord_vals
                .iter()
                .map(|&coord_val| {
                    point[coord_idx] = coord_val;
                    evaluate(&buffer, &point)
                })
                .collect();

            // Check that the three evaluations form a line
            // For a line through points (x0, y0), (x1, y1), (x2, y2):
            // y2 - y0 = (y1 - y0) * (x2 - x0) / (x1 - x0)
            // Rearranging: (y2 - y0) * (x1 - x0) = (y1 - y0) * (x2 - x0)
            let x0 = coord_vals[0];
            let x1 = coord_vals[1];
            let x2 = coord_vals[2];
            let y0 = evals[0];
            let y1 = evals[1];
            let y2 = evals[2];

            let lhs = (y2 - y0) * (x1 - x0);
            let rhs = (y1 - y0) * (x2 - x0);

            assert_eq!(lhs, rhs);
        }
    }
}
