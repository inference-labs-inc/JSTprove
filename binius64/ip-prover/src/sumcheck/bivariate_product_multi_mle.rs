// Copyright 2023-2025 Irreducible Inc.

use std::cmp::max;

use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{FieldBuffer, multilinear::fold::fold_highest_var_inplace};
use binius_utils::rayon::prelude::*;
use itertools::{Itertools, izip};

use super::{common::SumcheckProver, error::Error, gruen32::Gruen32, round_evals::RoundEvals2};
use crate::sumcheck::common::MleCheckProver;

/// Multiple claim version of `BivariateProductMlecheckProver` that can prove mlechecks
/// that share the evaluation point. This allows deduplicating folding and evaluation work.
pub struct BivariateProductMultiMlecheckProver<P: PackedField> {
    multilinears: Vec<FieldBuffer<P>>,
    last_coeffs_or_sums: RoundCoeffsOrSums<P::Scalar>,
    gruen32: Gruen32<P>,
}

impl<F: Field, P: PackedField<Scalar = F>> BivariateProductMultiMlecheckProver<P> {
    /// Constructs a prover, given the multilinear polynomial evaluations (in pairs) and
    /// evaluation claims on the shared evaluation point.
    pub fn new(
        multilinears: Vec<[FieldBuffer<P>; 2]>,
        eval_point: &[F],
        eval_claims: Vec<F>,
    ) -> Result<Self, Error> {
        let n_vars = eval_point.len();

        if multilinears
            .iter()
            .flatten()
            .any(|multilinear| multilinear.log_len() != n_vars)
        {
            return Err(Error::MultilinearSizeMismatch);
        }

        if multilinears.len() != eval_claims.len() {
            return Err(Error::EvalClaimsNumberMismatch);
        }

        let multilinears = multilinears.into_iter().flatten().collect_vec();
        let last_coeffs_or_sums = RoundCoeffsOrSums::Sums(eval_claims);

        let gruen32 = Gruen32::new(eval_point);

        Ok(Self {
            multilinears,
            last_coeffs_or_sums,
            gruen32,
        })
    }
}

impl<F, P> SumcheckProver<F> for BivariateProductMultiMlecheckProver<P>
where
    F: Field,
    P: PackedField<Scalar = F>,
{
    fn n_vars(&self) -> usize {
        self.gruen32.n_vars_remaining()
    }

    fn n_claims(&self) -> usize {
        match &self.last_coeffs_or_sums {
            RoundCoeffsOrSums::Coeffs(v) => v.len(),
            RoundCoeffsOrSums::Sums(v) => v.len(),
        }
    }

    fn execute(&mut self) -> Result<Vec<RoundCoeffs<F>>, Error> {
        let RoundCoeffsOrSums::Sums(sums) = &self.last_coeffs_or_sums else {
            return Err(Error::ExpectedFold);
        };

        assert!(self.n_vars() > 0);

        // Perform chunked summation: for every row, evaluate all compositions and add up
        // results to an array of round evals accumulators. Alternative would be to sum each
        // composition on its own pass, but that would require reading the entirety of eq field
        // buffer on each pass, which will evict the latter from the cache. By doing chunked
        // compute, we reasonably hope that eq chunk always stays in L1 cache.
        const MAX_CHUNK_VARS: usize = 8;
        let chunk_vars = max(MAX_CHUNK_VARS, P::LOG_WIDTH).min(self.n_vars() - 1);

        let packed_prime_evals = (0..1 << (self.n_vars() - 1 - chunk_vars))
			.into_par_iter()
			.fold(
				|| vec![RoundEvals2::default(); sums.len()],
				|mut packed_prime_evals: Vec<RoundEvals2<P>>, chunk_index| {
					let eq_chunk = self.gruen32.eq_expansion().chunk(chunk_vars, chunk_index);

					for (round_evals, (evals_a, evals_b)) in
						izip!(&mut packed_prime_evals, self.multilinears.iter().tuples())
					{
						let (evals_a_0, evals_a_1) = evals_a.split_half_ref();
						let (evals_b_0, evals_b_1) = evals_b.split_half_ref();

						let evals_a_0_chunk = evals_a_0.chunk(chunk_vars, chunk_index);
						let evals_b_0_chunk = evals_b_0.chunk(chunk_vars, chunk_index);
						let evals_a_1_chunk = evals_a_1.chunk(chunk_vars, chunk_index);
						let evals_b_1_chunk = evals_b_1.chunk(chunk_vars, chunk_index);

						for (&eq_i, &evals_a_0_i, &evals_b_0_i, &evals_a_1_i, &evals_b_1_i) in izip!(
							eq_chunk.as_ref(),
							evals_a_0_chunk.as_ref(),
							evals_b_0_chunk.as_ref(),
							evals_a_1_chunk.as_ref(),
							evals_b_1_chunk.as_ref()
						) {
							let evals_a_inf_i = evals_a_0_i + evals_a_1_i;
							let evals_b_inf_i = evals_b_0_i + evals_b_1_i;

							round_evals.y_1 += eq_i * evals_a_1_i * evals_b_1_i;
							round_evals.y_inf += eq_i * evals_a_inf_i * evals_b_inf_i;
						}
					}

					packed_prime_evals
				},
			)
			.reduce(
				|| vec![RoundEvals2::default(); sums.len()],
				|lhs, rhs| izip!(lhs, rhs).map(|(l, r)| l + &r).collect(),
			);

        let alpha = self.gruen32.next_coordinate();
        let round_coeffs = izip!(sums, packed_prime_evals)
            .map(|(&sum, packed_evals)| {
                let round_evals = packed_evals.sum_scalars(self.n_vars());
                round_evals.interpolate_eq(sum, alpha)
            })
            .collect::<Vec<_>>();

        self.last_coeffs_or_sums = RoundCoeffsOrSums::Coeffs(round_coeffs.clone());
        Ok(round_coeffs)
    }

    fn fold(&mut self, challenge: F) -> Result<(), Error> {
        let RoundCoeffsOrSums::Coeffs(prime_coeffs) = &self.last_coeffs_or_sums else {
            return Err(Error::ExpectedExecute);
        };

        assert!(self.n_vars() > 0);

        let sums = prime_coeffs
            .iter()
            .map(|coeffs| coeffs.evaluate(challenge))
            .collect();

        self.multilinears
            .par_iter_mut()
            .for_each(|multilinear| fold_highest_var_inplace(multilinear, challenge));

        self.gruen32.fold(challenge);
        self.last_coeffs_or_sums = RoundCoeffsOrSums::Sums(sums);
        Ok(())
    }

    fn finish(self) -> Result<Vec<F>, Error> {
        if self.n_vars() > 0 {
            let error = match self.last_coeffs_or_sums {
                RoundCoeffsOrSums::Coeffs(_) => Error::ExpectedFold,
                RoundCoeffsOrSums::Sums(_) => Error::ExpectedExecute,
            };

            return Err(error);
        }

        let multilinear_evals = self
            .multilinears
            .into_iter()
            .map(|multilinear| multilinear.get(0))
            .collect();

        Ok(multilinear_evals)
    }
}

impl<F, P> MleCheckProver<F> for BivariateProductMultiMlecheckProver<P>
where
    F: Field,
    P: PackedField<Scalar = F>,
{
    fn eval_point(&self) -> &[F] {
        self.gruen32.eval_point()
    }
}

enum RoundCoeffsOrSums<F: Field> {
    Coeffs(Vec<RoundCoeffs<F>>),
    Sums(Vec<F>),
}

#[cfg(test)]
mod tests {
    use std::iter::repeat_with;

    use binius_field::{FieldOps, Random};
    use binius_math::{
        multilinear::{eq::eq_ind_partial_eval, evaluate::evaluate_inplace},
        test_utils::{Packed128b, random_field_buffer, random_scalars},
    };
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;
    use crate::sumcheck::{MleToSumCheckDecorator, bivariate_product_mle};

    #[test]
    fn test_bivariate_multi_mlecheck_conforms_to_single_bivariate_mlecheck() {
        type P = Packed128b;
        type F = <P as FieldOps>::Scalar;

        let n_vars = 8;
        let mut rng = StdRng::seed_from_u64(0);

        // Generate two random multilinear polynomials
        let multilinear_a = random_field_buffer::<P>(&mut rng, n_vars);
        let multilinear_b = random_field_buffer::<P>(&mut rng, n_vars);

        // Compute product on the hypercube
        let product = itertools::zip_eq(multilinear_a.as_ref(), multilinear_b.as_ref())
            .map(|(&l, &r)| l * r)
            .collect_vec();
        let product_buffer = FieldBuffer::new(n_vars, product);

        // Claim eval point
        let eval_point = random_scalars::<F>(&mut rng, n_vars);
        let eval_claim = evaluate_inplace(product_buffer, &eval_point);

        let multilinears = [multilinear_a, multilinear_b];

        let mut single_prover =
            bivariate_product_mle::new(multilinears.clone(), eval_point.clone(), eval_claim)
                .unwrap();

        let mut multi_prover = BivariateProductMultiMlecheckProver::new(
            [multilinears].to_vec(),
            &eval_point,
            vec![eval_claim],
        )
        .unwrap();

        for _round in 0..n_vars {
            let round_coeffs_single = single_prover.execute().unwrap();
            let round_coeffs_multi = multi_prover.execute().unwrap();

            assert_eq!(round_coeffs_single, round_coeffs_multi);

            let challenge = F::random(&mut rng);
            single_prover.fold(challenge).unwrap();
            multi_prover.fold(challenge).unwrap();
        }

        let multilinear_evals_single = single_prover.finish().unwrap();
        let multilinear_evals_multi = multi_prover.finish().unwrap();
        assert_eq!(multilinear_evals_single, multilinear_evals_multi);
    }

    fn test_bivariate_product_multi_mlecheck_consistency_helper<
        F: Field,
        P: PackedField<Scalar = F>,
    >(
        n_vars: usize,
        n_pairs: usize,
    ) {
        // Bivariate product multiplied by equality indicator
        let degree = 3;
        let mut rng = StdRng::seed_from_u64(0);

        // Validate round polynomials by evaluating them at degree + 1 random points
        let samples = random_scalars::<F>(&mut rng, degree + 1);

        // Claim eval point
        let eval_point = random_scalars::<F>(&mut rng, n_vars);

        // A copy of 2 * n_pairs + 1 multilinears for reference logic
        let mut folded_multilinears = repeat_with(|| random_field_buffer::<P>(&mut rng, n_vars))
            .take(n_pairs * 2)
            .collect_vec();

        // Witness copy for the prover
        let (pairs, remainder) = folded_multilinears.as_chunks::<2>();
        assert_eq!(remainder.len(), 0);

        let multilinears = pairs.iter().cloned().collect_vec();

        // Compute MLE of the product
        let eval_claims = multilinears
            .iter()
            .map(|[l, r]| {
                let product = itertools::zip_eq(l.as_ref(), r.as_ref())
                    .map(|(&l, &r)| l * r)
                    .collect_vec();
                let product_buffer = FieldBuffer::new(n_vars, product);
                evaluate_inplace(product_buffer, &eval_point)
            })
            .collect_vec();

        let mlecheck_prover =
            BivariateProductMultiMlecheckProver::new(multilinears, &eval_point, eval_claims)
                .unwrap();
        let mut prover = MleToSumCheckDecorator::new(mlecheck_prover);

        // Append eq indicator at the end
        folded_multilinears.push(eq_ind_partial_eval(&eval_point));

        // Compare  mlecheck prover and naive degree-3 sumcheck by sampling
        // the round polynomials at multiple random points
        for n_vars_remaining in (1..=n_vars).rev() {
            // Round polynomials from the prover
            let coeffs = prover.execute().unwrap();

            // Sample the witness at different points (in specialized variable)
            for &sample in &samples {
                let sample_broadcast = P::broadcast(sample);
                let lerps = folded_multilinears
                    .iter()
                    .map(|multilinear| {
                        let (evals_0, evals_1) = multilinear.split_half_ref();
                        izip!(evals_0.as_ref(), evals_1.as_ref())
                            .map(|(&eval_0, &eval_1)| eval_0 + (eval_1 - eval_0) * sample_broadcast)
                            .collect_vec()
                    })
                    .collect_vec();

                assert_eq!(lerps.len(), 2 * n_pairs + 1);
                let (eq_ind, pairs) = lerps.split_last().unwrap();

                // Naive sum computation at the sample point
                for ((l, r), coeffs) in izip!(pairs.iter().tuples(), &coeffs) {
                    assert_eq!(coeffs.0.len(), degree + 1);
                    let eval = izip!(eq_ind, l, r)
                        .map(|(&eq, &l, &r)| eq * l * r)
                        .sum::<P>()
                        .iter()
                        .take(1 << n_vars_remaining)
                        .sum::<F>();

                    // Test conformance to the mlecheck round polynomial
                    assert_eq!(eval, coeffs.evaluate(sample));
                }
            }

            let challenge = F::random(&mut rng);
            prover.fold(challenge).unwrap();

            for folded in &mut folded_multilinears {
                fold_highest_var_inplace(folded, challenge);
            }
        }

        let multilinear_evals = prover.finish().unwrap();
        assert_eq!(multilinear_evals.len(), n_pairs * 2);
    }

    #[test]
    fn test_bivariate_product_multi_mlecheck_consistency() {
        for (n_vars, n_pairs) in [(0, 0), (0, 4), (1, 5), (7, 1), (3, 3)] {
            test_bivariate_product_multi_mlecheck_consistency_helper::<_, Packed128b>(
                n_vars, n_pairs,
            );
        }
    }
}
