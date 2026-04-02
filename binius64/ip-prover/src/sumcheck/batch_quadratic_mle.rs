// Copyright 2025-2026 The Binius Developers

use std::cmp::max;

use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{
    AsSlicesMut, FieldBuffer, FieldSliceMut, multilinear::fold::fold_highest_var_inplace,
};
use binius_utils::rayon::prelude::*;
use itertools::{Itertools, izip};

use crate::sumcheck::{
    Error,
    common::{MleCheckProver, SumcheckProver},
    gruen32::Gruen32,
    round_evals::RoundEvals2,
};

/// Batch MLE-check prover for M quadratic compositions over N multilinears.
///
/// This prover runs a single sumcheck instance that amortizes the work of M independent
/// quadratic MLE checks by evaluating all compositions in one pass per round. It uses the
/// same Gruen32-style degree-2 interpolation trick as the single-claim prover but keeps
/// a vector of claims/round polynomials and folds all multilinears in lockstep.
pub struct BatchQuadraticMleCheckProver<
    P: PackedField,
    Composition,
    InfinityComposition,
    const N: usize,
    const M: usize,
> {
    // Packed evaluations of the input multilinears; mutated in-place during folding.
    multilinears: Box<dyn AsSlicesMut<P, N> + Send>,
    // Full quadratic composition evaluated on the "x = 1" branch for each multilinear.
    composition: Composition,
    // Composition restricted to highest-degree terms for the "x = ∞" evaluation (Karatsuba).
    infinity_composition: InfinityComposition,
    // State machine storage: last round's evals (execute input) or current coeffs (fold input).
    last_coeffs_or_eval: RoundCoeffsOrEvals<P::Scalar, M>,
    // Tracks the eq-indicator expansion and evaluation-point folding across rounds.
    gruen32: Gruen32<P>,
}

impl<F, P, Composition, InfinityComposition, const N: usize, const M: usize>
    BatchQuadraticMleCheckProver<P, Composition, InfinityComposition, N, M>
where
    F: Field,
    P: PackedField<Scalar = F>,
    Composition: Fn([P; N], &mut [P; M]) + Sync,
    InfinityComposition: Fn([P; N], &mut [P; M]) + Sync,
{
    pub fn new(
        mut multilinears: impl AsSlicesMut<P, N> + Send + 'static,
        composition: Composition,
        infinity_composition: InfinityComposition,
        eval_point: Vec<F>,
        eval_claims: [F; M],
    ) -> Result<Self, Error> {
        let n_vars = eval_point.len();
        assert!(N > 0 && M > 0);
        for multilinear in &multilinears.as_slices_mut() {
            // All multilinears must agree on the number of variables for consistent folding.
            if multilinear.log_len() != n_vars {
                return Err(Error::MultilinearSizeMismatch);
            }
        }

        // The first execute round consumes the claimed composite evaluations at eval_point.
        let last_coeffs_or_eval = RoundCoeffsOrEvals::Evals(eval_claims);
        // Gruen32 owns the eq-indicator expansion tied to eval_point and drives per-round folding.
        let gruen32 = Gruen32::new(&eval_point);

        Ok(Self {
            multilinears: Box::new(multilinears),
            composition,
            infinity_composition,
            last_coeffs_or_eval,
            gruen32,
        })
    }

    /// Gets mutable slices of the multilinears, truncated to the current number of variables.
    ///
    /// Truncation keeps the in-place folds consistent with the remaining variables tracked by
    /// Gruen32 and avoids touching irrelevant higher-dimensional data.
    fn multilinears_mut(&mut self) -> [FieldSliceMut<'_, P>; N] {
        let n_vars = self.gruen32.n_vars_remaining();
        let mut slices = self.multilinears.as_slices_mut();
        for slice in &mut slices {
            slice.truncate(n_vars);
        }
        slices
    }
}

impl<F, P, Composition, InfinityComposition, const N: usize, const M: usize> SumcheckProver<F>
    for BatchQuadraticMleCheckProver<P, Composition, InfinityComposition, N, M>
where
    F: Field,
    P: PackedField<Scalar = F>,
    Composition: Fn([P; N], &mut [P; M]) + Sync,
    InfinityComposition: Fn([P; N], &mut [P; M]) + Sync,
{
    fn n_vars(&self) -> usize {
        self.gruen32.n_vars_remaining()
    }

    fn n_claims(&self) -> usize {
        M
    }

    fn execute(&mut self) -> Result<Vec<RoundCoeffs<F>>, Error> {
        // State machine: execute consumes evals from the previous round and produces new coeffs.
        let last_eval = match &self.last_coeffs_or_eval {
            RoundCoeffsOrEvals::Evals(evals) => *evals,
            RoundCoeffsOrEvals::Coeffs(_) => return Err(Error::ExpectedFold),
        };

        // There must be at least one variable left to sum over in this round.
        let n_vars_remaining = self.gruen32.n_vars_remaining();
        assert!(n_vars_remaining > 0);

        // eq_expansion corresponds to the equality indicator over the remaining variables,
        // already folded on the current variable, so it has one fewer dimension.
        let eq_expansion = self.gruen32.eq_expansion();
        assert_eq!(eq_expansion.log_len(), n_vars_remaining - 1);

        // Local bindings keep names compact in the hot loop without changing borrow scopes.
        let comp = &self.composition;
        let inf_comp = &self.infinity_composition;

        // Get multilinear slices and truncate to current n_vars.
        // This is inlined here (instead of multilinears_mut) to keep the borrow local to execute.
        let mut multilinears = self.multilinears.as_slices_mut();
        for slice in &mut multilinears {
            slice.truncate(n_vars_remaining);
        }
        // Split each multilinear into low/high halves for the top variable:
        // the low half corresponds to x=0, the high half to x=1.
        let (splits_0, splits_1) = multilinears
            .iter()
            .map(FieldBuffer::split_half_ref)
            .collect::<(Vec<_>, Vec<_>)>();

        // Perform chunked summation: for every row, evaluate all compositions and add up
        // results to an array of round evals accumulators. Alternative would be to sum each
        // composition on its own pass, but that would require reading the entirety of eq field
        // buffer on each pass, which will evict the latter from the cache. By doing chunked
        // compute, we reasonably hope that eq chunk always stays in L1 cache.
        const MAX_CHUNK_VARS: usize = 8;
        // Choose a chunk width that fits cache while still honoring packed width.
        let chunk_vars = max(MAX_CHUNK_VARS, P::LOG_WIDTH).min(n_vars_remaining - 1);
        let chunk_count = 1 << (n_vars_remaining - 1 - chunk_vars);

        // Parallel-reduce across eq chunks to amortize composition evaluation.
        // Each worker accumulates packed y_1 / y_inf values for all M claims.
        let packed_prime_evals = (0..chunk_count)
            .into_par_iter()
            .try_fold(
                || [[P::default(); M]; 2],
                |mut packed_prime_evals, chunk_index| -> Result<_, Error> {
                    let eq_chunk = eq_expansion.chunk(chunk_vars, chunk_index);

                    // Scratch buffers are reused per row to avoid allocations in the hot loop.
                    let [mut y_1_scratch, mut y_inf_scratch] = [[P::default(); M]; 2];
                    let splits_0_chunk = splits_0
                        .iter()
                        .map(|slice| slice.chunk(chunk_vars, chunk_index))
                        .collect::<Vec<_>>();
                    let splits_1_chunk = splits_1
                        .iter()
                        .map(|slice| slice.chunk(chunk_vars, chunk_index))
                        .collect::<Vec<_>>();

                    // Accumulate packed evals for this chunk; first index is y_1/y_inf.
                    let [y_1, y_inf] = &mut packed_prime_evals;
                    for (idx, &eq_i) in eq_chunk.as_ref().iter().enumerate() {
                        // Gather the idx-th evaluations of every multilinear at both halves.
                        let mut evals_1 = [P::default(); N];
                        let mut evals_inf = [P::default(); N];

                        for i in 0..N {
                            let lo_i = splits_0_chunk[i].as_ref()[idx];
                            let hi_i = splits_1_chunk[i].as_ref()[idx];

                            // Compose once with the high half and once with the lo+hi combination.
                            // The lo+hi branch corresponds to evaluation at infinity for
                            // multilinears.
                            evals_1[i] = hi_i;
                            evals_inf[i] = lo_i + hi_i;
                        }

                        // Apply the compositions for this equality term.
                        comp(evals_1, &mut y_1_scratch);
                        inf_comp(evals_inf, &mut y_inf_scratch);

                        for i in 0..M {
                            // Weight by eq indicator to keep the sumcheck claim aligned to
                            // eval_point.
                            y_1[i] += y_1_scratch[i] * eq_i;
                            y_inf[i] += y_inf_scratch[i] * eq_i;
                        }
                    }

                    Ok(packed_prime_evals)
                },
            )
            .try_reduce(
                || [[P::default(); M]; 2],
                |lhs, rhs| {
                    let mut out = [[P::default(); M]; 2];
                    for claim_idx in 0..M {
                        out[0][claim_idx] = lhs[0][claim_idx] + rhs[0][claim_idx];
                        out[1][claim_idx] = lhs[1][claim_idx] + rhs[1][claim_idx];
                    }
                    Ok(out)
                },
            )?;

        // Sample the next coordinate and interpolate each round polynomial.
        // The coordinate ties this round's sum to the original evaluation point.
        let alpha = self.gruen32.next_coordinate();
        let round_coeffs = izip!(
            last_eval.iter().copied(),
            packed_prime_evals[0].iter().copied(),
            packed_prime_evals[1].iter().copied()
        )
        .map(|(sum, y_1, y_inf)| {
            // Sum packed values into scalars, then interpolate using the expected sum.
            // sum_scalars collapses packed lanes down to scalar totals for this round.
            let round_evals = RoundEvals2 { y_1, y_inf }.sum_scalars(n_vars_remaining);
            round_evals.interpolate_eq(sum, alpha)
        })
        .collect::<Vec<_>>();
        // State transition: execute produces coeffs for fold to consume.
        self.last_coeffs_or_eval = RoundCoeffsOrEvals::Coeffs(
            round_coeffs
                .clone()
                .try_into()
                .expect("Will have M elements."),
        );
        Ok(round_coeffs)
    }

    fn fold(&mut self, challenge: F) -> Result<(), Error> {
        // State machine: fold consumes coeffs and produces evals at the verifier challenge.
        let RoundCoeffsOrEvals::Coeffs(prime_coeffs) = &self.last_coeffs_or_eval else {
            return Err(Error::ExpectedExecute);
        };

        // n_vars is decremented in fold, so we must have at least one variable left here.
        assert!(
            self.n_vars() > 0,
            "n_vars is decremented in fold; \
			fold changes last_coeffs_or_eval to Eval variant; \
			fold only executes with Coeffs variant; \
			thus, n_vars should be > 0"
        );

        // Evaluate each round polynomial at the verifier's challenge to form the next sum claim.
        let evals = prime_coeffs
            .iter()
            .map(|coeffs| coeffs.evaluate(challenge))
            .collect_array()
            .expect("Will have size M");

        // Fold all multilinears on the highest variable using the same challenge.
        // This matches the verifier's restriction of the random point.
        for multilinear in &mut self.multilinears_mut() {
            fold_highest_var_inplace(multilinear, challenge);
        }

        // Keep the equality polynomial in sync with the folding of multilinears.
        self.gruen32.fold(challenge);
        // State transition: fold produces evals for the next execute.
        self.last_coeffs_or_eval = RoundCoeffsOrEvals::Evals(evals);
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<F>, Error> {
        // Finish is only valid after all folds complete (i.e., zero variables remain).
        if self.n_vars() > 0 {
            let error = match self.last_coeffs_or_eval {
                RoundCoeffsOrEvals::Coeffs(_) => Error::ExpectedFold,
                RoundCoeffsOrEvals::Evals(_) => Error::ExpectedExecute,
            };

            return Err(error);
        }

        // With no variables remaining, each multilinear has length 1 and can be read directly.
        // These are the evaluations at the verifier's random point.
        let multilinear_evals = self
            .multilinears_mut()
            .into_iter()
            .map(|multilinear| multilinear.get(0))
            .collect();

        Ok(multilinear_evals)
    }
}

impl<F, P, Composition, InfinityComposition, const N: usize, const M: usize> MleCheckProver<F>
    for BatchQuadraticMleCheckProver<P, Composition, InfinityComposition, N, M>
where
    F: Field,
    P: PackedField<Scalar = F>,
    Composition: Fn([P; N], &mut [P; M]) + Sync,
    InfinityComposition: Fn([P; N], &mut [P; M]) + Sync,
{
    fn eval_point(&self) -> &[F] {
        &self.gruen32.eval_point()[..self.n_vars()]
    }
}

#[derive(Debug, Clone)]
enum RoundCoeffsOrEvals<F: Field, const M: usize> {
    Coeffs([RoundCoeffs<F>; M]),
    Evals([F; M]),
}

#[cfg(test)]
mod tests {
    use std::array;

    use binius_field::{FieldOps, Random};
    use binius_math::{
        FieldBuffer,
        multilinear::{
            eq::eq_ind_partial_eval, evaluate::evaluate_inplace, fold::fold_highest_var_inplace,
        },
        test_utils::{Packed128b, random_field_buffer, random_scalars},
    };
    use itertools::{Itertools, izip};
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;
    use crate::sumcheck::{MleToSumCheckDecorator, quadratic_mle::QuadraticMleCheckProver};

    const N: usize = 3;
    const M: usize = 2;
    type CompFn<P> = fn([P; N]) -> P;

    // Test composition functions
    fn comp_0<P: PackedField>([a, b, c]: [P; N]) -> P {
        a * b - c
    }

    fn inf_comp_0<P: PackedField>([a, b, _c]: [P; N]) -> P {
        a * b
    }

    fn comp_1<P: PackedField>([a, b, c]: [P; N]) -> P {
        (a + b) * c
    }

    fn inf_comp_1<P: PackedField>([a, b, c]: [P; N]) -> P {
        (a + b) * c
    }

    fn batch_comp<P: PackedField>(evals: [P; N], out: &mut [P; M]) {
        let [a, b, c] = evals;
        out[0] = a * b - c;
        out[1] = (a + b) * c;
    }

    fn batch_inf_comp<P: PackedField>(evals: [P; N], out: &mut [P; M]) {
        let [a, b, c] = evals;
        out[0] = a * b;
        out[1] = (a + b) * c;
    }

    // Generates evaluation claims for an array of multilinears.
    fn eval_claims<F, P>(multilinears: &[FieldBuffer<P>; N], eval_point: &[F]) -> [F; M]
    where
        F: Field,
        P: PackedField<Scalar = F>,
    {
        let n_vars = eval_point.len();
        array::from_fn(|claim_idx| {
            let composite_vals = (0..1 << n_vars.saturating_sub(P::LOG_WIDTH))
                .map(|i| {
                    let evals = array::from_fn(|j| multilinears[j].as_ref()[i]);
                    match claim_idx {
                        0 => comp_0(evals),
                        1 => comp_1(evals),
                        _ => unreachable!("M is fixed to 2"),
                    }
                })
                .collect_vec();
            let composite_buffer = FieldBuffer::new(n_vars, composite_vals);
            evaluate_inplace(composite_buffer, eval_point)
        })
    }

    #[test]
    fn test_batch_quadratic_mlecheck_conforms_to_single_quadratic_mlecheck() {
        type P = Packed128b;
        type F = <P as FieldOps>::Scalar;

        let n_vars = 8;
        let mut rng = StdRng::seed_from_u64(0);

        // Create arbitrary multilinears.
        let multilinears: [FieldBuffer<P>; N] =
            array::from_fn(|_| random_field_buffer::<P>(&mut rng, n_vars));
        let eval_point = random_scalars::<F>(&mut rng, n_vars);
        let eval_claims = eval_claims::<F, P>(&multilinears, &eval_point);

        let mut single_provers: [QuadraticMleCheckProver<P, CompFn<P>, CompFn<P>, N>; M] = [
            QuadraticMleCheckProver::new(
                multilinears.clone(),
                comp_0::<P> as CompFn<P>,
                inf_comp_0::<P> as CompFn<P>,
                eval_point.clone(),
                eval_claims[0],
            )
            .unwrap(),
            QuadraticMleCheckProver::new(
                multilinears.clone(),
                comp_1::<P> as CompFn<P>,
                inf_comp_1::<P> as CompFn<P>,
                eval_point.clone(),
                eval_claims[1],
            )
            .unwrap(),
        ];

        let mut batch_prover = BatchQuadraticMleCheckProver::new(
            multilinears.clone(),
            batch_comp::<P>,
            batch_inf_comp::<P>,
            eval_point,
            eval_claims,
        )
        .unwrap();

        for _round in 0..n_vars {
            let round_coeffs_batch = batch_prover.execute().unwrap();
            assert_eq!(round_coeffs_batch.len(), M);

            for (single, batch_coeffs) in single_provers.iter_mut().zip(round_coeffs_batch.iter()) {
                let round_coeffs_single = single.execute().unwrap();
                assert_eq!(round_coeffs_single[0], *batch_coeffs);
            }

            let challenge = F::random(&mut rng);
            for single in &mut single_provers {
                single.fold(challenge).unwrap();
            }
            batch_prover.fold(challenge).unwrap();
        }

        let [single_0, single_1] = single_provers;
        let multilinear_evals_0 = single_0.finish().unwrap();
        let multilinear_evals_1 = single_1.finish().unwrap();
        let multilinear_evals_batch = batch_prover.finish().unwrap();

        assert_eq!(multilinear_evals_0, multilinear_evals_batch);
        assert_eq!(multilinear_evals_1, multilinear_evals_batch);
    }

    fn test_batch_quadratic_mlecheck_consistency_helper<F, P>(n_vars: usize)
    where
        F: Field,
        P: PackedField<Scalar = F>,
    {
        let degree = 3;
        let mut rng = StdRng::seed_from_u64(0);

        let samples = random_scalars::<F>(&mut rng, degree + 1);
        let eval_point = random_scalars::<F>(&mut rng, n_vars);

        let multilinears: [FieldBuffer<P>; N] =
            array::from_fn(|_| random_field_buffer::<P>(&mut rng, n_vars));
        let eval_claims = eval_claims::<F, P>(&multilinears, &eval_point);

        let mlecheck_prover = BatchQuadraticMleCheckProver::new(
            multilinears.clone(),
            batch_comp::<P>,
            batch_inf_comp::<P>,
            eval_point.clone(),
            eval_claims,
        )
        .unwrap();

        let mut prover = MleToSumCheckDecorator::new(mlecheck_prover);

        let mut folded_multilinears = multilinears.to_vec();
        folded_multilinears.push(eq_ind_partial_eval(&eval_point));

        for n_vars_remaining in (1..=n_vars).rev() {
            let coeffs = prover.execute().unwrap();
            assert_eq!(coeffs.len(), M);

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

                assert_eq!(lerps.len(), N + 1);
                let (eq_ind, evals) = lerps.split_last().unwrap();

                for (claim_idx, round_coeffs) in coeffs.iter().enumerate() {
                    let packed_eval = (0..eq_ind.len())
                        .map(|idx| {
                            let evals_at_idx = array::from_fn(|j| evals[j][idx]);
                            let composed = match claim_idx {
                                0 => comp_0(evals_at_idx),
                                1 => comp_1(evals_at_idx),
                                _ => unreachable!("M is fixed to 2"),
                            };
                            eq_ind[idx] * composed
                        })
                        .sum::<P>();

                    let eval = packed_eval.iter().take(1 << n_vars_remaining).sum::<F>();
                    assert_eq!(eval, round_coeffs.evaluate(sample));
                }
            }

            let challenge = F::random(&mut rng);
            prover.fold(challenge).unwrap();
            for folded in &mut folded_multilinears {
                fold_highest_var_inplace(folded, challenge);
            }
        }

        let multilinear_evals = prover.finish().unwrap();
        assert_eq!(multilinear_evals.len(), N);
    }

    #[test]
    fn test_batch_quadratic_mlecheck_consistency() {
        for n_vars in [0, 1, 3, 7] {
            test_batch_quadratic_mlecheck_consistency_helper::<_, Packed128b>(n_vars);
        }
    }
}
