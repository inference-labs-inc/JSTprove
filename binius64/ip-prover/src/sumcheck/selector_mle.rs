// Copyright 2023-2025 Irreducible Inc.

#![allow(dead_code)]

use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{FieldBuffer, multilinear::fold::fold_highest_var_inplace};
use binius_utils::{bitwise::Bitwise, rayon::prelude::*};
use itertools::izip;

use super::{
    common::SumcheckProver, error::Error, gruen32::Gruen32, round_evals::RoundEvals2,
    switchover::BinarySwitchover,
};

pub struct Claim<F: Field> {
    pub point: Vec<F>,
    pub value: F,
}

/// A [`SumcheckProver`] implementation that proves an mlecheck over many compositions of the
/// form `selected * selector + (1 - selector)`, where `selected` is the shared large field
/// multilinear and `selector` comes from the set of 1-bit multilinears. Unlike other multi mlecheck
/// provers however the evaluation point is _not_ shared but is specified per selector.
///
/// The set of 1-bit multilinears is represented by a power-of-two long slice of bitmasks, and the
/// multilinear set is constructed by arranging the bitmasks as a 2D matrix in row-major order and
/// taking vertical slices. This representation is very compact and has no embedding overhead.
///
/// To combat memory blowup issues arising from folding 1-bit multilinears, this prover introduces
/// switchover. See `BinarySwitchover` for more in-depth explanation of the mechanism. Also note
/// that the need to expand the equality indicator for each multilinear still results in some
/// blowup.
pub struct SelectorMlecheckProver<'b, P: PackedField, B: Bitwise> {
    last_coeffs_or_sums: RoundCoeffsOrSums<P::Scalar>,
    selected: FieldBuffer<P>,
    gruen32s: Vec<Gruen32<P>>,
    switchover: BinarySwitchover<'b, P, B>,
}

impl<'b, F: Field, P: PackedField<Scalar = F>, B: Bitwise> SelectorMlecheckProver<'b, P, B> {
    /// Constructs a prover, given `bitmasks` as representation of 1-bit columns, `selected` being
    /// the shared large field multilinear, individual `claims` per selector and `switchover` as
    /// the round at which 1-bit columns should be folded.
    pub fn new(
        selected: FieldBuffer<P>,
        claims: Vec<Claim<F>>,
        bitmasks: &'b [B],
        switchover: usize,
    ) -> Result<Self, Error> {
        let n_vars = selected.log_len();

        if claims.iter().any(|claim| claim.point.len() != n_vars) {
            return Err(Error::MultilinearSizeMismatch);
        }

        if bitmasks.len() != selected.len() {
            return Err(Error::BitmasksSizeMismatch);
        }

        const MAX_CHUNK_VARS: usize = 8;
        let (gruen32s, sums) = claims
            .into_par_iter()
            .map(|Claim { point, value }| (Gruen32::new_with_suffix(MAX_CHUNK_VARS, &point), value))
            .collect::<(Vec<_>, Vec<_>)>();

        let switchover = BinarySwitchover::new(sums.len(), switchover.min(n_vars), bitmasks);
        let last_coeffs_or_sums = RoundCoeffsOrSums::Sums(sums);

        Ok(Self {
            last_coeffs_or_sums,
            selected,
            gruen32s,
            switchover,
        })
    }
}

impl<'b, F, P, B> SumcheckProver<F> for SelectorMlecheckProver<'b, P, B>
where
    F: Field,
    P: PackedField<Scalar = F>,
    B: Bitwise,
{
    fn n_vars(&self) -> usize {
        self.selected.log_len()
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
        // compute, we reasonably hope that eq chunk always stays in L1 cache. We can also
        // leverage the outer product representation of the eq indicator in the Gruen32 struct.
        //
        // We also do switchover there, which by definition requires small scratchpads to hold
        // large field partial evaluations of the transparent multilinears.
        let chunk_vars = self
            .gruen32s
            .first()
            .map(|gruen32| gruen32.chunk_eq_expansion().log_len())
            .unwrap_or_default();
        let chunk_count = 1 << (self.n_vars() - 1 - chunk_vars);

        let packed_prime_evals =
            (0..chunk_count)
                .into_par_iter()
                .fold(
                    || {
                        (
                            vec![RoundEvals2::default(); sums.len()],
                            FieldBuffer::<P>::zeros(chunk_vars),
                            FieldBuffer::<P>::zeros(chunk_vars),
                        )
                    },
                    |(mut packed_prime_evals, mut binary_chunk_0, mut binary_chunk_1),
                     chunk_index| {
                        let (selected_0, selected_1) = self.selected.split_half_ref();

                        let selected_0_chunk = selected_0.chunk(chunk_vars, chunk_index);
                        let selected_1_chunk = selected_1.chunk(chunk_vars, chunk_index);

                        for (bit_offset, (round_evals, gruen32)) in
                            izip!(&mut packed_prime_evals, &self.gruen32s).enumerate()
                        {
                            let eq_chunk = gruen32.chunk_eq_expansion();
                            let eq_suffix_eval = gruen32.suffix_eq_expansion().get(chunk_index);

                            let selector_0_chunk = self.switchover.get_chunk(
                                &mut binary_chunk_0,
                                bit_offset,
                                chunk_vars,
                                chunk_index,
                            );

                            let selector_1_chunk = self.switchover.get_chunk(
                                &mut binary_chunk_1,
                                bit_offset,
                                chunk_vars,
                                chunk_index | chunk_count,
                            );

                            let mut chunk_round_evals = RoundEvals2::default();
                            for (
                                &eq_i,
                                &selected_0_i,
                                &selected_1_i,
                                &selector_0_i,
                                &selector_1_i,
                            ) in izip!(
                                eq_chunk.as_ref(),
                                selected_0_chunk.as_ref(),
                                selected_1_chunk.as_ref(),
                                selector_0_chunk.as_ref(),
                                selector_1_chunk.as_ref(),
                            ) {
                                let selected_inf_i = selected_0_i + selected_1_i;
                                let selector_inf_i = selector_0_i + selector_1_i;

                                // selected * selector + (1 - selector)
                                // @one: selector * (selected - 1) + 1
                                // @inf: selector * selected (note that lower degree terms are dropped)
                                chunk_round_evals.y_1 +=
                                    eq_i * (selector_1_i * (selected_1_i - P::one()) + P::one());
                                chunk_round_evals.y_inf += eq_i * selector_inf_i * selected_inf_i;
                            }

                            // Apply the common factor from the outer product representation of the eq
                            // ind
                            *round_evals += &(chunk_round_evals * eq_suffix_eval);
                        }

                        (packed_prime_evals, binary_chunk_0, binary_chunk_1)
                    },
                )
                .map(|(evals, _, _)| evals)
                .reduce(
                    || vec![RoundEvals2::<P>::default(); sums.len()],
                    |lhs, rhs| izip!(lhs, rhs).map(|(l, r)| l + &r).collect(),
                );

        // This prover has multiple evaluation points and cannot implement MleCheckProver.
        let (prime_coeffs, round_coeffs) = izip!(&self.gruen32s, sums, packed_prime_evals)
            .map(|(gruen32, &sum, packed_prime_evals)| {
                gruen32.interpolate2(sum, packed_prime_evals.sum_scalars(self.n_vars() - 1))
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        self.last_coeffs_or_sums = RoundCoeffsOrSums::Coeffs(prime_coeffs);
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

        self.gruen32s
            .par_iter_mut()
            .for_each(|gruen32| gruen32.fold(challenge));

        self.switchover.fold(challenge);
        fold_highest_var_inplace(&mut self.selected, challenge);

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

        let mut multilinear_evals = Vec::with_capacity(self.gruen32s.len() + 1);

        for selector in self.switchover.finalize() {
            debug_assert_eq!(selector.log_len(), 0);
            let eval = selector.get(0);
            multilinear_evals.push(eval);
        }

        debug_assert_eq!(self.selected.log_len(), 0);
        multilinear_evals.push(self.selected.get(0));

        Ok(multilinear_evals)
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
        multilinear::evaluate::evaluate as multilinear_evaluate,
        test_utils::{Packed128b, random_scalars},
    };
    use itertools::Itertools;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;
    use crate::sumcheck::{
        MleToSumCheckDecorator, bivariate_product_multi_mle::BivariateProductMultiMlecheckProver,
    };

    type P = Packed128b;
    type F = <P as FieldOps>::Scalar;

    #[test]
    fn test_bivariate_mlecheck_conformance() {
        let mut rng = StdRng::seed_from_u64(0);

        let n_vars = 8;
        let selector_count = 3;

        let selector_mask = (1u16 << selector_count) - 1;
        let bitmasks = repeat_with(|| rng.random::<u16>() & selector_mask)
            .take(1 << n_vars)
            .collect_vec();

        // Compare the round polynomials of the SelectorMlecheckProver and sum of round
        // polynomials of two bivariate provers evaluating selector * selected and (1-selector) * 1
        let selected_scalars = random_scalars::<F>(&mut rng, 1 << n_vars);
        let selected = FieldBuffer::<P>::from_values(&selected_scalars);

        let ones_scalars = repeat_with(|| F::ONE).take(1 << n_vars).collect_vec();
        let ones = FieldBuffer::<P>::from_values(&ones_scalars);

        let bivariate_provers_and_claims = (0..selector_count)
            .map(|i| {
                let mut selector_scalars = bitmasks
                    .iter()
                    .map(|b| if (b >> i) & 1 == 1 { F::ONE } else { F::ZERO })
                    .collect_vec();
                let direct_selector = FieldBuffer::<P>::from_values(&selector_scalars);

                let zeroed_selected_scalars = izip!(&selected_scalars, &selector_scalars)
                    .map(|(&selected, &selector)| selected * selector)
                    .collect_vec();
                let zeroed_selected = FieldBuffer::<P>::from_values(&zeroed_selected_scalars);

                for scalar in &mut selector_scalars {
                    *scalar += F::ONE;
                }

                let inverted_selector_scalars = selector_scalars;
                let inverted_selector = FieldBuffer::<P>::from_values(&inverted_selector_scalars);

                let masked_selected_scalars =
                    izip!(&zeroed_selected_scalars, &inverted_selector_scalars)
                        .map(|(&zeroed_selected, &inverted_selector)| {
                            zeroed_selected + inverted_selector
                        })
                        .collect_vec();
                let masked_selected = FieldBuffer::<P>::from_values(&masked_selected_scalars);

                let point = random_scalars::<F>(&mut rng, n_vars);
                let value = multilinear_evaluate(&masked_selected, &point);

                let direct_eval_claim = multilinear_evaluate(&zeroed_selected, &point);
                let direct_mle_prover = BivariateProductMultiMlecheckProver::new(
                    [[direct_selector, selected.clone()]].to_vec(),
                    &point,
                    vec![direct_eval_claim],
                )
                .unwrap();
                let direct_prover = MleToSumCheckDecorator::new(direct_mle_prover);

                let inverted_eval_claim = multilinear_evaluate(&inverted_selector, &point);
                let inverted_mle_prover = BivariateProductMultiMlecheckProver::new(
                    [[inverted_selector, ones.clone()]].to_vec(),
                    &point,
                    vec![inverted_eval_claim],
                )
                .unwrap();
                let inverted_prover = MleToSumCheckDecorator::new(inverted_mle_prover);

                let selector_mlecheck_claim = Claim { point, value };

                ((direct_prover, inverted_prover), selector_mlecheck_claim)
            })
            .collect_vec();

        let (mut bivariate_provers, claims) = bivariate_provers_and_claims
            .into_iter()
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let switchover = 0;
        let mut selector_prover =
            SelectorMlecheckProver::new(selected, claims, &bitmasks, switchover).unwrap();

        for _n_rounds_remaining in (1..=n_vars).rev() {
            // NB: this is unsound, for test usage only!
            let challenge = F::random(&mut rng);

            let all_selector_coeffs = selector_prover.execute().unwrap();
            selector_prover.fold(challenge).unwrap();

            for (selector_coeffs, (direct_prover, inverted_prover)) in
                izip!(all_selector_coeffs, &mut bivariate_provers)
            {
                let direct_coeffs = direct_prover.execute().unwrap();
                let inverted_coeffs = inverted_prover.execute().unwrap();

                direct_prover.fold(challenge).unwrap();
                inverted_prover.fold(challenge).unwrap();

                assert_eq!(direct_coeffs.len(), 1);
                assert_eq!(inverted_coeffs.len(), 1);
                assert_eq!(
                    selector_coeffs,
                    direct_coeffs[0].clone() + &inverted_coeffs[0]
                );
            }
        }
    }
}
