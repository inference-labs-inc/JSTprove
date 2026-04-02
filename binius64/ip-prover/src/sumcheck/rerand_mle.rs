// Copyright 2025 Irreducible Inc.

use std::cmp::max;

use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::FieldBuffer;
use binius_utils::{bitwise::Bitwise, rayon::prelude::*};
use itertools::izip;

use super::{
    common::{MleCheckProver, SumcheckProver},
    error::Error,
    gruen32::Gruen32,
    round_evals::RoundEvals1,
    switchover::BinarySwitchover,
};

/// A [`SumcheckProver`] implementation that can "rerandomize" evaluation claims on a set of 1-bit
/// multilinears, with those claims sharing the evaluation point. This is essentially a degree-1
/// mlecheck.
///
/// The set of 1-bit multilinears is represented by a power-of-two long slice of bitmasks, and the
/// multilinear set is constructed by arranging the bitmasks as a 2D matrix in row-major order and
/// taking vertical slices. This representation is very compact and has no embedding overhead.
///
/// To combat memory blowup issues arising from folding 1-bit multilinears, this prover introduces
/// switchover. See `BinarySwitchover` for more in-depth explanation of the mechanism.
pub struct RerandMlecheckProver<'b, P: PackedField, B: Bitwise> {
    last_coeffs_or_sums: RoundCoeffsOrSums<P::Scalar>,
    gruen32: Gruen32<P>,
    switchover: BinarySwitchover<'b, P, B>,
}

impl<'b, F, P, B> RerandMlecheckProver<'b, P, B>
where
    F: Field,
    P: PackedField<Scalar = F>,
    B: Bitwise,
{
    /// Constructs a prover, given `bitmasks` as representation of 1-bit columns, evaluation claims
    /// `eval_claims` on the shared `eval_point`, and `switchover` being the round at which 1-bit
    /// columns should be folded.
    pub fn new(
        eval_point: &[F],
        eval_claims: &[F],
        bitmasks: &'b [B],
        switchover: usize,
    ) -> Result<Self, Error> {
        let n_vars = eval_point.len();

        if bitmasks.len() != 1 << n_vars {
            return Err(Error::BitmasksSizeMismatch);
        }

        let gruen32 = Gruen32::new(eval_point);
        let switchover = BinarySwitchover::new(eval_claims.len(), switchover.min(n_vars), bitmasks);
        let last_coeffs_or_sums = RoundCoeffsOrSums::Sums(eval_claims.to_vec());

        Ok(Self {
            last_coeffs_or_sums,
            gruen32,
            switchover,
        })
    }
}

impl<'b, F, P, B> SumcheckProver<F> for RerandMlecheckProver<'b, P, B>
where
    F: Field,
    P: PackedField<Scalar = F>,
    B: Bitwise,
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
        //
        // We also do switchover there, which by definition requires small scratchpads to hold
        // large field partial evaluations of the transparent multilinears.
        const MAX_CHUNK_VARS: usize = 8;
        let chunk_vars = max(MAX_CHUNK_VARS, P::LOG_WIDTH).min(self.n_vars() - 1);
        let chunk_count = 1 << (self.n_vars() - 1 - chunk_vars);

        let packed_prime_evals = (0..chunk_count)
            .into_par_iter()
            .fold(
                // A chunk-sized scratchpad is needed for switchover.
                || {
                    (
                        vec![RoundEvals1::default(); sums.len()],
                        FieldBuffer::<P>::zeros(chunk_vars),
                    )
                },
                |(mut packed_prime_evals, mut binary_chunk): (
                    Vec<RoundEvals1<P>>,
                    FieldBuffer<P>,
                ),
                 chunk_index| {
                    let eq_chunk = self.gruen32.eq_expansion().chunk(chunk_vars, chunk_index);

                    for (bit_offset, round_evals) in packed_prime_evals.iter_mut().enumerate() {
                        // Degree-1 composition - evaluate at 1 only
                        let evals_1_chunk = self.switchover.get_chunk(
                            &mut binary_chunk,
                            bit_offset,
                            chunk_vars,
                            chunk_index | chunk_count,
                        );
                        for (&eq_i, &evals_1_i) in izip!(eq_chunk.as_ref(), evals_1_chunk.as_ref())
                        {
                            round_evals.y_1 += eq_i * evals_1_i;
                        }
                    }

                    (packed_prime_evals, binary_chunk)
                },
            )
            .map(|(evals, _)| evals)
            .reduce(
                || vec![RoundEvals1::default(); sums.len()],
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
        let RoundCoeffsOrSums::Coeffs(round_coeffs) = &self.last_coeffs_or_sums else {
            return Err(Error::ExpectedExecute);
        };

        assert!(self.n_vars() > 0);

        let sums = round_coeffs
            .iter()
            .map(|coeffs| coeffs.evaluate(challenge))
            .collect::<Vec<F>>();

        self.switchover.fold(challenge);
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
            .switchover
            .finalize()
            .into_iter()
            .map(|multilinear| {
                debug_assert_eq!(multilinear.log_len(), 0);
                multilinear.get(0)
            })
            .collect();

        Ok(multilinear_evals)
    }
}

impl<'b, F, P, B> MleCheckProver<F> for RerandMlecheckProver<'b, P, B>
where
    F: Field,
    P: PackedField<Scalar = F>,
    B: Bitwise,
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

    use binius_field::{Field, FieldOps, Random};
    use binius_math::{
        multilinear::evaluate::evaluate,
        test_utils::{Packed128b, random_scalars},
    };
    use binius_utils::{bitwise::BitSelector, random_access_sequence::RandomAccessSequence};
    use itertools::Itertools;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;
    use crate::sumcheck::bivariate_product_multi_mle::BivariateProductMultiMlecheckProver;

    type P = Packed128b;
    type F = <P as FieldOps>::Scalar;

    fn test_bivariate_mlecheck_conformance_helper(n_vars: usize, switchover: usize) {
        let mut rng = StdRng::seed_from_u64(0);

        let selector_count = 3;
        let eval_point = random_scalars::<F>(&mut rng, n_vars);

        let selector_mask = (1u16 << selector_count) - 1;
        let bitmasks = repeat_with(|| rng.random::<u16>() & selector_mask)
            .take(1 << n_vars)
            .collect_vec();

        // Compare the round polynomials and final multilinear evals of bivariate product mlecheck
        // and the rerandomization prover. Bivariate product prover is invoked by embedding bits
        // into large field elements and doing a product with a constant multilinear of ones.

        let ones_scalars = repeat_with(|| F::ONE).take(1 << n_vars).collect_vec();
        let ones = FieldBuffer::<P>::from_values(&ones_scalars);

        let selectors_with_claims = (0..selector_count)
            .map(|bit_offset| {
                let bit_selector = BitSelector::new(bit_offset, &bitmasks);

                let selector_scalars = (0..bit_selector.len())
                    .map(|i| if bit_selector.get(i) { F::ONE } else { F::ZERO })
                    .collect_vec();
                let selector = FieldBuffer::<P>::from_values(&selector_scalars);

                let eval_claim = evaluate(&selector, &eval_point);
                (selector, eval_claim)
            })
            .collect_vec();

        let (selectors, eval_claims) = selectors_with_claims
            .into_iter()
            .unzip::<_, _, Vec<_>, Vec<_>>();
        let mut multi_bivariate_prover = BivariateProductMultiMlecheckProver::new(
            selectors
                .into_iter()
                .map(|selector| [selector, ones.clone()])
                .collect_vec(),
            &eval_point,
            eval_claims.clone(),
        )
        .unwrap();

        let mut rerand_prover =
            RerandMlecheckProver::<P, _>::new(&eval_point, &eval_claims, &bitmasks, switchover)
                .unwrap();

        for _round in 0..n_vars {
            let rerand_round_coeffs = rerand_prover.execute().unwrap();
            let multi_bivariate_round_coeffs = multi_bivariate_prover.execute().unwrap();

            for (rerand, mut multi) in izip!(rerand_round_coeffs, multi_bivariate_round_coeffs) {
                // Bivariate product prover sizes round polynomials for degree-3 but they are
                // actually degree-2 and the highest coefficient is zero.
                assert_eq!(multi.0.pop(), Some(F::ZERO));
                assert_eq!(rerand, multi);
            }

            let challenge = F::random(&mut rng);
            rerand_prover.fold(challenge).unwrap();
            multi_bivariate_prover.fold(challenge).unwrap();
        }

        let rerand_multilinear_evals = rerand_prover.finish().unwrap();
        let multi_bivariate_multilinear_evals = multi_bivariate_prover.finish().unwrap();

        for (rerand, multi) in izip!(
            rerand_multilinear_evals,
            multi_bivariate_multilinear_evals.chunks(2)
        ) {
            assert_eq!(rerand, multi[0]);
            assert_eq!(F::ONE, multi[1]);
        }
    }

    #[test]
    fn test_bivariate_mlecheck_conformance() {
        for (n_vars, switchover) in [(8, 1), (8, 3), (8, 9)] {
            test_bivariate_mlecheck_conformance_helper(n_vars, switchover);
        }
    }
}
