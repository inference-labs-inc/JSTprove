// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_field::{Field, PackedField};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
    FieldBuffer, FieldSlice, multilinear, multilinear::eq::eq_ind_partial_eval,
    univariate::evaluate_univariate,
};
use binius_spartan_frontend::constraint_system::{MulConstraint, Operand, WitnessIndex};
use binius_utils::{checked_arithmetics::checked_log_2, rayon::prelude::*};

/// Transpose of the wiring sparse matrix.
#[derive(Debug)]
pub struct WiringTranspose {
    flat_keys: Vec<Key>,
    keys_start_by_witness_index: Vec<u32>,
    log_witness_size: usize,
}

#[derive(Debug, Clone)]
pub struct Key {
    pub operand_idx: u8,
    pub constraint_idx: u32,
}

impl WiringTranspose {
    pub fn transpose(witness_size: usize, mul_constraints: &[MulConstraint<WitnessIndex>]) -> Self {
        let mut operands_keys_by_wit_idx = vec![Vec::new(); witness_size];

        let mut n_total_keys = 0;
        for (i, MulConstraint { a, b, c }) in mul_constraints.iter().enumerate() {
            for (operand_idx, operand) in [a, b, c].into_iter().enumerate() {
                for &witness_idx in operand.wires() {
                    operands_keys_by_wit_idx[witness_idx.0 as usize].push(Key {
                        operand_idx: operand_idx as u8,
                        constraint_idx: i as u32,
                    });
                    n_total_keys += 1;
                }
            }
        }

        // Flatten the sparse matrix representation.
        let mut operand_keys = Vec::with_capacity(n_total_keys);
        let mut operand_key_start_by_word = Vec::with_capacity(witness_size);
        for keys in operands_keys_by_wit_idx {
            let start = operand_keys.len() as u32;
            operand_keys.extend(keys);
            operand_key_start_by_word.push(start);
        }

        let log_witness_size = checked_log_2(witness_size);

        Self {
            flat_keys: operand_keys,
            keys_start_by_witness_index: operand_key_start_by_word,
            log_witness_size,
        }
    }

    /// Returns the log2 of the witness size.
    pub fn log_witness_size(&self) -> usize {
        self.log_witness_size
    }

    /// Returns the witness size.
    pub fn witness_size(&self) -> usize {
        1 << self.log_witness_size
    }

    /// Returns an iterator over keys for a specific witness index.
    pub fn keys_for_witness(&self, witness_idx: usize) -> &[Key] {
        let start = self.keys_start_by_witness_index[witness_idx] as usize;
        let end = self
            .keys_start_by_witness_index
            .get(witness_idx + 1)
            .map(|&x| x as usize)
            .unwrap_or(self.flat_keys.len());
        &self.flat_keys[start..end]
    }
}

/// Computes the batched polynomial ℓ = w + χ eq(r_pub || 0), where w is the wiring polynomial
/// and χ is `batch_coeff`.
///
/// Since eq(r_pub || 0) takes the value 0 on all hypercube vertices except the first 2^r_pub.len(),
/// only that first chunk needs to be updated.
fn compute_l_poly<F: Field, P: PackedField<Scalar = F>>(
    wiring_poly: FieldBuffer<P>,
    r_public: &[F],
    batch_coeff: F,
) -> FieldBuffer<P> {
    let mut l_poly = wiring_poly;

    {
        let mut l_poly_public_chunk = l_poly.chunk_mut(r_public.len(), 0);
        let mut l_poly_public_chunk = l_poly_public_chunk.get();

        let eq_public = eq_ind_partial_eval::<P>(r_public);

        let batch_coeff_packed = P::broadcast(batch_coeff);
        for (dst, src) in iter::zip(l_poly_public_chunk.as_mut(), eq_public.as_ref()) {
            *dst += *src * batch_coeff_packed;
        }
    }

    l_poly
}

/// Folds the wiring matrix along the constraint axis by partially evaluating at r_x.
///
/// Also batches the three operands (a, b, c) using powers of lambda.
/// Returns a multilinear polynomial over witness indices where each coefficient is the
/// weighted sum of constraint contributions.
pub fn fold_constraints<F: Field, P: PackedField<Scalar = F>>(
    transposed: &WiringTranspose,
    lambda: F,
    r_x: &[F],
) -> FieldBuffer<P> {
    // Compute eq indicator tensor for constraint evaluation points
    let r_x_tensor = eq_ind_partial_eval::<F>(r_x);

    // Batching powers for the three operands
    let lambda_powers = [F::ONE, lambda, lambda.square()];

    // Create packed field buffer for witness indices
    let witness_size = transposed.witness_size();
    let log_witness_size = transposed.log_witness_size();
    let len = 1 << log_witness_size.saturating_sub(P::LOG_WIDTH);

    // Process in parallel over chunks of P::WIDTH witness indices
    let result = (0..len)
        .into_par_iter()
        .map(|packed_idx| {
            let base_witness_idx = packed_idx << P::LOG_WIDTH;

            P::from_fn(|scalar_idx| {
                let witness_idx = base_witness_idx + scalar_idx;
                if witness_idx >= witness_size {
                    return F::ZERO;
                }

                let mut acc = F::ZERO;
                for key in transposed.keys_for_witness(witness_idx) {
                    let r_x_weight = r_x_tensor[key.constraint_idx as usize];
                    let lambda_weight = lambda_powers[key.operand_idx as usize];
                    acc += r_x_weight * lambda_weight;
                }
                acc
            })
        })
        .collect::<Vec<_>>();

    FieldBuffer::new(log_witness_size, result.into_boxed_slice())
}

/// Result of computing the wiring relation for IOP proving.
///
/// Contains the folding polynomial (l_poly) and the claimed batched sum
/// that will be passed to the IOP channel's `prove_oracle_relations` method.
pub struct WiringRelation<P: PackedField> {
    /// The folding polynomial: wiring poly + batch_coeff * eq(r_public, ·)
    pub l_poly: FieldBuffer<P>,
    /// The claimed batched sum: λ-batched mulcheck evals + batch_coeff * public_eval
    pub batched_sum: P::Scalar,
}

/// Computes the wiring relation for IOP proving.
///
/// Samples batching challenges from the channel, computes the folding polynomial,
/// and returns the relation data needed for the IOP channel's `prove_oracle_relations` method.
pub fn compute_wiring_relation<F: Field, P: PackedField<Scalar = F>>(
    wiring_transpose: &WiringTranspose,
    witness: &FieldSlice<P>,
    r_public: &[F],
    r_x: &[F],
    mulcheck_evals: &[F],
    channel: &mut impl IPProverChannel<F>,
) -> WiringRelation<P> {
    // Sample batching challenges
    let lambda = channel.sample();
    let batch_coeff = channel.sample();

    // Fold constraints with batching and compute the folded polynomial
    let wiring_poly = fold_constraints(wiring_transpose, lambda, r_x);
    let l_poly = compute_l_poly(wiring_poly, r_public, batch_coeff);

    // Compute the public input evaluation
    let public = witness.chunk(r_public.len(), 0);
    let public_eval = multilinear::evaluate::evaluate(&public, r_public);

    // Compute the batched sum
    let batched_sum = evaluate_univariate(mulcheck_evals, lambda) + batch_coeff * public_eval;

    WiringRelation {
        l_poly,
        batched_sum,
    }
}

/// Witness data for multiplication constraint checking.
///
/// Contains the evaluated operands a, b, and c for all multiplication constraints,
/// packed into field buffers for efficient processing.
pub struct MulCheckWitness<P: PackedField> {
    pub a: FieldBuffer<P>,
    pub b: FieldBuffer<P>,
    pub c: FieldBuffer<P>,
}

/// Evaluates an operand by XORing witness values at the specified indices.
fn eval_operand<P: PackedField>(
    witness: &FieldSlice<P>,
    operand: &Operand<WitnessIndex>,
) -> P::Scalar
where
    P::Scalar: Field,
{
    operand
        .wires()
        .iter()
        .map(|idx| witness.get(idx.0 as usize))
        .sum()
}

/// Builds the witness for multiplication constraint checking.
///
/// Extracts and packs the a, b, and c operand values for each multiplication constraint.
/// This is analogous to `build_bitand_witness` in binius-prover but works with B128
/// field elements instead of word-level operations.
#[tracing::instrument(skip_all, level = "debug")]
pub fn build_mulcheck_witness<F: Field, P: PackedField<Scalar = F>>(
    mul_constraints: &[MulConstraint<WitnessIndex>],
    witness: FieldSlice<P>,
) -> MulCheckWitness<P> {
    fn get_a(c: &MulConstraint<WitnessIndex>) -> &Operand<WitnessIndex> {
        &c.a
    }
    fn get_b(c: &MulConstraint<WitnessIndex>) -> &Operand<WitnessIndex> {
        &c.b
    }
    fn get_c(c: &MulConstraint<WitnessIndex>) -> &Operand<WitnessIndex> {
        &c.c
    }

    let n_constraints = mul_constraints.len();
    assert!(n_constraints > 0, "mul_constraints must not be empty");

    let log_n_constraints = checked_log_2(n_constraints);

    let len = 1 << log_n_constraints.saturating_sub(P::LOG_WIDTH);
    let mut a = Vec::<P>::with_capacity(len);
    let mut b = Vec::<P>::with_capacity(len);
    let mut c = Vec::<P>::with_capacity(len);

    (
        a.spare_capacity_mut(),
        b.spare_capacity_mut(),
        c.spare_capacity_mut(),
    )
        .into_par_iter()
        .enumerate()
        .for_each(|(i, (a_i, b_i, c_i))| {
            let offset = i << P::LOG_WIDTH;

            for (dst, get_operand) in [
                (
                    a_i,
                    get_a as fn(&MulConstraint<WitnessIndex>) -> &Operand<WitnessIndex>,
                ),
                (
                    b_i,
                    get_b as fn(&MulConstraint<WitnessIndex>) -> &Operand<WitnessIndex>,
                ),
                (
                    c_i,
                    get_c as fn(&MulConstraint<WitnessIndex>) -> &Operand<WitnessIndex>,
                ),
            ] {
                let val = P::from_fn(|j| {
                    let constraint_idx = offset + j;
                    if constraint_idx < n_constraints {
                        eval_operand(&witness, get_operand(&mul_constraints[constraint_idx]))
                    } else {
                        F::ZERO
                    }
                });
                dst.write(val);
            }
        });

    // Safety: all entries in a, b, c are initialized in the parallel loop above.
    unsafe {
        a.set_len(len);
        b.set_len(len);
        c.set_len(len);
    }

    MulCheckWitness {
        a: FieldBuffer::new(log_n_constraints, a.into_boxed_slice()),
        b: FieldBuffer::new(log_n_constraints, b.into_boxed_slice()),
        c: FieldBuffer::new(log_n_constraints, c.into_boxed_slice()),
    }
}

#[cfg(test)]
mod tests {
    use binius_field::{BinaryField128bGhash as B128, Field, Random};
    use binius_math::{
        multilinear::{eq::eq_ind_partial_eval, evaluate::evaluate},
        test_utils::{Packed128b, random_scalars},
        univariate::evaluate_univariate,
    };
    use binius_spartan_frontend::constraint_system::{MulConstraint, Operand, WitnessIndex};
    use binius_spartan_verifier::wiring::evaluate_wiring_mle;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use smallvec::SmallVec;

    use super::*;

    /// Generate random MulConstraints for testing.
    /// Each operand has 0-4 random wires.
    fn generate_random_constraints(
        rng: &mut StdRng,
        n_constraints: usize,
        witness_size: usize,
    ) -> Vec<MulConstraint<WitnessIndex>> {
        (0..n_constraints)
            .map(|_| {
                let a = generate_random_operand(rng, witness_size);
                let b = generate_random_operand(rng, witness_size);
                let c = generate_random_operand(rng, witness_size);
                MulConstraint { a, b, c }
            })
            .collect()
    }

    fn generate_random_operand(rng: &mut StdRng, witness_size: usize) -> Operand<WitnessIndex> {
        let n_wires = rng.random_range(0..=4);
        let wires: SmallVec<[WitnessIndex; 4]> = (0..n_wires)
            .map(|_| WitnessIndex(rng.random_range(0..witness_size as u32)))
            .collect();
        Operand::new(wires)
    }

    /// Evaluate the wiring MLE using the transposed representation.
    fn evaluate_wiring_mle_transposed<F: Field>(
        transposed: &WiringTranspose,
        lambda: F,
        r_x_tensor: &[F],
        r_y_tensor: &[F],
    ) -> F {
        let mut acc = [F::ZERO; 3];

        for witness_idx in 0..transposed.witness_size() {
            let r_y_weight = r_y_tensor[witness_idx];
            for key in transposed.keys_for_witness(witness_idx) {
                let r_x_weight = r_x_tensor[key.constraint_idx as usize];
                acc[key.operand_idx as usize] += r_x_weight * r_y_weight;
            }
        }

        evaluate_univariate(&acc, lambda)
    }

    #[test]
    fn test_wiring_transpose_equivalence() {
        let mut rng = StdRng::seed_from_u64(0);

        // Generate random constraints
        let log_n_constraints = 4;
        let log_witness_size = 5;

        let n_constraints = 1 << log_n_constraints;
        let witness_size = 1 << log_witness_size;

        let constraints = generate_random_constraints(&mut rng, n_constraints, witness_size);

        // Sample random evaluation points
        let r_x = random_scalars::<B128>(&mut rng, log_n_constraints);
        let r_y = random_scalars::<B128>(&mut rng, log_witness_size);
        let lambda = B128::random(&mut rng);

        // Compute expected result using the original representation
        let expected = evaluate_wiring_mle(&constraints, lambda, &r_x, &r_y);

        // Compute result using the transposed representation
        let transposed = WiringTranspose::transpose(witness_size, &constraints);
        let r_x_tensor = eq_ind_partial_eval::<B128>(&r_x);
        let r_y_tensor = eq_ind_partial_eval::<B128>(&r_y);
        let actual = evaluate_wiring_mle_transposed(
            &transposed,
            lambda,
            r_x_tensor.as_ref(),
            r_y_tensor.as_ref(),
        );

        assert_eq!(
            actual, expected,
            "Transposed evaluation does not match original evaluation"
        );
    }

    #[test]
    fn test_fold_constraints_equivalence() {
        let mut rng = StdRng::seed_from_u64(0);

        let log_n_constraints = 4;
        let log_witness_size = 5;

        let n_constraints = 1 << log_n_constraints;
        let witness_size = 1 << log_witness_size;

        // Generate random constraints
        let constraints = generate_random_constraints(&mut rng, n_constraints, witness_size);

        // Sample random evaluation points
        let r_x = random_scalars::<B128>(&mut rng, log_n_constraints);
        let r_y = random_scalars::<B128>(&mut rng, log_witness_size);
        let lambda = B128::random(&mut rng);

        // Method 1: Compute expected result using evaluate_wiring_mle
        let expected = evaluate_wiring_mle(&constraints, lambda, &r_x, &r_y);

        // Method 2: Use fold_constraints then evaluate at r_y
        let transposed = WiringTranspose::transpose(witness_size, &constraints);
        let folded = fold_constraints::<_, Packed128b>(&transposed, lambda, &r_x);
        let actual = evaluate(&folded, &r_y);

        assert_eq!(
            actual, expected,
            "fold_constraints + evaluate does not match evaluate_wiring_mle"
        );
    }

    #[test]
    fn test_wiring_prove_verify() {
        use binius_hash::StdDigest;
        use binius_iop::{
            channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec},
            naive_channel::NaiveVerifierChannel,
        };
        use binius_iop_prover::{channel::IOPProverChannel, naive_channel::NaiveProverChannel};
        use binius_math::{inner_product::inner_product_buffers, test_utils::random_field_buffer};
        use binius_spartan_frontend::constraint_system::{
            BlindingInfo, ConstraintSystem, ConstraintSystemPadded,
        };
        use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};

        type StdChallenger = HasherChallenger<StdDigest>;

        let mut rng = StdRng::seed_from_u64(0);

        // Parameters
        let log_n_constraints = 4;
        let log_public = 3;
        let log_witness_size = 5;

        let n_constraints = 1 << log_n_constraints;
        let witness_size = 1 << log_witness_size;

        // Generate random constraints
        let constraints = generate_random_constraints(&mut rng, n_constraints, witness_size);

        // Create random witness
        let witness = random_field_buffer::<Packed128b>(&mut rng, log_witness_size);

        // Compute mulcheck witness
        let mulcheck_witness = build_mulcheck_witness(&constraints, witness.to_ref());

        // Sample r_x (sumcheck evaluation point for constraint axis)
        let r_x = random_scalars::<B128>(&mut rng, log_n_constraints);

        // Compute mulcheck evaluations at r_x
        let r_x_tensor = eq_ind_partial_eval::<Packed128b>(&r_x);
        let mulcheck_evals = [
            inner_product_buffers(&mulcheck_witness.a, &r_x_tensor),
            inner_product_buffers(&mulcheck_witness.b, &r_x_tensor),
            inner_product_buffers(&mulcheck_witness.c, &r_x_tensor),
        ];

        // Sample r_public
        let r_public = random_scalars::<B128>(&mut rng, log_public);

        // Compute public input evaluation
        let public = witness.chunk(log_public, 0);
        let public_eval = evaluate(&public, &r_public);

        // Create transposed wiring
        let wiring_transpose = WiringTranspose::transpose(witness_size, &constraints);

        // Create constraint system for verifier (compute_claim doesn't actually use it,
        // but we need a valid reference for the type signature)
        let constraint_system = ConstraintSystem::new(
            vec![],              // constants
            0,                   // n_inout
            0,                   // n_private
            log_public as u32,   // log_public
            constraints.clone(), // mul_constraints
            WitnessIndex(0),     // one_wire (dummy for test)
        );
        let constraint_system_padded = ConstraintSystemPadded::new(
            constraint_system,
            BlindingInfo {
                n_dummy_wires: 0,
                n_dummy_constraints: 0,
            },
        );

        let oracle_specs = vec![OracleSpec {
            log_msg_len: log_witness_size,
        }];

        // === PROVER SIDE ===
        let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
        let mut prover_channel = NaiveProverChannel::<B128, Packed128b, _>::new(
            &mut prover_transcript,
            oracle_specs.clone(),
        );

        // Send witness oracle
        let witness_oracle = prover_channel.send_oracle(witness.to_ref());

        // Compute wiring relation (this samples lambda and batch_coeff from the channel)
        let wiring_relation = compute_wiring_relation(
            &wiring_transpose,
            &witness.to_ref(),
            &r_public,
            &r_x,
            &mulcheck_evals,
            &mut prover_channel,
        );

        // Finish the IOP with the oracle relation
        prover_channel.prove_oracle_relations([(
            witness_oracle,
            wiring_relation.l_poly.clone(),
            wiring_relation.batched_sum,
        )]);

        // === VERIFIER SIDE ===
        let mut verifier_transcript = prover_transcript.into_verifier();
        let mut verifier_channel =
            NaiveVerifierChannel::<B128, _>::new(&mut verifier_transcript, &oracle_specs);

        // Receive witness oracle
        let witness_oracle = verifier_channel
            .recv_oracle()
            .expect("recv_oracle should succeed");

        // Compute wiring claim (samples the same lambda and batch_coeff as prover)
        let wiring_claim = binius_spartan_verifier::wiring::compute_claim(
            &constraint_system_padded,
            &r_public,
            &mulcheck_evals,
            public_eval,
            &mut verifier_channel,
        );

        // Verify that prover and verifier computed the same batched_sum
        assert_eq!(
            wiring_relation.batched_sum, wiring_claim.batched_sum,
            "Prover and verifier batched_sum mismatch"
        );

        // Build the transparent closure using the prover's l_poly for evaluation.
        let l_poly = wiring_relation.l_poly;
        let transparent = Box::new(move |point: &[_]| evaluate(&l_poly, point));

        // Finish verification. The naive channel verifies the inner product directly inside
        // verify_oracle_relations(), reads the transparent polynomial from the transcript,
        // and checks that the transparent closure evaluation matches.
        verifier_channel
            .verify_oracle_relations([OracleLinearRelation {
                oracle: witness_oracle,
                transparent,
                claim: wiring_claim.batched_sum,
            }])
            .expect("verify_oracle_relations should succeed (inner product verified)");
    }
}
