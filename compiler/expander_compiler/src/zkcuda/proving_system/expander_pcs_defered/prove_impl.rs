use std::collections::HashMap;

use arith::{Field, SimdField};
use expander_utils::timer::Timer;
use gkr_engine::{
    ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, GKREngine, MPIConfig, MPIEngine,
    Proof as BytesProof, Transcript,
};
use polynomials::{EqPolynomial, MultiLinearPoly, RefMultiLinearPoly, SumOfProductsPoly};
use serdes::ExpSerde;
use sumcheck::{IOPProof, SumCheck};

use crate::{
    frontend::{Config, SIMDField},
    utils::misc::next_power_of_two,
    zkcuda::{
        context::ComputationGraph,
        proving_system::{
            expander::{
                commit_impl::local_commit_impl,
                structs::{
                    ExpanderCommitment, ExpanderCommitmentState, ExpanderProof, ExpanderProverSetup,
                },
            },
            expander_parallelized::prove_impl::{
                partition_challenge_and_location_for_pcs_mpi, prove_kernel_gkr,
            },
            CombinedProof, Expander,
        },
    },
};

const SAME_POLY_REDUCTION_MIN_VARS: usize = 8;

pub fn max_len_setup_commit_impl<C, ECCConfig>(
    prover_setup: &ExpanderProverSetup<C::FieldConfig, C::PCSConfig>,
    vals: &[SIMDField<C>],
) -> (
    ExpanderCommitment<C::FieldConfig, C::PCSConfig>,
    ExpanderCommitmentState<C::FieldConfig, C::PCSConfig>,
)
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    assert_eq!(prover_setup.p_keys.len(), 1);
    let len_to_commit = prover_setup.p_keys.keys().next().cloned().unwrap();

    let actual_len = vals.len();
    assert!(len_to_commit >= actual_len);

    let (mut commitment, state) =
        local_commit_impl::<C, ECCConfig>(prover_setup.p_keys.get(&len_to_commit).unwrap(), vals);

    commitment.vals_len = actual_len;
    (commitment, state)
}

fn unpack_poly_to_challenge_field<F: FieldEngine>(
    simd_evals: &[F::SimdCircuitField],
) -> Vec<F::ChallengeField> {
    let pack_size = <F::SimdCircuitField as SimdField>::PACK_SIZE;
    let mut result = Vec::with_capacity(simd_evals.len() * pack_size);
    for eval in simd_evals {
        for scalar in eval.unpack() {
            result.push(F::ChallengeField::from(scalar));
        }
    }
    result
}

fn compute_same_poly_reduction_sumcheck<F: FieldEngine>(
    f_challenge_evals: &[F::ChallengeField],
    challenges: &[ExpanderSingleVarChallenge<F>],
    transcript: &mut impl Transcript,
) -> (
    IOPProof<F::ChallengeField>,
    F::ChallengeField,
    Vec<F::ChallengeField>,
) {
    let m = challenges.len();
    let ell = (m as f64).log2().ceil() as usize;
    let num_vars = challenges[0].rz.len() + challenges[0].r_simd.len();

    let alpha: Vec<F::ChallengeField> = transcript.generate_field_elements(ell);

    let evals: Vec<F::ChallengeField> = challenges
        .iter()
        .map(|c| {
            let xs = c.local_xs();
            let eq_evals = EqPolynomial::build_eq_x_r(&xs);
            eq_evals
                .iter()
                .zip(f_challenge_evals.iter())
                .map(|(e, f)| *e * *f)
                .sum()
        })
        .collect();

    let mut eq_alpha_j = vec![F::ChallengeField::ZERO; m];
    for j in 0..m {
        let mut j_bits = vec![F::ChallengeField::ZERO; ell];
        for (bit_idx, bit_val) in j_bits.iter_mut().enumerate() {
            if (j >> bit_idx) & 1 == 1 {
                *bit_val = F::ChallengeField::ONE;
            }
        }
        eq_alpha_j[j] = EqPolynomial::eq_vec(&alpha, &j_bits);
    }

    let claimed_sum: F::ChallengeField = eq_alpha_j
        .iter()
        .zip(evals.iter())
        .map(|(a, v)| *a * *v)
        .sum();

    let n = 1usize << num_vars;
    let mut g_evals = vec![F::ChallengeField::ZERO; n];
    for (j, challenge) in challenges.iter().enumerate() {
        let xs = challenge.local_xs();
        let eq_x_zj = EqPolynomial::build_eq_x_r(&xs);
        for (i, g_eval) in g_evals.iter_mut().enumerate().take(n) {
            *g_eval += eq_alpha_j[j] * eq_x_zj[i];
        }
    }

    let f_poly = MultiLinearPoly {
        coeffs: f_challenge_evals.to_vec(),
    };
    let g_poly = MultiLinearPoly { coeffs: g_evals };

    let mut sumcheck_poly = SumOfProductsPoly {
        f_and_g_pairs: vec![],
    };
    sumcheck_poly.add_pair(f_poly, g_poly);

    let proof = SumCheck::prove(&sumcheck_poly, transcript);

    (proof, claimed_sum, evals)
}

pub fn open_with_same_poly_reduction<C, ECCConfig>(
    prover_setup: &ExpanderProverSetup<C::FieldConfig, C::PCSConfig>,
    vals: &[&[SIMDField<C>]],
    challenges: &[ExpanderSingleVarChallenge<C::FieldConfig>],
    commitment_indices: &[usize],
) -> ExpanderProof
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    let mut transcript = C::TranscriptConfig::new();
    let max_length = prover_setup.p_keys.keys().max().cloned().unwrap_or(0);
    let params =
        <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::gen_params(max_length.ilog2() as usize, 1);
    let scratch_pad = <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::init_scratch_pad(
        &params,
        &MPIConfig::prover_new(),
    );

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (claim_idx, &commit_idx) in commitment_indices.iter().enumerate() {
        groups.entry(commit_idx).or_default().push(claim_idx);
    }

    let mut sorted_group_keys: Vec<usize> = groups.keys().copied().collect();
    sorted_group_keys.sort();

    let mut bytes = vec![];
    let num_groups = sorted_group_keys.len() as u64;
    num_groups.serialize_into(&mut bytes).unwrap();

    transcript.lock_proof();

    for &commit_idx in &sorted_group_keys {
        let claim_indices = &groups[&commit_idx];
        let group_challenges: Vec<_> = claim_indices
            .iter()
            .map(|&i| challenges[i].clone())
            .collect();
        let poly_data = vals[claim_indices[0]];

        let should_reduce = claim_indices.len() > 1
            && group_challenges[0].num_vars() >= SAME_POLY_REDUCTION_MIN_VARS;

        if should_reduce {
            1u8.serialize_into(&mut bytes).unwrap();
            let num_claims = claim_indices.len() as u64;
            num_claims.serialize_into(&mut bytes).unwrap();

            let f_challenge_evals = unpack_poly_to_challenge_field::<C::FieldConfig>(poly_data);

            let (sumcheck_proof, claimed_sum, evals) =
                compute_same_poly_reduction_sumcheck::<C::FieldConfig>(
                    &f_challenge_evals,
                    &group_challenges,
                    &mut transcript,
                );

            evals.serialize_into(&mut bytes).unwrap();
            sumcheck_proof.serialize_into(&mut bytes).unwrap();
            claimed_sum.serialize_into(&mut bytes).unwrap();

            let reduced_rz = sumcheck_proof.export_point_to_expander();
            let r_simd_len = group_challenges[0].r_simd.len();
            let reduced_challenge = ExpanderSingleVarChallenge::new(
                reduced_rz[r_simd_len..].to_vec(),
                reduced_rz[..r_simd_len].to_vec(),
                vec![],
            );

            let poly_ref = RefMultiLinearPoly::from_ref(poly_data);
            let opening = <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::open(
                &params,
                &MPIConfig::prover_new(),
                prover_setup.p_keys.get(&max_length).unwrap(),
                &poly_ref,
                &reduced_challenge,
                &mut transcript,
                &scratch_pad,
            );

            if let Some(ref o) = opening {
                o.serialize_into(&mut bytes).unwrap();
            }
        } else {
            0u8.serialize_into(&mut bytes).unwrap();
            let num_claims = claim_indices.len() as u64;
            num_claims.serialize_into(&mut bytes).unwrap();

            for &claim_idx in claim_indices {
                let poly_ref = RefMultiLinearPoly::from_ref(vals[claim_idx]);

                let eval =
                    <C::FieldConfig as FieldEngine>::single_core_eval_circuit_vals_at_expander_challenge(
                        vals[claim_idx],
                        &challenges[claim_idx],
                    );
                eval.serialize_into(&mut bytes).unwrap();

                let opening = <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::open(
                    &params,
                    &MPIConfig::prover_new(),
                    prover_setup.p_keys.get(&max_length).unwrap(),
                    &poly_ref,
                    &challenges[claim_idx],
                    &mut transcript,
                    &scratch_pad,
                );

                if let Some(ref o) = opening {
                    o.serialize_into(&mut bytes).unwrap();
                }
            }
        }
    }

    transcript.unlock_proof();

    ExpanderProof {
        data: vec![BytesProof { bytes }],
    }
}

pub fn mpi_prove_with_pcs_defered<C, ECCConfig>(
    global_mpi_config: &MPIConfig,
    prover_setup: &ExpanderProverSetup<C::FieldConfig, C::PCSConfig>,
    computation_graph: &ComputationGraph<ECCConfig>,
    values: &[impl AsRef<[SIMDField<C>]>],
) -> Option<CombinedProof<ECCConfig, Expander<C>>>
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    let commit_timer = Timer::new("Commit to all input", global_mpi_config.is_root());
    let (commitments, _states) = if global_mpi_config.is_root() {
        let (commitments, states) = values
            .iter()
            .map(|value| max_len_setup_commit_impl::<C, ECCConfig>(prover_setup, value.as_ref()))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        (Some(commitments), Some(states))
    } else {
        (None, None)
    };
    commit_timer.stop();

    let mut vals_ref = vec![];
    let mut challenges = vec![];
    let mut all_commitment_indices = vec![];

    let prove_timer = Timer::new(
        "Prove all kernels (NO PCS Opening)",
        global_mpi_config.is_root(),
    );
    let proofs = computation_graph
        .proof_templates()
        .iter()
        .map(|template| {
            let commitment_values = template
                .commitment_indices()
                .iter()
                .map(|&idx| values[idx].as_ref())
                .collect::<Vec<_>>();

            let gkr_end_state = prove_kernel_gkr::<C::FieldConfig, C::TranscriptConfig, ECCConfig>(
                global_mpi_config,
                &computation_graph.kernels()[template.kernel_id()],
                &commitment_values,
                next_power_of_two(template.parallel_count()),
                template.is_broadcast(),
            );

            if global_mpi_config.is_root() {
                let (mut transcript, challenge) = gkr_end_state.unwrap();
                assert!(challenge.challenge_y().is_none());
                let challenge = challenge.challenge_x();

                let (local_vals_ref, local_challenges, local_indices) = extract_pcs_claims::<C>(
                    &commitment_values,
                    &challenge,
                    template.is_broadcast(),
                    next_power_of_two(template.parallel_count()),
                    template.commitment_indices(),
                );

                vals_ref.extend(local_vals_ref);
                challenges.extend(local_challenges);
                all_commitment_indices.extend(local_indices);

                Some(ExpanderProof {
                    data: vec![transcript.finalize_and_get_proof()],
                })
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    prove_timer.stop();

    if global_mpi_config.is_root() {
        let mut proofs = proofs.into_iter().map(|p| p.unwrap()).collect::<Vec<_>>();

        let pcs_opening_timer = Timer::new("Same-poly reduction PCS Opening", true);
        let pcs_proof = open_with_same_poly_reduction::<C, ECCConfig>(
            prover_setup,
            &vals_ref,
            &challenges,
            &all_commitment_indices,
        );
        pcs_opening_timer.stop();

        proofs.push(pcs_proof);
        Some(CombinedProof {
            commitments: commitments.unwrap(),
            proofs,
        })
    } else {
        None
    }
}

pub fn extract_pcs_claims<'a, C: GKREngine>(
    commitments_values: &[&'a [SIMDField<C>]],
    gkr_challenge: &ExpanderSingleVarChallenge<C::FieldConfig>,
    is_broadcast: &[bool],
    parallel_count: usize,
    commitment_indices: &[usize],
) -> (
    Vec<&'a [SIMDField<C>]>,
    Vec<ExpanderSingleVarChallenge<C::FieldConfig>>,
    Vec<usize>,
) {
    let mut commitment_values_rt = vec![];
    let mut challenges = vec![];
    let mut indices = vec![];

    for (i, (&commitment_val, &ib)) in commitments_values.iter().zip(is_broadcast).enumerate() {
        let val_len = commitment_val.len();
        let (challenge_for_pcs, _) = partition_challenge_and_location_for_pcs_mpi(
            gkr_challenge,
            val_len,
            parallel_count,
            ib,
        );

        commitment_values_rt.push(commitment_val);
        challenges.push(challenge_for_pcs);
        indices.push(commitment_indices[i]);
    }

    (commitment_values_rt, challenges, indices)
}
