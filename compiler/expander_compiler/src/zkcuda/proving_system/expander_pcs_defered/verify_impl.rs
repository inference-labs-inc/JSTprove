use std::collections::HashMap;
use std::io::Cursor;

use arith::Field;
use expander_utils::timer::Timer;
use gkr::gkr_verify;
use gkr_engine::{
    ExpanderDualVarChallenge, ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, GKREngine,
    Proof as BytesProof, Transcript,
};
use polynomials::EqPolynomial;
use serdes::ExpSerde;
use sumcheck::{IOPProof, SumCheck};

use crate::{
    frontend::Config,
    utils::misc::next_power_of_two,
    zkcuda::{
        context::ComputationGraph,
        kernel::Kernel,
        proving_system::{
            expander::structs::{ExpanderCommitment, ExpanderProof, ExpanderVerifierSetup},
            expander_parallelized::prove_impl::partition_challenge_and_location_for_pcs_mpi,
            CombinedProof, Commitment, Expander,
        },
    },
};

fn verifier_extract_pcs_claims<'a, C, ECCConfig>(
    commitments: &[&'a ExpanderCommitment<C::FieldConfig, C::PCSConfig>],
    gkr_challenge: &ExpanderSingleVarChallenge<C::FieldConfig>,
    is_broadcast: &[bool],
    parallel_count: usize,
    commitment_indices: &[usize],
) -> (
    Vec<&'a ExpanderCommitment<C::FieldConfig, C::PCSConfig>>,
    Vec<ExpanderSingleVarChallenge<C::FieldConfig>>,
    Vec<usize>,
)
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    let mut commitments_rt = vec![];
    let mut challenges = vec![];
    let mut indices = vec![];

    for (i, (&commitment, ib)) in commitments.iter().zip(is_broadcast).enumerate() {
        let val_len =
            <ExpanderCommitment<C::FieldConfig, C::PCSConfig> as Commitment<ECCConfig>>::vals_len(
                commitment,
            );
        let (challenge_for_pcs, _) = partition_challenge_and_location_for_pcs_mpi(
            gkr_challenge,
            val_len,
            parallel_count,
            *ib,
        );

        commitments_rt.push(commitment);
        challenges.push(challenge_for_pcs);
        indices.push(commitment_indices[i]);
    }

    (commitments_rt, challenges, indices)
}

pub fn verify_gkr<C, ECCConfig>(
    kernel: &Kernel<ECCConfig>,
    proof: &ExpanderProof,
    parallel_count: usize,
) -> (bool, ExpanderDualVarChallenge<C::FieldConfig>)
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    let mut expander_circuit = kernel.layered_circuit().export_to_expander_flatten();

    let mut transcript = C::TranscriptConfig::new();
    expander_circuit.fill_rnd_coefs(&mut transcript);

    let mut cursor = Cursor::new(&proof.data[0].bytes);
    let (verified, challenge, _claimed_v0, _claimed_v1) = gkr_verify(
        parallel_count,
        &expander_circuit,
        &[],
        &<C::FieldConfig as FieldEngine>::ChallengeField::ZERO,
        &mut transcript,
        &mut cursor,
        None,
    );

    if !verified {
        println!("Failed to verify GKR proof");
        return (false, challenge);
    }

    (true, challenge)
}

pub fn verify_same_poly_reduction_pcs<C, ECCConfig>(
    proof: &BytesProof,
    verifier_setup: &ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>,
    commitments: &[&ExpanderCommitment<C::FieldConfig, C::PCSConfig>],
    challenges: &[ExpanderSingleVarChallenge<C::FieldConfig>],
    commitment_indices: &[usize],
) -> bool
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,

    <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::Commitment:
        AsRef<<C::PCSConfig as ExpanderPCS<C::FieldConfig>>::Commitment>,
{
    type CF<C> = <<C as GKREngine>::FieldConfig as FieldEngine>::ChallengeField;

    let mut transcript = C::TranscriptConfig::new();
    let max_num_vars = verifier_setup.v_keys.keys().max().cloned().unwrap_or(0);
    let params = <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::gen_params(max_num_vars, 1);

    let mut proof_bytes = proof.bytes.clone();
    let mut cursor = Cursor::new(&mut proof_bytes);

    let num_groups = u64::deserialize_from(&mut cursor).unwrap() as usize;

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (claim_idx, &commit_idx) in commitment_indices.iter().enumerate() {
        groups.entry(commit_idx).or_default().push(claim_idx);
    }

    let mut sorted_group_keys: Vec<usize> = groups.keys().copied().collect();
    sorted_group_keys.sort();

    assert_eq!(sorted_group_keys.len(), num_groups);

    transcript.lock_proof();

    for &commit_idx in &sorted_group_keys {
        let claim_indices = &groups[&commit_idx];
        let group_challenges: Vec<_> = claim_indices
            .iter()
            .map(|&i| challenges[i].clone())
            .collect();
        let commitment = &commitments[claim_indices[0]].commitment;

        let is_reduced = u8::deserialize_from(&mut cursor).unwrap();
        let num_claims = u64::deserialize_from(&mut cursor).unwrap() as usize;
        assert_eq!(num_claims, claim_indices.len());

        if is_reduced == 1 {
            let evals = Vec::<CF<C>>::deserialize_from(&mut cursor).unwrap();
            let sumcheck_proof = IOPProof::<CF<C>>::deserialize_from(&mut cursor).unwrap();
            let claimed_sum = CF::<C>::deserialize_from(&mut cursor).unwrap();

            let m = group_challenges.len();
            let ell = (m as f64).log2().ceil() as usize;
            let num_vars = group_challenges[0].rz.len() + group_challenges[0].r_simd.len();

            let alpha: Vec<CF<C>> = transcript.generate_field_elements(ell);

            let mut eq_alpha_j = vec![CF::<C>::ZERO; m];
            for j in 0..m {
                let mut j_bits = vec![CF::<C>::ZERO; ell];
                for (bit_idx, bit_val) in j_bits.iter_mut().enumerate() {
                    if (j >> bit_idx) & 1 == 1 {
                        *bit_val = CF::<C>::ONE;
                    }
                }
                eq_alpha_j[j] = EqPolynomial::eq_vec(&alpha, &j_bits);
            }

            let expected_sum: CF<C> = eq_alpha_j
                .iter()
                .zip(evals.iter())
                .map(|(a, v)| *a * *v)
                .sum();

            if expected_sum != claimed_sum {
                println!("Same-poly reduction: claimed sum mismatch");
                return false;
            }

            let (sc_verified, subclaim) =
                SumCheck::verify(claimed_sum, &sumcheck_proof, num_vars, &mut transcript);

            if !sc_verified {
                println!("Same-poly reduction: sumcheck verification failed");
                return false;
            }

            let mut r = subclaim.point.clone();
            r.reverse();

            let mut g_at_r = CF::<C>::ZERO;
            for (j, challenge) in group_challenges.iter().enumerate() {
                let xs = challenge.local_xs();
                g_at_r += eq_alpha_j[j] * EqPolynomial::eq_vec(&r, &xs);
            }

            let f_at_r = subclaim.expected_evaluation * g_at_r.inv().unwrap();

            let r_simd_len = group_challenges[0].r_simd.len();
            let reduced_challenge = ExpanderSingleVarChallenge::new(
                r[r_simd_len..].to_vec(),
                r[..r_simd_len].to_vec(),
                vec![],
            );

            let opening = <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::Opening::deserialize_from(
                &mut cursor,
            )
            .unwrap();

            let pcs_verified = <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::verify(
                &params,
                verifier_setup.v_keys.get(&max_num_vars).unwrap(),
                commitment,
                &reduced_challenge,
                f_at_r,
                &mut transcript,
                &opening,
            );

            if !pcs_verified {
                println!("Same-poly reduction: PCS verification failed for reduced group");
                return false;
            }
        } else {
            for &claim_idx in claim_indices {
                let eval = CF::<C>::deserialize_from(&mut cursor).unwrap();

                let opening =
                    <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::Opening::deserialize_from(
                        &mut cursor,
                    )
                    .unwrap();

                let pcs_verified = <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::verify(
                    &params,
                    verifier_setup.v_keys.get(&max_num_vars).unwrap(),
                    &commitments[claim_idx].commitment,
                    &challenges[claim_idx],
                    eval,
                    &mut transcript,
                    &opening,
                );

                if !pcs_verified {
                    println!("Same-poly reduction: PCS verification failed for individual claim");
                    return false;
                }
            }
        }
    }

    transcript.unlock_proof();
    true
}

pub fn verify<C, ECCConfig>(
    verifier_setup: &ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>,
    computation_graph: &ComputationGraph<ECCConfig>,
    mut proof: CombinedProof<ECCConfig, Expander<C>>,
) -> bool
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,

    <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::Commitment:
        AsRef<<C::PCSConfig as ExpanderPCS<C::FieldConfig>>::Commitment>,
{
    let verification_timer = Timer::new("Total Verification", true);
    let pcs_batch_opening = proof.proofs.pop().unwrap();

    let gkr_verification_timer = Timer::new("GKR Verification", true);
    let verified_with_pcs_claims = proof
        .proofs
        .iter()
        .zip(computation_graph.proof_templates().iter())
        .map(|(local_proof, template)| {
            let local_commitments = template
                .commitment_indices()
                .iter()
                .map(|idx| &proof.commitments[*idx])
                .collect::<Vec<_>>();

            let (verified, challenge) = verify_gkr::<C, ECCConfig>(
                &computation_graph.kernels()[template.kernel_id()],
                local_proof,
                next_power_of_two(template.parallel_count()),
            );

            assert!(challenge.challenge_y().is_none());
            let challenge = challenge.challenge_x();

            let (local_commitments, challenges, local_indices) =
                verifier_extract_pcs_claims::<C, ECCConfig>(
                    &local_commitments,
                    &challenge,
                    template.is_broadcast(),
                    next_power_of_two(template.parallel_count()),
                    template.commitment_indices(),
                );

            (verified, local_commitments, challenges, local_indices)
        })
        .collect::<Vec<_>>();

    let gkr_verified = verified_with_pcs_claims.iter().all(|(v, _, _, _)| *v);
    if !gkr_verified {
        println!("Failed to verify GKR proofs");
        return false;
    }
    gkr_verification_timer.stop();

    let pcs_verification_timer = Timer::new("PCS Verification", true);
    let commitments_ref = verified_with_pcs_claims
        .iter()
        .flat_map(|(_, c, _, _)| c)
        .copied()
        .collect::<Vec<_>>();

    let challenges = verified_with_pcs_claims
        .iter()
        .flat_map(|(_, _, c, _)| c.clone())
        .collect::<Vec<_>>();

    let all_commitment_indices: Vec<usize> = verified_with_pcs_claims
        .iter()
        .flat_map(|(_, _, _, idx)| idx.clone())
        .collect();

    let pcs_verified = verify_same_poly_reduction_pcs::<C, ECCConfig>(
        &pcs_batch_opening.data[0],
        verifier_setup,
        &commitments_ref,
        &challenges,
        &all_commitment_indices,
    );
    pcs_verification_timer.stop();

    verification_timer.stop();
    gkr_verified && pcs_verified
}
