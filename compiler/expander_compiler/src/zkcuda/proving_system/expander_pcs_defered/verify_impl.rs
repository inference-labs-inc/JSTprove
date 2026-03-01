use std::io::Cursor;

use arith::Field;
use expander_utils::timer::Timer;
use gkr::gkr_verify;
use gkr_engine::{
    ExpanderDualVarChallenge, ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, GKREngine,
    Proof as BytesProof, Transcript,
};
use serdes::ExpSerde;

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
) -> (
    Vec<&'a ExpanderCommitment<C::FieldConfig, C::PCSConfig>>,
    Vec<ExpanderSingleVarChallenge<C::FieldConfig>>,
)
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    let mut commitments_rt = vec![];
    let mut challenges = vec![];

    for (&commitment, ib) in commitments.iter().zip(is_broadcast) {
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
    }

    (commitments_rt, challenges)
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
    );

    if !verified {
        println!("Failed to verify GKR proof");
        return (false, challenge);
    }

    (true, challenge)
}

pub fn verify_defered_pcs_opening<C, ECCConfig>(
    proof: &BytesProof,
    verifier_setup: &ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>,
    commitments: &[&ExpanderCommitment<C::FieldConfig, C::PCSConfig>],
    challenges: &[ExpanderSingleVarChallenge<C::FieldConfig>],
) -> bool
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,

    <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::Commitment:
        AsRef<<C::PCSConfig as ExpanderPCS<C::FieldConfig>>::Commitment>,
{
    let mut transcript = C::TranscriptConfig::new();
    let max_num_vars = verifier_setup.v_keys.keys().max().cloned().unwrap_or(0);
    let params = <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::gen_params(max_num_vars, 1);

    let mut defered_proof_bytes = proof.bytes.clone();
    let mut cursor = Cursor::new(&mut defered_proof_bytes);

    let commitments = commitments
        .iter()
        .map(|commitment| &commitment.commitment)
        .collect::<Vec<_>>();
    let vals =
        Vec::<<C::FieldConfig as FieldEngine>::ChallengeField>::deserialize_from(&mut cursor)
            .unwrap();
    let opening =
        <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::BatchOpening::deserialize_from(&mut cursor)
            .unwrap();

    transcript.lock_proof();
    let pcs_verified = <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::multi_points_batch_verify(
        &params,
        verifier_setup.v_keys.get(&max_num_vars).unwrap(),
        &commitments,
        challenges,
        &vals,
        &opening,
        &mut transcript,
    );
    transcript.unlock_proof();

    pcs_verified
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

            let (local_commitments, challenges) = verifier_extract_pcs_claims::<C, ECCConfig>(
                &local_commitments,
                &challenge,
                template.is_broadcast(),
                next_power_of_two(template.parallel_count()),
            );

            (verified, local_commitments, challenges)
        })
        .collect::<Vec<_>>();

    let gkr_verified = verified_with_pcs_claims.iter().all(|(v, _, _)| *v);
    if !gkr_verified {
        println!("Failed to verify GKR proofs");
        return false;
    }
    gkr_verification_timer.stop();

    let pcs_verification_timer = Timer::new("PCS Verification", true);
    let commitments_ref = verified_with_pcs_claims
        .iter()
        .flat_map(|(_, c, _)| c)
        .copied()
        .collect::<Vec<_>>();

    let challenges = verified_with_pcs_claims
        .iter()
        .flat_map(|(_, _, c)| c.clone())
        .collect::<Vec<_>>();

    let pcs_verified = verify_defered_pcs_opening::<C, ECCConfig>(
        &pcs_batch_opening.data[0],
        verifier_setup,
        &commitments_ref,
        &challenges,
    );
    pcs_verification_timer.stop();

    verification_timer.stop();
    gkr_verified && pcs_verified
}
