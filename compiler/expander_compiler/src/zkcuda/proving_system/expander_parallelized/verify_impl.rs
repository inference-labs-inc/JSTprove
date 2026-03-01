use std::io::{Cursor, Read};

use arith::Field;
use expander_utils::timer::Timer;
use gkr::gkr_verify;
use gkr_engine::{
    ExpanderDualVarChallenge, ExpanderSingleVarChallenge, FieldEngine, GKREngine, Transcript,
};
use polynomials::EqPolynomial;
use serdes::ExpSerde;

use crate::{
    frontend::Config,
    zkcuda::{
        kernel::Kernel,
        proving_system::{
            expander::{
                structs::{ExpanderCommitment, ExpanderProof, ExpanderVerifierSetup},
                verify_impl::verify_pcs,
            },
            expander_parallelized::prove_impl::partition_challenge_and_location_for_pcs_mpi,
            Commitment,
        },
    },
};

pub fn verify_kernel<C, ECCConfig>(
    verifier_setup: &ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>,
    kernel: &Kernel<ECCConfig>,
    proof: &ExpanderProof,
    commitments: &[&ExpanderCommitment<C::FieldConfig, C::PCSConfig>],
    parallel_count: usize,
    is_broadcast: &[bool],
) -> bool
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    let timer = Timer::new("verify", true);
    let mut expander_circuit = kernel.layered_circuit().export_to_expander_flatten();

    let mut transcript = C::TranscriptConfig::new();
    expander_circuit.fill_rnd_coefs(&mut transcript);

    let mut cursor = Cursor::new(&proof.data[0].bytes);
    let (mut verified, challenge, claimed_v0, claimed_v1) = gkr_verify(
        parallel_count,
        &expander_circuit,
        &[],
        &<C::FieldConfig as FieldEngine>::ChallengeField::ZERO,
        &mut transcript,
        &mut cursor,
    );

    if !verified {
        println!("Failed to verify GKR proof");
        return false;
    }

    verified = verify_pcs_opening_and_aggregation_mpi::<C, ECCConfig>(
        &mut cursor,
        kernel,
        verifier_setup,
        &challenge,
        claimed_v0,
        claimed_v1,
        commitments,
        is_broadcast,
        parallel_count,
        &mut transcript,
    );

    if !verified {
        println!("Failed to verify overall pcs");
        return false;
    }
    timer.stop();

    true
}

pub fn verify_pcs_opening_and_aggregation_mpi_impl<C, ECCConfig>(
    mut proof_reader: impl Read,
    kernel: &Kernel<ECCConfig>,
    v_keys: &ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>,
    challenge: &ExpanderSingleVarChallenge<C::FieldConfig>,
    y: &<C::FieldConfig as FieldEngine>::ChallengeField,
    commitments: &[&ExpanderCommitment<C::FieldConfig, C::PCSConfig>],
    is_broadcast: &[bool],
    parallel_count: usize,
    transcript: &mut C::TranscriptConfig,
) -> bool
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    let mut target_y = <C::FieldConfig as FieldEngine>::ChallengeField::ZERO;
    for ((input, commitment), ib) in kernel
        .layered_circuit_input()
        .iter()
        .zip(commitments.iter())
        .zip(is_broadcast)
    {
        let val_len =
            <ExpanderCommitment<C::FieldConfig, C::PCSConfig> as Commitment<ECCConfig>>::vals_len(
                commitment,
            );
        let (challenge_for_pcs, component_idx_vars) =
            partition_challenge_and_location_for_pcs_mpi(challenge, val_len, parallel_count, *ib);

        let claim =
            <C::FieldConfig as FieldEngine>::ChallengeField::deserialize_from(&mut proof_reader)
                .unwrap();
        transcript.append_field_element(&claim);

        let verified = verify_pcs::<C, ECCConfig>(
            &mut proof_reader,
            commitment,
            &challenge_for_pcs,
            &claim,
            v_keys,
            transcript,
        );

        if !verified {
            println!("Failed to verify individual pcs opening");
            return false;
        }

        let component_index = input.offset / input.len;
        let v_index = EqPolynomial::ith_eq_vec_elem(&component_idx_vars, component_index);

        target_y += v_index * claim;
    }

    // overall claim verification
    *y == target_y
}

pub fn verify_pcs_opening_and_aggregation_mpi<C, ECCConfig>(
    mut proof_reader: impl Read,
    kernel: &Kernel<ECCConfig>,
    v_keys: &ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>,
    challenge: &ExpanderDualVarChallenge<C::FieldConfig>,
    claim_v0: <C::FieldConfig as FieldEngine>::ChallengeField,
    claim_v1: Option<<C::FieldConfig as FieldEngine>::ChallengeField>,
    commitments: &[&ExpanderCommitment<C::FieldConfig, C::PCSConfig>],
    is_broadcast: &[bool],
    parallel_count: usize,
    transcript: &mut C::TranscriptConfig,
) -> bool
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    let challenges = if let Some(challenge_y) = challenge.challenge_y() {
        vec![challenge.challenge_x(), challenge_y]
    } else {
        vec![challenge.challenge_x()]
    };

    let claims = if let Some(claim_v1) = claim_v1 {
        vec![claim_v0, claim_v1]
    } else {
        vec![claim_v0]
    };

    assert_eq!(
        challenges.len(),
        claims.len(),
        "Number of challenges and claims must match"
    );

    challenges
        .iter()
        .zip(claims.iter())
        .all(|(challenge, claim)| {
            verify_pcs_opening_and_aggregation_mpi_impl::<C, ECCConfig>(
                &mut proof_reader,
                kernel,
                v_keys,
                challenge,
                claim,
                commitments,
                is_broadcast,
                parallel_count,
                transcript,
            )
        })
}
