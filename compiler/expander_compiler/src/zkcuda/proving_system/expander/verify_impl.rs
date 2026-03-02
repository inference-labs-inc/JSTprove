use std::io::Read;

use arith::Field;
use gkr_engine::{
    ExpanderDualVarChallenge, ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, GKREngine,
    Transcript,
};
use polynomials::EqPolynomial;
use serdes::ExpSerde;

use crate::{
    frontend::Config,
    zkcuda::{
        kernel::Kernel,
        proving_system::{
            expander::{
                prove_impl::partition_challenge_and_location_for_pcs_no_mpi,
                structs::{ExpanderCommitment, ExpanderVerifierSetup},
            },
            Commitment,
        },
    },
};

pub fn verify_pcs<C, ECCConfig>(
    mut proof_reader: impl Read,
    commitment: &ExpanderCommitment<C::FieldConfig, C::PCSConfig>,
    challenge: &ExpanderSingleVarChallenge<C::FieldConfig>,
    claim: &<C::FieldConfig as FieldEngine>::ChallengeField,
    v_keys: &ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>,
    transcript: &mut C::TranscriptConfig,
) -> bool
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    let val_len =
        <ExpanderCommitment<C::FieldConfig, C::PCSConfig> as Commitment<ECCConfig>>::vals_len(
            commitment,
        );

    let params =
        <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::gen_params(val_len.ilog2() as usize, 1);
    let v_key = v_keys.v_keys.get(&val_len).unwrap();

    let opening =
        <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::Opening::deserialize_from(&mut proof_reader)
            .unwrap();

    transcript.lock_proof();
    let verified = <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::verify(
        &params,
        v_key,
        &commitment.commitment,
        challenge,
        *claim,
        transcript,
        &opening,
    );
    transcript.unlock_proof();

    let mut buffer = vec![];
    opening
        .serialize_into(&mut buffer)
        .expect("Failed to serialize opening");
    transcript.append_u8_slice(&buffer);

    verified
}

#[allow(clippy::too_many_arguments)]
pub fn verify_pcs_opening_and_aggregation_no_mpi_impl<C, ECCConfig>(
    mut proof_reader: impl Read,
    kernel: &Kernel<ECCConfig>,
    v_keys: &ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>,
    challenge: &ExpanderSingleVarChallenge<C::FieldConfig>,
    y: &<C::FieldConfig as FieldEngine>::ChallengeField,
    commitments: &[&ExpanderCommitment<C::FieldConfig, C::PCSConfig>],
    is_broadcast: &[bool],
    parallel_index: usize,
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
            partition_challenge_and_location_for_pcs_no_mpi(
                challenge,
                val_len,
                parallel_index,
                parallel_count,
                *ib,
            );

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
            println!(
                "Failed to verify pcs opening for input at offset {}",
                input.offset
            );
            return false;
        }

        let component_index = input.offset / input.len;
        let v_index = EqPolynomial::ith_eq_vec_elem(&component_idx_vars, component_index);

        target_y += v_index * claim;
    }

    *y == target_y
}

pub fn verify_pcs_opening_and_aggregation_no_mpi<C, ECCConfig>(
    mut proof_reader: impl Read,
    kernel: &Kernel<ECCConfig>,
    v_keys: &ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>,
    challenge: &ExpanderDualVarChallenge<C::FieldConfig>,
    claim_v0: <C::FieldConfig as FieldEngine>::ChallengeField,
    claim_v1: Option<<C::FieldConfig as FieldEngine>::ChallengeField>,
    commitments: &[&ExpanderCommitment<C::FieldConfig, C::PCSConfig>],
    is_broadcast: &[bool],
    parallel_index: usize,
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

    challenges
        .into_iter()
        .zip(claims)
        .all(|(challenge, claim)| {
            verify_pcs_opening_and_aggregation_no_mpi_impl::<C, ECCConfig>(
                &mut proof_reader,
                kernel,
                v_keys,
                &challenge,
                &claim,
                commitments,
                is_broadcast,
                parallel_index,
                parallel_count,
                transcript,
            )
        })
}
