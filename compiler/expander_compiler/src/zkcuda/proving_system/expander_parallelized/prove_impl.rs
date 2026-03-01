use arith::Field;
use expander_utils::timer::Timer;
use gkr_engine::{
    ExpanderDualVarChallenge, ExpanderSingleVarChallenge, FieldEngine, GKREngine, MPIConfig,
    MPIEngine, Transcript,
};

use crate::{
    frontend::{Config, SIMDField},
    utils::misc::next_power_of_two,
    zkcuda::{
        context::ComputationGraph,
        kernel::Kernel,
        proving_system::{
            expander::{
                commit_impl::local_commit_impl,
                prove_impl::{
                    get_local_vals, partition_gkr_claims_and_open_pcs_no_mpi, pcs_local_open_impl,
                    prepare_expander_circuit, prove_gkr_with_local_vals,
                },
                structs::{ExpanderCommitmentState, ExpanderProof, ExpanderProverSetup},
            },
            expander_parallelized::server_ctrl::generate_local_mpi_config,
            CombinedProof, Expander,
        },
    },
};

pub fn mpi_prove_impl<C, ECCConfig>(
    _global_mpi_config: &MPIConfig,
    prover_setup: &ExpanderProverSetup<C::FieldConfig, C::PCSConfig>,
    computation_graph: &ComputationGraph<ECCConfig>,
    values: &[impl AsRef<[SIMDField<C>]>],
) -> Option<CombinedProof<ECCConfig, Expander<C>>>
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    let commit_timer = Timer::new("Commit to all input", true);
    let (commitments, _states) = values
        .iter()
        .map(|value| {
            local_commit_impl::<C, ECCConfig>(
                prover_setup.p_keys.get(&value.as_ref().len()).unwrap(),
                value.as_ref(),
            )
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();
    commit_timer.stop();

    let prove_timer = Timer::new("Prove all kernels", true);
    let proofs = computation_graph
        .proof_templates()
        .iter()
        .map(|template| {
            let commitment_values: Vec<_> = template
                .commitment_indices()
                .iter()
                .map(|&idx| values[idx].as_ref())
                .collect();

            let parallel_count = next_power_of_two(template.parallel_count());
            let kernel = &computation_graph.kernels()[template.kernel_id()];

            let (mut expander_circuit, mut prover_scratch) =
                prepare_expander_circuit::<C::FieldConfig, ECCConfig>(kernel, 1);

            let mut proof_data = vec![];
            for parallel_index in 0..parallel_count {
                let local_vals = get_local_vals(
                    &commitment_values,
                    template.is_broadcast(),
                    parallel_index,
                    parallel_count,
                );

                let mut transcript = C::TranscriptConfig::new();
                let challenge = prove_gkr_with_local_vals::<C::FieldConfig, C::TranscriptConfig>(
                    &mut expander_circuit,
                    &mut prover_scratch,
                    &local_vals,
                    kernel.layered_circuit_input(),
                    &mut transcript,
                    &MPIConfig::prover_new(),
                );

                partition_gkr_claims_and_open_pcs_no_mpi::<C>(
                    &challenge,
                    &commitment_values,
                    prover_setup,
                    template.is_broadcast(),
                    parallel_index,
                    parallel_count,
                    &mut transcript,
                );

                proof_data.push(transcript.finalize_and_get_proof());
            }

            ExpanderProof { data: proof_data }
        })
        .collect::<Vec<_>>();
    prove_timer.stop();

    Some(CombinedProof {
        commitments,
        proofs,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn prove_kernel_gkr<F, T, ECCConfig>(
    mpi_config: &MPIConfig,
    kernel: &Kernel<ECCConfig>,
    commitments_values: &[&[F::SimdCircuitField]],
    parallel_count: usize,
    is_broadcast: &[bool],
) -> Option<(T, ExpanderDualVarChallenge<F>)>
where
    F: FieldEngine,
    T: Transcript,
    ECCConfig: Config<FieldConfig = F>,
{
    let local_mpi_config = generate_local_mpi_config(mpi_config, parallel_count);

    local_mpi_config.as_ref()?;

    let local_mpi_config = local_mpi_config.unwrap();
    let local_world_size = local_mpi_config.world_size();
    let local_world_rank = local_mpi_config.world_rank();

    let local_commitment_values = get_local_vals(
        commitments_values,
        is_broadcast,
        local_world_rank,
        local_world_size,
    );

    let (mut expander_circuit, mut prover_scratch) =
        prepare_expander_circuit::<F, ECCConfig>(kernel, local_world_size);

    let mut transcript = T::new();
    let challenge = prove_gkr_with_local_vals::<F, T>(
        &mut expander_circuit,
        &mut prover_scratch,
        &local_commitment_values,
        kernel.layered_circuit_input(),
        &mut transcript,
        &local_mpi_config,
    );

    Some((transcript, challenge))
}

pub fn partition_challenge_and_location_for_pcs_mpi<F: FieldEngine>(
    gkr_challenge: &ExpanderSingleVarChallenge<F>,
    total_vals_len: usize,
    parallel_count: usize,
    is_broadcast: bool,
) -> (ExpanderSingleVarChallenge<F>, Vec<F::ChallengeField>) {
    let mut challenge = gkr_challenge.clone();
    let zero = F::ChallengeField::ZERO;
    if is_broadcast {
        let n_vals_vars = total_vals_len.ilog2() as usize;
        let component_idx_vars = challenge.rz[n_vals_vars..].to_vec();
        challenge.rz.resize(n_vals_vars, zero);
        challenge.r_mpi.clear();
        (challenge, component_idx_vars)
    } else {
        let n_vals_vars = (total_vals_len / parallel_count).ilog2() as usize;
        let component_idx_vars = challenge.rz[n_vals_vars..].to_vec();
        challenge.rz.resize(n_vals_vars, zero);

        challenge.rz.extend_from_slice(&challenge.r_mpi);
        challenge.r_mpi.clear();
        (challenge, component_idx_vars)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn partition_single_gkr_claim_and_open_pcs_mpi<C: GKREngine>(
    p_keys: &ExpanderProverSetup<C::FieldConfig, C::PCSConfig>,
    commitments_values: &[impl AsRef<[SIMDField<C>]>],
    commitments_state: &[&ExpanderCommitmentState<C::FieldConfig, C::PCSConfig>],
    gkr_challenge: &ExpanderSingleVarChallenge<C::FieldConfig>,
    is_broadcast: &[bool],
    transcript: &mut C::TranscriptConfig,
) {
    let parallel_count = 1 << gkr_challenge.r_mpi.len();
    for ((commitment_val, _state), ib) in commitments_values
        .iter()
        .zip(commitments_state)
        .zip(is_broadcast)
    {
        let val_len = commitment_val.as_ref().len();
        let (challenge_for_pcs, _) = partition_challenge_and_location_for_pcs_mpi(
            gkr_challenge,
            val_len,
            parallel_count,
            *ib,
        );

        pcs_local_open_impl::<C>(
            commitment_val.as_ref(),
            &challenge_for_pcs,
            p_keys,
            transcript,
        );
    }
}
