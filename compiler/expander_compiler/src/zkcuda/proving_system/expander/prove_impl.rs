use arith::Field;
use expander_circuit::Circuit;
use gkr::gkr_prove;
use gkr_engine::{
    ExpanderDualVarChallenge, ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, GKREngine,
    MPIConfig, Transcript,
};
use polynomials::RefMultiLinearPoly;
use serdes::ExpSerde;
use sumcheck::ProverScratchPad;

use crate::{
    frontend::Config,
    zkcuda::{
        kernel::{Kernel, LayeredCircuitInputVec},
        proving_system::expander::structs::ExpanderProverSetup,
    },
};

/// ECCCircuit -> ExpanderCircuit
/// Returns an additional prover scratch pad for later use in GKR.
pub fn prepare_expander_circuit<F, ECCConfig>(
    kernel: &Kernel<ECCConfig>,
    mpi_world_size: usize,
) -> (Circuit<F>, ProverScratchPad<F>)
where
    F: FieldEngine,
    ECCConfig: Config,
    ECCConfig::FieldConfig: FieldEngine<CircuitField = F::CircuitField>,
{
    let mut expander_circuit = kernel.layered_circuit().export_to_expander().flatten();
    expander_circuit.pre_process_gkr();

    let (max_num_input_var, max_num_output_var) = super::utils::max_n_vars(&expander_circuit);
    let prover_scratch =
        ProverScratchPad::<F>::new(max_num_input_var, max_num_output_var, mpi_world_size);

    (expander_circuit, prover_scratch)
}

/// Global values consist of several components, each of which can be either broadcasted or partitioned.
/// If it is broadcasted, the same value is used across all parallel instances.
///   i.e. global_vals[i] is the same for all parallel instances.
/// If it is partitioned, each parallel instance gets a slice of the values.
///   i.e. global_vals[i] is partitioned equally into parallel_num slices, and each
///     parallel instance gets one slice.
///
/// This function returns the local values for each parallel instance based on the global values and the broadcast information.
pub fn get_local_vals<'vals_life, F: Field>(
    global_vals: &'vals_life [impl AsRef<[F]>],
    is_broadcast: &[bool],
    parallel_index: usize,
    parallel_num: usize,
) -> Vec<&'vals_life [F]> {
    global_vals
        .iter()
        .zip(is_broadcast.iter())
        .map(|(vals, is_broadcast)| {
            if *is_broadcast {
                vals.as_ref()
            } else {
                let local_val_len = vals.as_ref().len() / parallel_num;
                &vals.as_ref()[local_val_len * parallel_index..local_val_len * (parallel_index + 1)]
            }
        })
        .collect::<Vec<_>>()
}

/// Local values consist of several components, whose location and length are specified by `partition_info`.
/// This function targets at relocating the local values into a single vector based on the partition information.
pub fn prepare_inputs_with_local_vals<F: Field>(
    input_len: usize,
    partition_info: &[LayeredCircuitInputVec],
    local_commitment_values: &[impl AsRef<[F]>],
) -> Vec<F> {
    let mut input_vals = vec![F::ZERO; input_len];
    for (partition, val) in partition_info.iter().zip(local_commitment_values.iter()) {
        assert!(partition.len == val.as_ref().len());
        input_vals[partition.offset..partition.offset + partition.len]
            .copy_from_slice(val.as_ref());
    }
    input_vals
}

pub fn prove_gkr_with_local_vals<F: FieldEngine, T: Transcript>(
    expander_circuit: &mut Circuit<F>,
    prover_scratch: &mut ProverScratchPad<F>,
    local_commitment_values: &[impl AsRef<[F::SimdCircuitField]>],
    partition_info: &[LayeredCircuitInputVec],
    transcript: &mut T,
    mpi_config: &MPIConfig,
) -> ExpanderDualVarChallenge<F> {
    expander_circuit.layers[0].input_vals = prepare_inputs_with_local_vals(
        1 << expander_circuit.log_input_size(),
        partition_info,
        local_commitment_values,
    );
    expander_circuit.fill_rnd_coefs(transcript);
    expander_circuit.evaluate();
    let (claimed_v, challenge) =
        gkr_prove(expander_circuit, prover_scratch, transcript, mpi_config);
    assert_eq!(claimed_v, F::ChallengeField::from(0));
    challenge
}

/// Challenge structure:
/// llll pppp cccc ssss
/// Where:
///     l is the challenge for the local values
///     p is the challenge for the parallel index
///     c is the selector for the components
///     s is the challenge for the SIMD values
/// All little endian.
///
/// At the moment of commiting, we commited to the values corresponding to
///     llll pppp ssss
/// At the end of GKR, we will have the challenge
///     llll cccc ssss
/// The pppp part is not included because we're proving kernel-by-kernel.
///
/// Arguments:
/// - `challenge`: The gkr challenge: llll cccc ssss
/// - `total_vals_len`: The length of llll pppp
/// - `parallel_index`: The index of the parallel execution. pppp part.
/// - `parallel_count`: The total number of parallel executions. pppp part.
/// - `is_broadcast`: Whether the challenge is broadcasted or not.
///
/// Returns:
///     llll pppp ssss challenge
///     cccc
pub fn partition_challenge_and_location_for_pcs_no_mpi<F: FieldEngine>(
    gkr_challenge: &ExpanderSingleVarChallenge<F>,
    total_vals_len: usize,
    parallel_index: usize,
    parallel_count: usize,
    is_broadcast: bool,
) -> (ExpanderSingleVarChallenge<F>, Vec<F::ChallengeField>) {
    assert_eq!(gkr_challenge.r_mpi.len(), 0);
    let mut challenge = gkr_challenge.clone();
    let zero = F::ChallengeField::ZERO;
    if is_broadcast {
        let n_vals_vars = total_vals_len.ilog2() as usize;
        let component_idx_vars = challenge.rz[n_vals_vars..].to_vec();
        challenge.rz.resize(n_vals_vars, zero);
        (challenge, component_idx_vars)
    } else {
        let n_vals_vars = (total_vals_len / parallel_count).ilog2() as usize;
        let component_idx_vars = challenge.rz[n_vals_vars..].to_vec();
        challenge.rz.resize(n_vals_vars, zero);

        let n_index_vars = parallel_count.ilog2() as usize;
        let index_vars = (0..n_index_vars)
            .map(|i| F::ChallengeField::from(((parallel_index >> i) & 1) as u32))
            .collect::<Vec<_>>();

        challenge.rz.extend_from_slice(&index_vars);
        (challenge, component_idx_vars)
    }
}

pub fn pcs_local_open_impl<C: GKREngine>(
    vals: &[<C::FieldConfig as FieldEngine>::SimdCircuitField],
    challenge: &ExpanderSingleVarChallenge<C::FieldConfig>,
    p_keys: &ExpanderProverSetup<C::FieldConfig, C::PCSConfig>,
    transcript: &mut C::TranscriptConfig,
) {
    assert_eq!(challenge.r_mpi.len(), 0);

    let val_len = vals.len();
    let params =
        <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::gen_params(val_len.ilog2() as usize, 1);
    let p_key = p_keys.p_keys.get(&val_len).unwrap();

    let poly = RefMultiLinearPoly::from_ref(vals);
    // TODO: Change this function in Expander to use rayon.
    let v = <C::FieldConfig as FieldEngine>::single_core_eval_circuit_vals_at_expander_challenge(
        vals, challenge,
    );
    transcript.append_field_element(&v);

    let mpi_config = MPIConfig::prover_new();

    transcript.lock_proof();
    let opening = <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::open(
        &params,
        &mpi_config,
        p_key,
        &poly,
        challenge,
        transcript,
        &<C::PCSConfig as ExpanderPCS<C::FieldConfig>>::init_scratch_pad(&params, &mpi_config),
    )
    .unwrap();
    transcript.unlock_proof();

    let mut buffer = vec![];
    opening
        .serialize_into(&mut buffer)
        .expect("Failed to serialize opening");
    transcript.append_u8_slice(&buffer);
}

#[inline(always)]
pub fn partition_gkr_claims_and_open_pcs_no_mpi_impl<C: GKREngine>(
    gkr_claim: &ExpanderSingleVarChallenge<C::FieldConfig>,
    global_vals: &[impl AsRef<[<C::FieldConfig as FieldEngine>::SimdCircuitField]>],
    p_keys: &ExpanderProverSetup<C::FieldConfig, C::PCSConfig>,
    is_broadcast: &[bool],
    parallel_index: usize,
    parallel_num: usize,
    transcript: &mut C::TranscriptConfig,
) {
    for (commitment_val, ib) in global_vals.iter().zip(is_broadcast) {
        let val_len = commitment_val.as_ref().len();
        let (challenge_for_pcs, _) = partition_challenge_and_location_for_pcs_no_mpi::<
            C::FieldConfig,
        >(
            gkr_claim, val_len, parallel_index, parallel_num, *ib
        );

        pcs_local_open_impl::<C>(
            commitment_val.as_ref(),
            &challenge_for_pcs,
            p_keys,
            transcript,
        );
    }
}

/// By saying opening local PCS, we mean that the r_mpi challenge is not used
/// Instead, the parallel_index is interpreted for the vertical index of the local PCS,
/// and appended to the local PCS challenge.
pub fn partition_gkr_claims_and_open_pcs_no_mpi<C: GKREngine>(
    gkr_claim: &ExpanderDualVarChallenge<C::FieldConfig>,
    global_vals: &[impl AsRef<[<C::FieldConfig as FieldEngine>::SimdCircuitField]>],
    p_keys: &ExpanderProverSetup<C::FieldConfig, C::PCSConfig>,
    is_broadcast: &[bool],
    parallel_index: usize,
    parallel_num: usize,
    transcript: &mut C::TranscriptConfig,
) {
    let challenges = if let Some(challenge_y) = gkr_claim.challenge_y() {
        vec![gkr_claim.challenge_x(), challenge_y]
    } else {
        vec![gkr_claim.challenge_x()]
    };

    challenges.into_iter().for_each(|challenge| {
        partition_gkr_claims_and_open_pcs_no_mpi_impl::<C>(
            &challenge,
            global_vals,
            p_keys,
            is_broadcast,
            parallel_index,
            parallel_num,
            transcript,
        );
    });
}
