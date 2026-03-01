use arith::{Field, Fr, SimdField};
use expander_utils::timer::Timer;
use gkr_engine::{
    BN254ConfigXN, ExpanderDualVarChallenge, FieldEngine, GKREngine, MPIConfig, MPIEngine,
    Transcript,
};

use crate::{
    frontend::{Config, SIMDField},
    utils::misc::next_power_of_two,
    zkcuda::{
        context::ComputationGraph,
        kernel::{Kernel, LayeredCircuitInputVec},
        proving_system::{
            expander::{
                commit_impl::local_commit_impl,
                config::{GetFieldConfig, GetPCS, GetTranscript, ZKCudaConfig},
                prove_impl::{
                    get_local_vals, prepare_expander_circuit, prepare_inputs_with_local_vals,
                },
                structs::{ExpanderProof, ExpanderProverSetup},
            },
            expander_no_oversubscribe::profiler::NBytesProfiler,
            expander_parallelized::{
                prove_impl::partition_single_gkr_claim_and_open_pcs_mpi,
                server_ctrl::generate_local_mpi_config,
            },
            expander_pcs_defered::prove_impl::{
                extract_pcs_claims, max_len_setup_commit_impl, open_defered_pcs,
            },
            CombinedProof, Expander,
        },
    },
};

pub fn mpi_prove_no_oversubscribe_impl<ZC: ZKCudaConfig>(
    global_mpi_config: &MPIConfig,
    prover_setup: &ExpanderProverSetup<GetFieldConfig<ZC>, GetPCS<ZC>>,
    computation_graph: &ComputationGraph<ZC::ECCConfig>,
    values: &[impl AsRef<[SIMDField<ZC::ECCConfig>]>],
    n_bytes_profiler: &mut NBytesProfiler,
) -> Option<CombinedProof<ZC::ECCConfig, Expander<ZC::GKRConfig>>>
where
    <ZC::GKRConfig as GKREngine>::FieldConfig: FieldEngine<CircuitField = Fr, ChallengeField = Fr>,
{
    let commit_timer = Timer::new("Commit to all input", global_mpi_config.is_root());
    let (commitments, states) = if global_mpi_config.is_root() {
        let (commitments, states) = values
            .iter()
            .map(|value| match ZC::BATCH_PCS {
                true => max_len_setup_commit_impl::<ZC::GKRConfig, ZC::ECCConfig>(
                    prover_setup,
                    value.as_ref(),
                ),
                false => local_commit_impl::<ZC::GKRConfig, ZC::ECCConfig>(
                    prover_setup.p_keys.get(&value.as_ref().len()).unwrap(),
                    value.as_ref(),
                ),
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();
        (Some(commitments), Some(states))
    } else {
        (None, None)
    };
    commit_timer.stop();

    let mut vals_ref = vec![];
    let mut challenges = vec![];

    let prove_timer = Timer::new("Prove all kernels", global_mpi_config.is_root());
    let proofs =
        computation_graph
            .proof_templates()
            .iter()
            .map(|template| {
                let commitment_values = template
                    .commitment_indices()
                    .iter()
                    .map(|&idx| values[idx].as_ref())
                    .collect::<Vec<_>>();

                let single_kernel_gkr_timer =
                    Timer::new("small gkr kernel", global_mpi_config.is_root());
                let gkr_end_state = prove_kernel_gkr_no_oversubscribe::<
                    GetFieldConfig<ZC>,
                    GetTranscript<ZC>,
                    ZC::ECCConfig,
                >(
                    global_mpi_config,
                    &computation_graph.kernels()[template.kernel_id()],
                    &commitment_values,
                    next_power_of_two(template.parallel_count()),
                    template.is_broadcast(),
                    n_bytes_profiler,
                );
                single_kernel_gkr_timer.stop();

                match ZC::BATCH_PCS {
                    true => {
                        if global_mpi_config.is_root() {
                            let (mut transcript, challenge) = gkr_end_state.unwrap();
                            assert!(challenge.challenge_y().is_none());
                            let challenge = challenge.challenge_x();

                            let (local_vals_ref, local_challenges) =
                                extract_pcs_claims::<ZC::GKRConfig>(
                                    &commitment_values,
                                    &challenge,
                                    template.is_broadcast(),
                                    next_power_of_two(template.parallel_count()),
                                );

                            vals_ref.extend(local_vals_ref);
                            challenges.extend(local_challenges);

                            Some(ExpanderProof {
                                data: vec![transcript.finalize_and_get_proof()],
                            })
                        } else {
                            None
                        }
                    }
                    false => {
                        if global_mpi_config.is_root() {
                            let pcs_open_timer = Timer::new("pcs open", true);
                            let (mut transcript, challenge) = gkr_end_state.unwrap();
                            let challenges = if let Some(challenge_y) = challenge.challenge_y() {
                                vec![challenge.challenge_x(), challenge_y]
                            } else {
                                vec![challenge.challenge_x()]
                            };

                            challenges.iter().for_each(|c| {
                                partition_single_gkr_claim_and_open_pcs_mpi::<ZC::GKRConfig>(
                                    prover_setup,
                                    &commitment_values,
                                    &template
                                        .commitment_indices()
                                        .iter()
                                        .map(|&idx| &states.as_ref().unwrap()[idx])
                                        .collect::<Vec<_>>(),
                                    c,
                                    template.is_broadcast(),
                                    &mut transcript,
                                );
                            });

                            pcs_open_timer.stop();
                            Some(ExpanderProof {
                                data: vec![transcript.finalize_and_get_proof()],
                            })
                        } else {
                            None
                        }
                    }
                }
            })
            .collect::<Vec<_>>();
    prove_timer.stop();

    match ZC::BATCH_PCS {
        true => {
            if global_mpi_config.is_root() {
                let mut proofs = proofs.into_iter().map(|p| p.unwrap()).collect::<Vec<_>>();

                let pcs_opening_timer = Timer::new("Batch PCS Opening for all kernels", true);
                let pcs_batch_opening = open_defered_pcs::<ZC::GKRConfig, ZC::ECCConfig>(
                    prover_setup,
                    &vals_ref,
                    &challenges,
                );
                pcs_opening_timer.stop();

                proofs.push(pcs_batch_opening);
                Some(CombinedProof {
                    commitments: commitments.unwrap(),
                    proofs,
                })
            } else {
                None
            }
        }
        false => {
            if global_mpi_config.is_root() {
                let proofs = proofs.into_iter().map(|p| p.unwrap()).collect::<Vec<_>>();
                Some(CombinedProof {
                    commitments: commitments.unwrap(),
                    proofs,
                })
            } else {
                None
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prove_kernel_gkr_no_oversubscribe<F, T, ECCConfig>(
    mpi_config: &MPIConfig,
    kernel: &Kernel<ECCConfig>,
    commitments_values: &[&[F::SimdCircuitField]],
    parallel_count: usize,
    is_broadcast: &[bool],
    n_bytes_profiler: &mut NBytesProfiler,
) -> Option<(T, ExpanderDualVarChallenge<F>)>
where
    F: FieldEngine<CircuitField = Fr, ChallengeField = Fr>,
    T: Transcript,
    ECCConfig: Config<FieldConfig = F>,
{
    let local_mpi_config = generate_local_mpi_config(mpi_config, parallel_count);

    local_mpi_config.as_ref()?;

    let local_mpi_config = local_mpi_config.unwrap();
    let local_world_size = local_mpi_config.world_size();

    let n_local_copies = parallel_count / local_world_size;
    match n_local_copies {
        1 => prove_kernel_gkr_internal::<F, F, T, ECCConfig>(
            &local_mpi_config,
            kernel,
            commitments_values,
            parallel_count,
            is_broadcast,
            n_bytes_profiler,
        ),
        2 => prove_kernel_gkr_internal::<F, BN254ConfigXN<2>, T, ECCConfig>(
            &local_mpi_config,
            kernel,
            commitments_values,
            parallel_count,
            is_broadcast,
            n_bytes_profiler,
        ),
        4 => prove_kernel_gkr_internal::<F, BN254ConfigXN<4>, T, ECCConfig>(
            &local_mpi_config,
            kernel,
            commitments_values,
            parallel_count,
            is_broadcast,
            n_bytes_profiler,
        ),
        8 => prove_kernel_gkr_internal::<F, BN254ConfigXN<8>, T, ECCConfig>(
            &local_mpi_config,
            kernel,
            commitments_values,
            parallel_count,
            is_broadcast,
            n_bytes_profiler,
        ),
        16 => prove_kernel_gkr_internal::<F, BN254ConfigXN<16>, T, ECCConfig>(
            &local_mpi_config,
            kernel,
            commitments_values,
            parallel_count,
            is_broadcast,
            n_bytes_profiler,
        ),
        32 => prove_kernel_gkr_internal::<F, BN254ConfigXN<32>, T, ECCConfig>(
            &local_mpi_config,
            kernel,
            commitments_values,
            parallel_count,
            is_broadcast,
            n_bytes_profiler,
        ),
        64 => prove_kernel_gkr_internal::<F, BN254ConfigXN<64>, T, ECCConfig>(
            &local_mpi_config,
            kernel,
            commitments_values,
            parallel_count,
            is_broadcast,
            n_bytes_profiler,
        ),
        128 => prove_kernel_gkr_internal::<F, BN254ConfigXN<128>, T, ECCConfig>(
            &local_mpi_config,
            kernel,
            commitments_values,
            parallel_count,
            is_broadcast,
            n_bytes_profiler,
        ),
        256 => prove_kernel_gkr_internal::<F, BN254ConfigXN<256>, T, ECCConfig>(
            &local_mpi_config,
            kernel,
            commitments_values,
            parallel_count,
            is_broadcast,
            n_bytes_profiler,
        ),
        512 => prove_kernel_gkr_internal::<F, BN254ConfigXN<512>, T, ECCConfig>(
            &local_mpi_config,
            kernel,
            commitments_values,
            parallel_count,
            is_broadcast,
            n_bytes_profiler,
        ),
        1024 => prove_kernel_gkr_internal::<F, BN254ConfigXN<1024>, T, ECCConfig>(
            &local_mpi_config,
            kernel,
            commitments_values,
            parallel_count,
            is_broadcast,
            n_bytes_profiler,
        ),
        2048 => prove_kernel_gkr_internal::<F, BN254ConfigXN<2048>, T, ECCConfig>(
            &local_mpi_config,
            kernel,
            commitments_values,
            parallel_count,
            is_broadcast,
            n_bytes_profiler,
        ),
        _ => {
            panic!("Unsupported parallel count: {parallel_count}");
        }
    }
}

pub fn prove_kernel_gkr_internal<FBasic, FMulti, T, ECCConfig>(
    mpi_config: &MPIConfig,
    kernel: &Kernel<ECCConfig>,
    commitments_values: &[&[FBasic::SimdCircuitField]],
    parallel_count: usize,
    is_broadcast: &[bool],
    n_bytes_profiler: &mut NBytesProfiler,
) -> Option<(T, ExpanderDualVarChallenge<FBasic>)>
where
    FBasic: FieldEngine<CircuitField = Fr, ChallengeField = Fr>,
    FMulti:
        FieldEngine<CircuitField = FBasic::CircuitField, ChallengeField = FBasic::ChallengeField>,
    T: Transcript,
    ECCConfig: Config<FieldConfig = FBasic>,
{
    let world_rank = mpi_config.world_rank();
    let world_size = mpi_config.world_size();
    let n_copies = parallel_count / world_size;

    let local_commitment_values = get_local_vals_multi_copies(
        commitments_values,
        is_broadcast,
        world_rank,
        n_copies,
        parallel_count,
    );

    let (mut expander_circuit, mut prover_scratch) =
        prepare_expander_circuit::<FMulti, ECCConfig>(kernel, world_size);

    let mut transcript = T::new();
    let challenge = prove_gkr_with_local_vals_multi_copies::<FBasic, FMulti, T>(
        &mut expander_circuit,
        &mut prover_scratch,
        &local_commitment_values,
        kernel.layered_circuit_input(),
        &mut transcript,
        mpi_config,
        n_bytes_profiler,
    );

    Some((transcript, challenge))
}

pub fn get_local_vals_multi_copies<'vals_life, F: Field>(
    global_vals: &'vals_life [impl AsRef<[F]>],
    is_broadcast: &[bool],
    local_world_rank: usize,
    n_copies: usize,
    parallel_count: usize,
) -> Vec<Vec<&'vals_life [F]>> {
    let parallel_indices = (0..n_copies)
        .map(|i| local_world_rank * n_copies + i)
        .collect::<Vec<_>>();

    parallel_indices
        .iter()
        .map(|&parallel_index| {
            get_local_vals(global_vals, is_broadcast, parallel_index, parallel_count)
        })
        .collect::<Vec<_>>()
}

pub fn prove_gkr_with_local_vals_multi_copies<FBasic, FMulti, T>(
    expander_circuit: &mut expander_circuit::Circuit<FMulti>,
    prover_scratch: &mut sumcheck::ProverScratchPad<FMulti>,
    local_commitment_values_multi_copies: &[Vec<impl AsRef<[FBasic::SimdCircuitField]>>],
    partition_info: &[LayeredCircuitInputVec],
    transcript: &mut T,
    mpi_config: &MPIConfig,
    _n_bytes_profiler: &mut NBytesProfiler,
) -> ExpanderDualVarChallenge<FBasic>
where
    FBasic: FieldEngine<CircuitField = Fr, ChallengeField = Fr>,
    FMulti:
        FieldEngine<CircuitField = FBasic::CircuitField, ChallengeField = FBasic::ChallengeField>,
    T: Transcript,
{
    let input_vals_multi_copies = local_commitment_values_multi_copies
        .iter()
        .map(|local_commitment_values| {
            prepare_inputs_with_local_vals(
                1 << expander_circuit.log_input_size(),
                partition_info,
                local_commitment_values,
            )
        })
        .collect::<Vec<_>>();

    let mut input_vals =
        vec![FMulti::SimdCircuitField::ZERO; 1 << expander_circuit.log_input_size()];

    for (i, vals) in input_vals.iter_mut().enumerate() {
        let vals_unpacked = input_vals_multi_copies
            .iter()
            .flat_map(|v| v[i].unpack())
            .collect::<Vec<_>>();
        *vals = FMulti::SimdCircuitField::pack(&vals_unpacked);
    }
    expander_circuit.layers[0].input_vals = input_vals;

    expander_circuit.fill_rnd_coefs(transcript);
    expander_circuit.evaluate();

    #[cfg(feature = "zkcuda_profile")]
    {
        expander_circuit.layers.iter().for_each(|layer| {
            layer.input_vals.iter().for_each(|val| {
                val.unpack().iter().for_each(|fr| {
                    _n_bytes_profiler.add_fr(*fr);
                })
            });
        });
    }

    let (claimed_v, challenge) =
        gkr::gkr_prove(expander_circuit, prover_scratch, transcript, mpi_config);
    assert_eq!(claimed_v, FBasic::ChallengeField::from(0u32));

    let n_simd_vars_basic = FBasic::SimdCircuitField::PACK_SIZE.ilog2() as usize;

    ExpanderDualVarChallenge {
        rz_0: challenge.rz_0,
        rz_1: challenge.rz_1,
        r_simd: challenge.r_simd[..n_simd_vars_basic].to_vec(),
        r_mpi: {
            let mut v = challenge.r_simd[n_simd_vars_basic..].to_vec();
            v.extend(&challenge.r_mpi);
            v
        },
    }
}
