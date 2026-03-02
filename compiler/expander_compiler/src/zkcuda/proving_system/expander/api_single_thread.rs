use std::io::Cursor;

use crate::circuit::config::Config;
use crate::frontend::SIMDField;
use crate::utils::misc::next_power_of_two;
use crate::zkcuda::context::ComputationGraph;
use crate::zkcuda::kernel::Kernel;
use crate::zkcuda::proving_system::expander::commit_impl::local_commit_impl;
use crate::zkcuda::proving_system::expander::prove_impl::{
    get_local_vals, partition_gkr_claims_and_open_pcs_no_mpi, prepare_expander_circuit,
    prove_gkr_with_local_vals,
};
use crate::zkcuda::proving_system::expander::setup_impl::local_setup_impl;
use crate::zkcuda::proving_system::expander::verify_impl::verify_pcs_opening_and_aggregation_no_mpi;
use crate::zkcuda::proving_system::{
    common::check_inputs, CombinedProof, KernelWiseProvingSystem, ProvingSystem,
};

use super::structs::{
    ExpanderCommitment, ExpanderCommitmentState, ExpanderProof, ExpanderProverSetup,
    ExpanderVerifierSetup,
};

use arith::Field;
use expander_utils::timer::Timer;
use gkr::gkr_verify;
use gkr_engine::{FieldEngine, GKREngine, MPIConfig, Transcript};

pub struct Expander<C: GKREngine> {
    _config: std::marker::PhantomData<C>,
}

impl<C, ECCConfig> KernelWiseProvingSystem<ECCConfig> for Expander<C>
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    type ProverSetup = ExpanderProverSetup<C::FieldConfig, C::PCSConfig>;
    type VerifierSetup = ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>;
    type Proof = ExpanderProof;
    type Commitment = ExpanderCommitment<C::FieldConfig, C::PCSConfig>;
    type CommitmentState = ExpanderCommitmentState<C::FieldConfig, C::PCSConfig>;

    fn setup(
        computation_graph: &crate::zkcuda::context::ComputationGraph<ECCConfig>,
    ) -> (Self::ProverSetup, Self::VerifierSetup) {
        local_setup_impl::<C, ECCConfig>(computation_graph)
    }

    fn commit(
        prover_setup: &Self::ProverSetup,
        vals: &[SIMDField<C>],
    ) -> (Self::Commitment, Self::CommitmentState) {
        local_commit_impl::<C, ECCConfig>(prover_setup.p_keys.get(&vals.len()).unwrap(), vals)
    }

    fn prove_kernel(
        prover_setup: &Self::ProverSetup,
        kernel: &Kernel<ECCConfig>,
        _commitments: &[&Self::Commitment],
        _commitments_state: &[&Self::CommitmentState],
        commitments_values: &[&[SIMDField<C>]],
        parallel_count: usize,
        is_broadcast: &[bool],
    ) -> Self::Proof {
        let timer = Timer::new("prove", true);
        check_inputs(kernel, commitments_values, parallel_count, is_broadcast);

        let (mut expander_circuit, mut prover_scratch) =
            prepare_expander_circuit::<C::FieldConfig, ECCConfig>(kernel, 1);

        let mut proof = ExpanderProof { data: vec![] };

        // For each parallel index, generate the GKR and PCS opening proof
        for parallel_index in 0..parallel_count {
            // TODO: Init with commitments
            let mut transcript = C::TranscriptConfig::new();
            let local_vals = get_local_vals(
                commitments_values,
                is_broadcast,
                parallel_index,
                parallel_count,
            );
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
                commitments_values,
                prover_setup,
                is_broadcast,
                parallel_index,
                parallel_count,
                &mut transcript,
            );

            proof.data.push(transcript.finalize_and_get_proof());
        }

        timer.stop();
        proof
    }

    fn verify_kernel(
        verifier_setup: &Self::VerifierSetup,
        kernel: &Kernel<ECCConfig>,
        proof: &Self::Proof,
        commitments: &[&Self::Commitment],
        parallel_count: usize,
        is_broadcast: &[bool],
    ) -> bool {
        let timer = Timer::new("verify", true);
        let mut expander_circuit = kernel.layered_circuit().export_to_expander_flatten();

        for i in 0..parallel_count {
            let mut transcript = C::TranscriptConfig::new();
            expander_circuit.fill_rnd_coefs(&mut transcript);

            let mut cursor = Cursor::new(&proof.data[i].bytes);
            let (mut verified, challenge, claimed_v0, claimed_v1) = gkr_verify(
                1,
                &expander_circuit,
                &[],
                &<C::FieldConfig as FieldEngine>::ChallengeField::ZERO,
                &mut transcript,
                &mut cursor,
            );

            if !verified {
                println!("Failed to verify GKR proof for parallel index {i}");
                return false;
            }

            verified &= verify_pcs_opening_and_aggregation_no_mpi::<C, ECCConfig>(
                &mut cursor,
                kernel,
                verifier_setup,
                &challenge,
                claimed_v0,
                claimed_v1,
                commitments,
                is_broadcast,
                i,
                parallel_count,
                &mut transcript,
            );

            if !verified {
                println!("Failed to verify overall pcs for parallel index {i}");
                return false;
            }
        }
        timer.stop();
        true
    }
}

// TODO: Generate this with procedural macros
// The idea is to implement the ProvingSystem trait for KernelWiseProvingSystem
// However, we can not simply implement ProvingSystem<C> for all KernelWiseProvingSystem<C> because
// If later we want a customized implementation of ProvingSystem for some struct A
// The compiler will not allow use to do so, complaining that KernelWiseProvingSystem may be later implemented for A
// causing a potential conflict.
// In this case, generate the implementation with a procedural macro seems to be the best solution.
impl<C: GKREngine, ECCConfig: Config<FieldConfig = C::FieldConfig>> ProvingSystem<ECCConfig>
    for Expander<C>
{
    type ProverSetup = <Self as KernelWiseProvingSystem<ECCConfig>>::ProverSetup;
    type VerifierSetup = <Self as KernelWiseProvingSystem<ECCConfig>>::VerifierSetup;
    type Proof = CombinedProof<ECCConfig, Self>;

    fn setup(
        computation_graph: &ComputationGraph<ECCConfig>,
    ) -> (Self::ProverSetup, Self::VerifierSetup) {
        <Self as KernelWiseProvingSystem<ECCConfig>>::setup(computation_graph)
    }

    fn prove(
        prover_setup: &Self::ProverSetup,
        computation_graph: &ComputationGraph<ECCConfig>,
        device_memories: Vec<Vec<SIMDField<ECCConfig>>>,
    ) -> Self::Proof {
        let (commitments, states) = device_memories
            .iter()
            .map(|device_memory| {
                <Self as KernelWiseProvingSystem<ECCConfig>>::commit(
                    prover_setup,
                    &device_memory[..],
                )
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let proofs: Vec<<Self as KernelWiseProvingSystem<ECCConfig>>::Proof> = computation_graph
            .proof_templates()
            .iter()
            .map(|template| {
                let (mut local_commitments, mut local_state, mut local_vals) =
                    (vec![], vec![], vec![]);
                for idx in template.commitment_indices() {
                    local_commitments.push(&commitments[*idx]);
                    local_state.push(&states[*idx]);
                    local_vals.push(&device_memories[*idx][..]);
                }

                <Self as KernelWiseProvingSystem<ECCConfig>>::prove_kernel(
                    prover_setup,
                    &computation_graph.kernels()[template.kernel_id()],
                    &local_commitments,
                    &local_state,
                    &local_vals,
                    next_power_of_two(template.parallel_count()),
                    template.is_broadcast(),
                )
            })
            .collect::<Vec<_>>();

        CombinedProof {
            commitments,
            proofs,
        }
    }

    fn verify(
        verifier_setup: &Self::VerifierSetup,
        computation_graph: &ComputationGraph<ECCConfig>,
        proof: &Self::Proof,
    ) -> bool {
        let verified = proof
            .proofs
            .iter()
            .zip(computation_graph.proof_templates().iter())
            .map(|(local_proof, template)| {
                let local_commitments = template
                    .commitment_indices()
                    .iter()
                    .map(|idx| &proof.commitments[*idx])
                    .collect::<Vec<_>>();

                <Self as KernelWiseProvingSystem<ECCConfig>>::verify_kernel(
                    verifier_setup,
                    &computation_graph.kernels()[template.kernel_id()],
                    local_proof,
                    &local_commitments,
                    next_power_of_two(template.parallel_count()),
                    template.is_broadcast(),
                )
            })
            .collect::<Vec<_>>();

        verified.iter().all(|x| *x)
    }

    fn post_process() {
        <Self as KernelWiseProvingSystem<ECCConfig>>::post_process();
    }
}
