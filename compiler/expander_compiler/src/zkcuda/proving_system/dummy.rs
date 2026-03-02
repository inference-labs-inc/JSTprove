use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serdes::ExpSerde;

use crate::circuit::config::{Config, SIMDField};
use crate::utils::misc::next_power_of_two;
use crate::zkcuda::context::ComputationGraph;
use crate::zkcuda::proving_system::{CombinedProof, KernelWiseProvingSystem, ProvingSystem};

use super::super::kernel::Kernel;

use super::{check_inputs, prepare_inputs, Commitment};

// dummy implementation of these traits

#[derive(Clone, ExpSerde)]
pub struct DummyCommitment<C: Config> {
    vals: Vec<SIMDField<C>>,
}

impl<C: Config> Commitment<C> for DummyCommitment<C> {
    fn vals_len(&self) -> usize {
        self.vals.len()
    }
}

#[derive(Clone, ExpSerde)]
pub struct DummyProof {
    cond: Vec<Vec<bool>>,
}

// TODO
/*#[deprecated(
    note = "DummyProvingSystem is a dummy implementation for testing purposes. Please use Expander."
)]*/
pub struct DummyProvingSystem<C: Config> {
    _config: std::marker::PhantomData<C>,
}

#[allow(deprecated)]
impl<C: Config> KernelWiseProvingSystem<C> for DummyProvingSystem<C> {
    type ProverSetup = ();
    type VerifierSetup = ();
    type Proof = DummyProof;
    type Commitment = DummyCommitment<C>;
    type CommitmentState = ();

    fn setup(computation_graph: &ComputationGraph<C>) -> (Self::ProverSetup, Self::VerifierSetup) {
        // let _ = computation_graph;
        computation_graph
            .commitments_lens()
            .iter()
            .for_each(|&x| println!("Setup length {x}"));

        ((), ())
    }

    fn commit(
        _prover_setup: &Self::ProverSetup,
        vals: &[SIMDField<C>],
    ) -> (Self::Commitment, Self::CommitmentState) {
        assert!(vals.len() & (vals.len() - 1) == 0);
        (
            DummyCommitment {
                vals: vals.to_vec(),
            },
            (),
        )
    }

    fn prove_kernel(
        _prover_setup: &Self::ProverSetup,
        kernel: &Kernel<C>,
        _commitments: &[&Self::Commitment],
        _commitments_state: &[&Self::CommitmentState],
        commitments_values: &[&[SIMDField<C>]],
        parallel_count: usize,
        is_broadcast: &[bool],
    ) -> DummyProof {
        check_inputs(kernel, commitments_values, parallel_count, is_broadcast);
        let mut res = vec![];
        for i in 0..parallel_count {
            let lc_input = prepare_inputs(
                kernel.layered_circuit(),
                kernel.layered_circuit_input(),
                commitments_values,
                is_broadcast,
                i,
            );
            let (_, cond) = kernel
                .layered_circuit()
                .eval_with_public_inputs_simd(lc_input, &[]);
            for x in cond.iter() {
                if !*x {
                    panic!("constraints not satisfied");
                }
            }
            res.push(cond);
        }
        DummyProof { cond: res }
    }
    fn verify_kernel(
        _verifier_setup: &Self::VerifierSetup,
        kernel: &Kernel<C>,
        proof: &Self::Proof,
        commitments: &[&Self::Commitment],
        parallel_count: usize,
        is_broadcast: &[bool],
    ) -> bool {
        let values = commitments.iter().map(|c| &c.vals[..]).collect::<Vec<_>>();
        check_inputs(kernel, &values, parallel_count, is_broadcast);
        for i in 0..parallel_count {
            let lc_input = prepare_inputs(
                kernel.layered_circuit(),
                kernel.layered_circuit_input(),
                &values,
                is_broadcast,
                i,
            );
            let (_, cond) = kernel
                .layered_circuit()
                .eval_with_public_inputs_simd(lc_input, &[]);
            if cond != proof.cond[i] {
                return false;
            }
        }
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
impl<C: Config> ProvingSystem<C> for DummyProvingSystem<C> {
    type ProverSetup = <Self as KernelWiseProvingSystem<C>>::ProverSetup;
    type VerifierSetup = <Self as KernelWiseProvingSystem<C>>::VerifierSetup;
    type Proof = CombinedProof<C, Self>;

    fn setup(computation_graph: &ComputationGraph<C>) -> (Self::ProverSetup, Self::VerifierSetup) {
        <Self as KernelWiseProvingSystem<C>>::setup(computation_graph)
    }

    fn prove(
        prover_setup: &Self::ProverSetup,
        computation_graph: &ComputationGraph<C>,
        device_memories: Vec<Vec<SIMDField<C>>>,
    ) -> Self::Proof {
        let (commitments, states) = device_memories
            .iter()
            .map(|device_memory| {
                <Self as KernelWiseProvingSystem<C>>::commit(prover_setup, &device_memory[..])
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let proofs: Vec<<Self as KernelWiseProvingSystem<C>>::Proof> = computation_graph
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

                <Self as KernelWiseProvingSystem<C>>::prove_kernel(
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
        computation_graph: &ComputationGraph<C>,
        proof: &Self::Proof,
    ) -> bool {
        let verified = proof
            .proofs
            .par_iter()
            .zip(computation_graph.proof_templates().par_iter())
            .map(|(local_proof, template)| {
                let local_commitments = template
                    .commitment_indices()
                    .iter()
                    .map(|idx| &proof.commitments[*idx])
                    .collect::<Vec<_>>();

                <Self as KernelWiseProvingSystem<C>>::verify_kernel(
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
        <Self as KernelWiseProvingSystem<C>>::post_process();
    }
}
