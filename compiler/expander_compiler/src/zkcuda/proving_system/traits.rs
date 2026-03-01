use serdes::ExpSerde;

use super::super::{context::ComputationGraph, kernel::Kernel};

use crate::circuit::config::{Config, SIMDField};

pub trait Commitment<C: Config>: Clone + ExpSerde {
    fn vals_len(&self) -> usize;
}

pub trait KernelWiseProvingSystem<C: Config> {
    type ProverSetup: Clone + Send + Sync + ExpSerde;
    type VerifierSetup: Clone + Send + Sync + ExpSerde;
    type Proof: Clone + Send + Sync + ExpSerde;
    type Commitment: Commitment<C> + Send + Sync + ExpSerde;
    type CommitmentState: Clone + Send + Sync + ExpSerde;

    fn setup(computation_graph: &ComputationGraph<C>) -> (Self::ProverSetup, Self::VerifierSetup);

    fn commit(
        prover_setup: &Self::ProverSetup,
        vals: &[SIMDField<C>],
    ) -> (Self::Commitment, Self::CommitmentState);

    #[allow(clippy::too_many_arguments)]
    fn prove_kernel(
        prover_setup: &Self::ProverSetup,
        kernel: &Kernel<C>,
        commitments: &[&Self::Commitment],
        commitments_state: &[&Self::CommitmentState],
        commitments_values: &[&[SIMDField<C>]],
        parallel_count: usize,
        is_broadcast: &[bool],
    ) -> Self::Proof;

    fn verify_kernel(
        verifier_setup: &Self::VerifierSetup,
        kernel: &Kernel<C>,
        proof: &Self::Proof,
        commitments: &[&Self::Commitment],
        parallel_count: usize,
        is_broadcast: &[bool],
    ) -> bool;

    fn post_process() {}
}

#[derive(ExpSerde)]
pub struct CombinedProof<C: Config, KP: KernelWiseProvingSystem<C>> {
    pub commitments: Vec<KP::Commitment>,
    pub proofs: Vec<KP::Proof>,
}

impl<C: Config, KP: KernelWiseProvingSystem<C>> Clone for CombinedProof<C, KP> {
    fn clone(&self) -> Self {
        CombinedProof {
            commitments: self.commitments.clone(),
            proofs: self.proofs.clone(),
        }
    }
}

pub trait ProvingSystem<C: Config> {
    type ProverSetup: Clone + Send + Sync + ExpSerde;
    type VerifierSetup: Clone + Send + Sync + ExpSerde;
    type Proof: Clone + Send + Sync + ExpSerde;

    fn setup(computation_graph: &ComputationGraph<C>) -> (Self::ProverSetup, Self::VerifierSetup);

    fn prove(
        prover_setup: &Self::ProverSetup,
        computation_graph: &ComputationGraph<C>,
        device_memories: Vec<Vec<SIMDField<C>>>,
    ) -> Self::Proof;

    fn verify(
        verifier_setup: &Self::VerifierSetup,
        computation_graph: &ComputationGraph<C>,
        proof: &Self::Proof,
    ) -> bool;

    /// This is a dedicated function to stop the running service
    /// For most proving systems, this is a no-op
    fn post_process() {}
}
