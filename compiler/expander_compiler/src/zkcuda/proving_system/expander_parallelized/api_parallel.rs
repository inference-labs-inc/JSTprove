use std::io::Cursor;

use crate::circuit::config::Config;
use crate::frontend::SIMDField;
use crate::utils::misc::next_power_of_two;
use crate::zkcuda::context::ComputationGraph;
use crate::zkcuda::proving_system::expander::structs::{
    ExpanderProverSetup, ExpanderVerifierSetup,
};
use crate::zkcuda::proving_system::expander::verify_impl::verify_pcs_opening_and_aggregation_no_mpi;
use crate::zkcuda::proving_system::expander_parallelized::client_utils::{
    client_launch_server_and_setup, client_parse_args, client_send_witness_and_prove, wait_async,
    ClientHttpHelper,
};
use crate::zkcuda::proving_system::{CombinedProof, ProvingSystem};

use super::super::Expander;

use arith::Field;
use expander_utils::timer::Timer;
use gkr::gkr_verify;
use gkr_engine::{FieldEngine, GKREngine, Transcript};

pub struct ParallelizedExpander<C: GKREngine> {
    _config: std::marker::PhantomData<C>,
}

impl<C: GKREngine, ECCConfig: Config<FieldConfig = C::FieldConfig>> ProvingSystem<ECCConfig>
    for ParallelizedExpander<C>
{
    type ProverSetup = ExpanderProverSetup<C::FieldConfig, C::PCSConfig>;
    type VerifierSetup = ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>;
    type Proof = CombinedProof<ECCConfig, Expander<C>>;

    fn setup(
        computation_graph: &crate::zkcuda::context::ComputationGraph<ECCConfig>,
    ) -> (Self::ProverSetup, Self::VerifierSetup) {
        let server_binary =
            client_parse_args().unwrap_or("../target/release/expander_server".to_owned());
        client_launch_server_and_setup::<C, ECCConfig>(&server_binary, computation_graph, false)
    }

    fn prove(
        _prover_setup: &Self::ProverSetup,
        _computation_graph: &crate::zkcuda::context::ComputationGraph<ECCConfig>,
        device_memories: Vec<Vec<SIMDField<ECCConfig>>>,
    ) -> Self::Proof {
        client_send_witness_and_prove(device_memories)
    }

    fn verify(
        verifier_setup: &Self::VerifierSetup,
        computation_graph: &ComputationGraph<ECCConfig>,
        proof: &Self::Proof,
    ) -> bool {
        let verification_timer = Timer::new("Verify all kernels", true);
        let verified = proof
            .proofs
            .iter()
            .zip(computation_graph.proof_templates().iter())
            .all(|(local_proof, template)| {
                let local_commitments = template
                    .commitment_indices()
                    .iter()
                    .map(|idx| &proof.commitments[*idx])
                    .collect::<Vec<_>>();

                let parallel_count = next_power_of_two(template.parallel_count());
                if local_proof.data.len() != parallel_count {
                    println!(
                        "Proof data length mismatch: expected {parallel_count}, got {}",
                        local_proof.data.len()
                    );
                    return false;
                }
                let kernel = &computation_graph.kernels()[template.kernel_id()];
                let mut expander_circuit = kernel.layered_circuit().export_to_expander_flatten();

                for i in 0..parallel_count {
                    let mut transcript = C::TranscriptConfig::new();
                    expander_circuit.fill_rnd_coefs(&mut transcript);

                    let mut cursor = Cursor::new(&local_proof.data[i].bytes);
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
                        &local_commitments,
                        template.is_broadcast(),
                        i,
                        parallel_count,
                        &mut transcript,
                    );

                    if !verified {
                        println!("Failed to verify PCS for parallel index {i}");
                        return false;
                    }
                }
                true
            });
        verification_timer.stop();

        verified
    }

    fn post_process() {
        wait_async(ClientHttpHelper::request_exit())
    }
}
