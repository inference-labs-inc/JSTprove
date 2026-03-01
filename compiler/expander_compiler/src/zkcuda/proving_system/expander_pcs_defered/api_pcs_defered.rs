use gkr_engine::{ExpanderPCS, GKREngine};

use crate::{
    frontend::{Config, SIMDField},
    zkcuda::proving_system::{
        expander::structs::{ExpanderProverSetup, ExpanderVerifierSetup},
        expander_parallelized::client_utils::{
            client_launch_server_and_setup, client_parse_args, client_send_witness_and_prove,
            wait_async, ClientHttpHelper,
        },
        CombinedProof, Expander, ProvingSystem,
    },
};

pub struct ExpanderPCSDefered<C: GKREngine> {
    _config: std::marker::PhantomData<C>,
}

impl<C, ECCConfig> ProvingSystem<ECCConfig> for ExpanderPCSDefered<C>
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,

    <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::Commitment:
        AsRef<<C::PCSConfig as ExpanderPCS<C::FieldConfig>>::Commitment>,
{
    type ProverSetup = ExpanderProverSetup<C::FieldConfig, C::PCSConfig>;

    type VerifierSetup = ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>;

    type Proof = CombinedProof<ECCConfig, Expander<C>>;

    fn setup(
        computation_graph: &crate::zkcuda::context::ComputationGraph<ECCConfig>,
    ) -> (Self::ProverSetup, Self::VerifierSetup) {
        let server_binary = client_parse_args()
            .unwrap_or("../target/release/expander_server_pcs_defered".to_owned());
        client_launch_server_and_setup::<C, ECCConfig>(&server_binary, computation_graph, true)
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
        computation_graph: &crate::zkcuda::context::ComputationGraph<ECCConfig>,
        proof: &Self::Proof,
    ) -> bool {
        super::verify_impl::verify(verifier_setup, computation_graph, proof.clone())
    }

    fn post_process() {
        wait_async(ClientHttpHelper::request_exit())
    }
}
