use gkr_engine::{GKREngine, MPIEngine};

use crate::{
    frontend::Config,
    zkcuda::{
        context::ComputationGraph,
        proving_system::{
            expander::structs::{ExpanderProverSetup, ExpanderVerifierSetup},
            expander_parallelized::server_fns::{read_circuit, ServerFns},
            expander_pcs_defered::{
                prove_impl::mpi_prove_with_pcs_defered, setup_impl::pcs_setup_max_length_only,
            },
            CombinedProof, Expander, ExpanderPCSDefered,
        },
    },
};

impl<C, ECCConfig> ServerFns<C, ECCConfig> for ExpanderPCSDefered<C>
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    fn setup_request_handler(
        global_mpi_config: &gkr_engine::MPIConfig,
        setup_file: Option<String>,
        computation_graph: &mut ComputationGraph<ECCConfig>,
        prover_setup: &mut ExpanderProverSetup<C::FieldConfig, C::PCSConfig>,
        verifier_setup: &mut ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>,
    ) {
        let setup_file = setup_file.expect("Setup file path must be provided");

        read_circuit::<C, ECCConfig>(global_mpi_config, setup_file, computation_graph);
        if global_mpi_config.is_root() {
            (*prover_setup, *verifier_setup) =
                pcs_setup_max_length_only::<C, ECCConfig>(computation_graph);
        }
    }

    fn prove_request_handler(
        global_mpi_config: &gkr_engine::MPIConfig,
        prover_setup: &ExpanderProverSetup<
            <C as gkr_engine::GKREngine>::FieldConfig,
            <C as gkr_engine::GKREngine>::PCSConfig,
        >,
        computation_graph: &ComputationGraph<ECCConfig>,
        values: &[impl AsRef<[crate::frontend::SIMDField<C>]>],
    ) -> Option<CombinedProof<ECCConfig, Expander<C>>> {
        mpi_prove_with_pcs_defered(global_mpi_config, prover_setup, computation_graph, values)
    }
}
