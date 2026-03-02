use gkr_engine::{GKREngine, MPIConfig, MPIEngine};
use serdes::ExpSerde;

use crate::{
    frontend::{Config, SIMDField},
    zkcuda::{
        context::ComputationGraph,
        proving_system::{
            expander::{
                setup_impl::local_setup_impl,
                structs::{ExpanderProverSetup, ExpanderVerifierSetup},
            },
            expander_parallelized::{
                prove_impl::mpi_prove_impl, shared_memory_utils::SharedMemoryEngine,
            },
            CombinedProof, Expander, ParallelizedExpander,
        },
    },
};

pub trait ServerFns<C, ECCConfig>
where
    C: gkr_engine::GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    fn setup_request_handler(
        global_mpi_config: &MPIConfig,
        setup_file: Option<String>,
        computation_graph: &mut ComputationGraph<ECCConfig>,
        prover_setup: &mut ExpanderProverSetup<C::FieldConfig, C::PCSConfig>,
        verifier_setup: &mut ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>,
    );

    fn prove_request_handler(
        global_mpi_config: &MPIConfig,
        prover_setup: &ExpanderProverSetup<C::FieldConfig, C::PCSConfig>,
        computation_graph: &ComputationGraph<ECCConfig>,
        values: &[impl AsRef<[SIMDField<C>]>],
    ) -> Option<CombinedProof<ECCConfig, Expander<C>>>;

    fn setup_shared_witness(
        _global_mpi_config: &MPIConfig,
        witness_target: &mut Vec<Vec<SIMDField<C>>>,
    ) {
        witness_target.clear();

        let witness_v = SharedMemoryEngine::read_witness_from_shared_memory::<C::FieldConfig>();
        *witness_target = witness_v;
    }

    fn shared_memory_clean_up(
        _global_mpi_config: &MPIConfig,
        _computation_graph: ComputationGraph<ECCConfig>,
        _witness: Vec<Vec<SIMDField<C>>>,
    ) {
    }
}

impl<C, ECCConfig> ServerFns<C, ECCConfig> for ParallelizedExpander<C>
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    fn setup_request_handler(
        global_mpi_config: &MPIConfig,
        setup_file: Option<String>,
        computation_graph: &mut ComputationGraph<ECCConfig>,
        prover_setup: &mut ExpanderProverSetup<C::FieldConfig, C::PCSConfig>,
        verifier_setup: &mut ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>,
    ) {
        let setup_file = setup_file.expect("Setup file path must be provided");

        read_circuit::<C, ECCConfig>(global_mpi_config, setup_file, computation_graph);
        if global_mpi_config.is_root() {
            (*prover_setup, *verifier_setup) = local_setup_impl::<C, ECCConfig>(computation_graph);
        }
    }

    fn prove_request_handler(
        global_mpi_config: &MPIConfig,
        prover_setup: &ExpanderProverSetup<C::FieldConfig, C::PCSConfig>,
        computation_graph: &ComputationGraph<ECCConfig>,
        values: &[impl AsRef<[SIMDField<C>]>],
    ) -> Option<CombinedProof<ECCConfig, Expander<C>>>
    where
        C: GKREngine,
        ECCConfig: Config<FieldConfig = C::FieldConfig>,
    {
        mpi_prove_impl(global_mpi_config, prover_setup, computation_graph, values)
    }
}

pub fn broadcast_string(_global_mpi_config: &MPIConfig, string: Option<String>) -> String {
    string.expect("String must be provided in broadcast_string")
}

pub fn read_circuit<C, ECCConfig>(
    _global_mpi_config: &MPIConfig,
    setup_file: String,
    computation_graph: &mut ComputationGraph<ECCConfig>,
) where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    let computation_graph_bytes =
        std::fs::read(setup_file).expect("Failed to read computation graph from file");

    let cg = ComputationGraph::<ECCConfig>::deserialize_from(std::io::Cursor::new(
        computation_graph_bytes,
    ))
    .expect("Failed to deserialize computation graph from file");

    *computation_graph = cg;
}
