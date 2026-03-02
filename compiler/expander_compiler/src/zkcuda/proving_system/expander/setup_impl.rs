use std::collections::HashMap;

use gkr_engine::{GKREngine, MPIConfig};

use crate::{
    frontend::Config,
    zkcuda::{
        context::ComputationGraph,
        proving_system::expander::{
            structs::{ExpanderProverSetup, ExpanderVerifierSetup},
            utils::pcs_testing_setup_fixed_seed,
        },
    },
};

pub fn local_setup_impl<C, ECCConfig>(
    computation_graph: &ComputationGraph<ECCConfig>,
) -> (
    ExpanderProverSetup<C::FieldConfig, C::PCSConfig>,
    ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>,
)
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    let mut p_keys = HashMap::new();
    let mut v_keys = HashMap::new();
    for commitment_len in computation_graph.commitments_lens().iter() {
        if p_keys.contains_key(commitment_len) {
            continue;
        }
        let (_params, p_key, v_key, _scratch) = pcs_testing_setup_fixed_seed::<
            C::FieldConfig,
            C::TranscriptConfig,
            C::PCSConfig,
        >(*commitment_len, &MPIConfig::prover_new());
        p_keys.insert(*commitment_len, p_key);
        v_keys.insert(*commitment_len, v_key);
    }

    (
        ExpanderProverSetup { p_keys },
        ExpanderVerifierSetup { v_keys },
    )
}
