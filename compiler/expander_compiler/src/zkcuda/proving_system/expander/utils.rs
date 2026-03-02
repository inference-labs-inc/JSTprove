use expander_circuit::Circuit as ExpanderCircuit;
use gkr_engine::{ExpanderPCS, FieldEngine, MPIConfig, StructuredReferenceString, Transcript};
use poly_commit::expander_pcs_init_testing_only;

#[allow(clippy::type_complexity)]
pub fn pcs_testing_setup_fixed_seed<F: FieldEngine, T: Transcript, PCS: ExpanderPCS<F>>(
    vals_len: usize,
    mpi_config: &MPIConfig,
) -> (
    PCS::Params,
    <PCS::SRS as StructuredReferenceString>::PKey,
    <PCS::SRS as StructuredReferenceString>::VKey,
    PCS::ScratchPad,
) {
    expander_pcs_init_testing_only::<F, PCS>(vals_len.ilog2() as usize, mpi_config)
}

pub fn max_n_vars<C: FieldEngine>(circuit: &ExpanderCircuit<C>) -> (usize, usize) {
    let mut max_num_input_var = 0;
    let mut max_num_output_var = 0;
    for layer in circuit.layers.iter() {
        max_num_input_var = max_num_input_var.max(layer.input_var_num);
        max_num_output_var = max_num_output_var.max(layer.output_var_num);
    }
    (max_num_input_var, max_num_output_var)
}
