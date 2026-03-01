use builder::RootBuilder;

use crate::circuit::layered::{CrossLayerInputType, NormalInputType};
use crate::circuit::{ir, layered};

pub mod api;
pub mod builder;
pub mod circuit;
pub mod debug;
pub mod sub_circuit;
pub mod variables;
pub mod witness;

pub use circuit::declare_circuit;
pub type API<C> = builder::RootBuilder<C>;
pub use crate::circuit::config::*;
pub use crate::compile::CompileOptions;
pub use crate::field::{BN254Fr, Field, FieldArith, GF2, M31};
pub use crate::hints::registry::{EmptyHintCaller, HintCaller, HintRegistry};
pub use crate::utils::error::Error;
pub use api::{BasicAPI, RootAPI};
pub use builder::Variable;
pub use circuit::Define;
pub use macros::memorized;
pub use witness::WitnessSolver;

pub mod internal {
    pub use super::circuit::{
        declare_circuit_default, declare_circuit_dump_into, declare_circuit_field_type,
        declare_circuit_load_from, declare_circuit_num_vars,
    };
    pub use super::variables::{DumpLoadTwoVariables, DumpLoadVariables};
    // pub use crate::utils::serde::Serde;
}

pub mod extra {

    pub use super::api::UnconstrainedAPI;
    pub use super::debug::DebugBuilder;
    pub use super::sub_circuit::{
        HashStructureAndPrimitive, JoinVecVariables, RebuildVecVariables,
    };
    pub use crate::hints::registry::{EmptyHintCaller, HintCaller, HintRegistry};
    // pub use crate::utils::serde::Serde;

    use super::{internal, CircuitField, Config, Define, Variable};

    pub fn debug_eval<
        C: Config,
        Cir: internal::DumpLoadTwoVariables<Variable> + Define<C> + Clone,
        CA: internal::DumpLoadTwoVariables<CircuitField<C>>,
        H: HintCaller<CircuitField<C>>,
    >(
        circuit: &Cir,
        assignment: &CA,
        hint_caller: H,
    ) -> Vec<CircuitField<C>> {
        let (num_inputs, num_public_inputs) = circuit.num_vars();
        let (a_num_inputs, a_num_public_inputs) = assignment.num_vars();
        assert_eq!(num_inputs, a_num_inputs);
        assert_eq!(num_public_inputs, a_num_public_inputs);
        let mut inputs = Vec::new();
        let mut public_inputs = Vec::new();
        assignment.dump_into(&mut inputs, &mut public_inputs);
        let (mut root_builder, input_variables, public_input_variables) =
            DebugBuilder::<C, H>::new(inputs, public_inputs, hint_caller);
        let mut circuit = circuit.clone();
        let mut vars_ptr = input_variables.as_slice();
        let mut public_vars_ptr = public_input_variables.as_slice();
        circuit.load_from(&mut vars_ptr, &mut public_vars_ptr);
        circuit.define(&mut root_builder);
        root_builder.get_outputs()
    }
}

#[cfg(test)]
mod tests;

pub struct CompileResult<C: Config> {
    pub witness_solver: WitnessSolver<C>,
    pub layered_circuit: layered::Circuit<C, NormalInputType>,
}

pub struct CompileResultCrossLayer<C: Config> {
    pub witness_solver: WitnessSolver<C>,
    pub layered_circuit: layered::Circuit<C, CrossLayerInputType>,
}

fn build<C: Config, Cir: internal::DumpLoadTwoVariables<Variable> + Define<C> + Clone>(
    circuit: &Cir,
) -> ir::source::RootCircuit<C> {
    let (num_inputs, num_public_inputs) = circuit.num_vars();
    let (mut root_builder, input_variables, public_input_variables) =
        RootBuilder::<C>::new(num_inputs, num_public_inputs);
    let mut circuit = circuit.clone();
    let mut vars_ptr = input_variables.as_slice();
    let mut public_vars_ptr = public_input_variables.as_slice();
    circuit.load_from(&mut vars_ptr, &mut public_vars_ptr);
    circuit.define(&mut root_builder);
    root_builder.build()
}

pub fn compile<C: Config, Cir: internal::DumpLoadTwoVariables<Variable> + Define<C> + Clone>(
    circuit: &Cir,
    options: CompileOptions,
) -> Result<CompileResult<C>, Error> {
    let root = build(circuit);
    let (irw, lc) = crate::compile::compile_with_options::<C, _>(&root, options)?;
    Ok(CompileResult {
        witness_solver: WitnessSolver { circuit: irw },
        layered_circuit: lc,
    })
}

pub fn compile_cross_layer<
    C: Config,
    Cir: internal::DumpLoadTwoVariables<Variable> + Define<C> + Clone,
>(
    circuit: &Cir,
    options: CompileOptions,
) -> Result<CompileResultCrossLayer<C>, Error> {
    let root = build(circuit);
    let (irw, lc) = crate::compile::compile_with_options::<C, _>(&root, options)?;
    Ok(CompileResultCrossLayer {
        witness_solver: WitnessSolver { circuit: irw },
        layered_circuit: lc,
    })
}
