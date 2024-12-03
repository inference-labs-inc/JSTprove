use builder::RootBuilder;

use crate::circuit::{ir, layered};

mod api;
mod builder;
mod circuit;
mod variables;
mod witness;

pub use circuit::declare_circuit;
pub type API<C> = builder::RootBuilder<C>;
pub use crate::circuit::config::*;
pub use crate::field::{Field, BN254, GF2, M31};
pub use crate::utils::error::Error;
pub use api::BasicAPI;
pub use builder::Variable;
pub use circuit::Define;
pub use witness::WitnessSolver;

pub mod internal {
    pub use super::circuit::{
        declare_circuit_default, declare_circuit_dump_into, declare_circuit_field_type,
        declare_circuit_load_from, declare_circuit_num_vars,
    };
    pub use super::variables::{DumpLoadTwoVariables, DumpLoadVariables};
    pub use crate::utils::serde::Serde;
}

pub mod extra {
    pub use super::api::UnconstrainedAPI;
    pub use crate::utils::serde::Serde;
}

#[cfg(test)]
mod tests;

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

pub struct CompileResult<C: Config> {
    pub witness_solver: WitnessSolver<C>,
    pub layered_circuit: layered::Circuit<C>,
}

pub fn compile<C: Config, Cir: internal::DumpLoadTwoVariables<Variable> + Define<C> + Clone>(
    circuit: &Cir,
) -> Result<CompileResult<C>, Error> {
    let root = build(circuit);
    let (irw, lc) = crate::compile::compile::<C>(&root)?;
    Ok(CompileResult {
        witness_solver: WitnessSolver { circuit: irw },
        layered_circuit: lc,
    })
}
