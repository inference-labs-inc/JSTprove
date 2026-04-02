// Copyright 2025 Irreducible Inc.
//! Circuit representation in the evaluation form.
//!
//! The main purpose of the evaluation form is to evaluate and assign the intermediate witness
//! values. Those are also referred as internal wires.

mod builder;
mod const_eval;
mod interpreter;
#[cfg(test)]
mod tests;

use binius_core::{ValueIndex, ValueVec};
pub use builder::BytecodeBuilder;
pub use const_eval::evaluate_gate_constants;
use cranelift_entity::SecondaryMap;

use crate::compiler::{
    circuit::PopulateError,
    gate,
    gate_graph::{GateGraph, Wire},
    hints::HintRegistry,
    pathspec::PathSpecTree,
};

/// Compiled evaluation form for circuit witness computation
pub struct EvalForm {
    /// Compiled bytecode instructions
    bytecode: Vec<u8>,
    /// Number of evaluation instructions
    n_eval_insn: usize,
    /// Registered hint handlers
    hint_registry: HintRegistry,
}

impl EvalForm {
    /// Build the evaluation form from the gate graph
    pub(crate) fn build(
        gate_graph: &GateGraph,
        wire_mapping: &SecondaryMap<Wire, ValueIndex>,
    ) -> Self {
        let mut builder = BytecodeBuilder::new();
        let mut hint_registry = HintRegistry::new();

        // Combined wire to register mapping
        let wire_to_reg = |wire: Wire| -> u32 {
            if let Some(&ValueIndex(idx)) = wire_mapping.get(wire) {
                idx // ValueVec index
            } else {
                panic!("Wire {wire:?} not mapped");
            }
        };

        // Build bytecode for each gate
        for (gate_id, data) in gate_graph.gates.iter() {
            gate::emit_gate_bytecode(
                gate_id,
                data,
                gate_graph,
                &mut builder,
                wire_to_reg,
                &mut hint_registry,
            );
        }

        let (bytecode, n_eval_insn) = builder.finalize();
        EvalForm {
            bytecode,
            n_eval_insn,
            hint_registry,
        }
    }

    /// Execute the evaluation form to populate witness values
    pub fn evaluate(
        &self,
        value_vec: &mut ValueVec,
        path_spec_tree: Option<&PathSpecTree>,
    ) -> Result<(), PopulateError> {
        let mut interpreter = interpreter::Interpreter::new(&self.bytecode, &self.hint_registry);
        interpreter.run_with_value_vec(value_vec, path_spec_tree)?;
        Ok(())
    }

    /// Get the number of evaluation instructions
    pub fn n_eval_insn(&self) -> usize {
        self.n_eval_insn
    }
}
