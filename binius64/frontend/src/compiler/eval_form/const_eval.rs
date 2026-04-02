// Copyright 2025 Irreducible Inc.
//! Constant evaluation support for gates.

use binius_core::{ValueIndex, ValueVec, ValueVecLayout, Word};

use super::{BytecodeBuilder, interpreter::Interpreter};
use crate::compiler::{
    gate::{self, opcode::OpcodeShape},
    gate_graph::{Gate, GateData, GateGraph, GateParam, Wire},
    hints::HintRegistry,
};

/// Creates a wire mapping from gate wires to sequential register indices.
/// Returns the mapping and the total number of registers used.
fn create_wire_mapping(gate_param: &GateParam) -> (std::collections::HashMap<Wire, u32>, u32) {
    let mut wire_mapping = std::collections::HashMap::new();
    let mut wire_index = 0u32;

    // Helper to map a slice of wires sequentially
    let mut map_wires = |wires: &[Wire]| {
        for &wire in wires {
            wire_mapping.insert(wire, wire_index);
            wire_index += 1;
        }
    };

    // Map all wire types in order
    map_wires(gate_param.constants);
    map_wires(gate_param.inputs);
    map_wires(gate_param.outputs);
    map_wires(gate_param.aux);

    (wire_mapping, wire_index)
}

/// Sets up a ValueVec with the gate's constants loaded.
fn setup_value_vec(shape: &OpcodeShape, concrete_inputs: &[Word], wire_count: u32) -> ValueVec {
    let layout = ValueVecLayout {
        n_const: shape.const_in.len(),
        n_inout: 0,
        n_witness: 0,
        n_internal: wire_count as usize,
        offset_inout: shape.const_in.len(),
        offset_witness: shape.const_in.len(),
        committed_total_len: wire_count as usize,
        n_scratch: 0,
    };
    let mut value_vec = ValueVec::new(layout);

    // Load gate's built-in constants and provided input constants
    for (i, &const_val) in shape
        .const_in
        .iter()
        .chain(concrete_inputs.iter())
        .enumerate()
    {
        value_vec.set(i, const_val);
    }

    value_vec
}

/// Extracts output values from the ValueVec after evaluation.
fn extract_outputs(value_vec: &ValueVec, shape: &OpcodeShape) -> Vec<Word> {
    let output_start = shape.const_in.len() + shape.n_in;
    (output_start..output_start + shape.n_out)
        .map(|i| value_vec[ValueIndex(i as u32)])
        .collect()
}

/// Convenience function to evaluate a gate with constant inputs.
///
/// This is a cleaner alternative to `evaluate_constant_gate` that eliminates
/// parameter redundancy by looking up gate data internally.
pub fn evaluate_gate_constants(
    graph: &GateGraph,
    gate: Gate,
    constants: &[Word],
) -> Result<Vec<Word>, String> {
    evaluate_constant_gate(gate, &graph.gates[gate], graph, constants)
}

/// Evaluate a gate with constant inputs using the existing interpreter logic.
///
/// This function reuses the `emit_eval_bytecode` logic from each gate module
/// to ensure consistency between runtime evaluation and constant propagation.
pub fn evaluate_constant_gate(
    gate: Gate,
    data: &GateData,
    graph: &GateGraph,
    concrete_inputs: &[Word],
) -> Result<Vec<Word>, String> {
    let shape = data.shape();
    let gate_param = data.gate_param();

    // Set up wire mapping and value vector
    let (wire_mapping, wire_count) = create_wire_mapping(&gate_param);
    let mut value_vec = setup_value_vec(&shape, concrete_inputs, wire_count);

    // Create wire-to-register lookup function
    let wire_to_reg = |wire: Wire| -> u32 {
        *wire_mapping
            .get(&wire)
            .unwrap_or_else(|| panic!("Wire {:?} not mapped", wire))
    };

    // Generate bytecode for this gate
    let mut builder = BytecodeBuilder::new();
    let mut hint_registry = HintRegistry::new();
    gate::emit_gate_bytecode(
        gate,
        data,
        graph,
        &mut builder,
        wire_to_reg,
        &mut hint_registry,
    );
    let (bytecode, _) = builder.finalize();

    // Run evaluation
    let mut interpreter = Interpreter::new(&bytecode, &hint_registry);
    interpreter
        .run_with_value_vec(&mut value_vec, None)
        .map_err(|e| format!("Constant evaluation failed: {:?}", e))?;

    // Extract and return output values
    Ok(extract_outputs(&value_vec, &shape))
}

#[cfg(test)]
mod tests {
    use binius_core::Word;

    use super::*;
    use crate::compiler::{gate::opcode::Opcode, gate_graph::GateGraph};

    /// Helper to create a gate with constant inputs for testing
    fn create_test_gate(opcode: Opcode, input_values: &[Word]) -> (GateGraph, Gate, Vec<Word>) {
        let mut graph = GateGraph::new();
        let root = graph.path_spec_tree.root();

        // Create constant wires for inputs using the proper helper function
        let input_wires: Vec<_> = input_values
            .iter()
            .map(|&val| graph.add_constant(val))
            .collect();

        // Create output wires using the proper helper function
        let outputs: Vec<_> = (0..opcode.shape(&[]).n_out)
            .map(|_| graph.add_witness())
            .collect();

        let gate = graph.emit_gate(root, opcode, input_wires, outputs);

        (graph, gate, input_values.to_vec())
    }

    #[test]
    fn test_band_constant_eval() {
        let (graph, gate, constants) = create_test_gate(
            Opcode::Band,
            &[Word::from_u64(0xFF00FF00), Word::from_u64(0x0F0F0F0F)],
        );

        let result = evaluate_gate_constants(&graph, gate, &constants).unwrap();
        assert_eq!(result[0], Word::from_u64(0x0F000F00));
    }

    #[test]
    fn test_bxor_constant_eval() {
        let (graph, gate, constants) = create_test_gate(
            Opcode::Bxor,
            &[Word::from_u64(0xFF00FF00), Word::from_u64(0x0F0F0F0F)],
        );

        let result = evaluate_gate_constants(&graph, gate, &constants).unwrap();
        assert_eq!(result[0], Word::from_u64(0xF00FF00F));
    }

    #[test]
    fn test_bor_constant_eval() {
        let (graph, gate, constants) = create_test_gate(
            Opcode::Bor,
            &[Word::from_u64(0xFF00FF00), Word::from_u64(0x0F0F0F0F)],
        );

        let result = evaluate_gate_constants(&graph, gate, &constants).unwrap();
        assert_eq!(result[0], Word::from_u64(0xFF0FFF0F));
    }

    #[test]
    fn test_imul_constant_eval() {
        // Test IMUL (has 2 outputs: hi, lo)
        let (graph, gate, constants) = create_test_gate(
            Opcode::Imul,
            &[Word::from_u64(0x123456789ABCDEF0), Word::from_u64(0x10)],
        );

        let result = evaluate_gate_constants(&graph, gate, &constants).unwrap();
        assert_eq!(result[1], Word::from_u64(0x23456789ABCDEF00)); // lo
        assert_eq!(result[0], Word::from_u64(0x1)); // hi
    }

    #[test]
    fn test_iadd_cin_cout_constant_eval() {
        // Test with carry in (MSB = 1 means carry bit is 1)
        let (graph, gate, constants) = create_test_gate(
            Opcode::IaddCinCout,
            &[
                Word::from_u64(0xFFFFFFFFFFFFFFFF),
                Word::from_u64(0x1),
                Word::from_u64(0x8000000000000000), // carry in (MSB = 1)
            ],
        );

        let result = evaluate_gate_constants(&graph, gate, &constants).unwrap();
        assert_eq!(result[0], Word::from_u64(0x1)); // sum: 0xFF...FF + 1 + 1 = 1 (with overflow)
        // Carry out shows carries at all bit positions
        assert_eq!(result[1], Word::from_u64(0xFFFFFFFFFFFFFFFF));
    }

    #[test]
    fn test_isub_bin_bout_constant_eval() {
        // Test subtraction: 0x10 - 0x5 = 0xB
        let (graph, gate, constants) = create_test_gate(
            Opcode::IsubBinBout,
            &[
                Word::from_u64(0x10),
                Word::from_u64(0x5),
                Word::from_u64(0x0), // no borrow in
            ],
        );

        let result = evaluate_gate_constants(&graph, gate, &constants).unwrap();
        assert_eq!(result[0], Word::from_u64(0xB)); // diff: 0x10 - 0x5 = 0xB
        // Borrow out shows borrows at bit positions - for 0x10 - 0x5, borrows occur at bits 0-3
        assert_eq!(result[1], Word::from_u64(0xF)); // borrow out at bits 0-3
    }
}
