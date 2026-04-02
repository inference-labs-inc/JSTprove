// Copyright 2025 Irreducible Inc.
//! Constant propagation optimization pass.
//!
//! This module implements constant propagation for the gate graph, identifying gates
//! with all-constant inputs, evaluating them at compile time, and replacing their
//! outputs with constant wires.

use std::collections::{HashSet, VecDeque};

use super::{
    eval_form::evaluate_gate_constants,
    gate_graph::{Gate, GateGraph, WireKind},
};

/// Performs constant propagation on the gate graph.
///
/// This optimization identifies gates with all-constant inputs, evaluates them at compile time,
/// and replaces their outputs with constant wires. The process iterates until no more constants
/// can be propagated.
///
/// Returns the number of wires that were replaced with constants.
pub fn constant_propagation(graph: &mut GateGraph) -> usize {
    // First rebuild use-def chains to ensure they're up to date
    graph.rebuild_use_def_chains();

    // Initialize worklist with all gates that might be evaluable
    let mut worklist: VecDeque<Gate> = VecDeque::new();
    let mut in_worklist: HashSet<Gate> = HashSet::new();

    // Add all gates that use constant wires to the initial worklist.
    //
    // Note that wire uses are sorted. This is to ensure that the pass is deterministic.
    for (wire, _) in graph.iter_const_wires() {
        let mut gates_using_wire: Vec<Gate> = graph.get_wire_uses(wire).iter().copied().collect();
        gates_using_wire.sort();
        for gate in gates_using_wire {
            if in_worklist.insert(gate) {
                worklist.push_back(gate);
            }
        }
    }

    let mut total_replaced = 0;

    // Process worklist until empty
    while let Some(gate) = worklist.pop_front() {
        in_worklist.remove(&gate);

        // Try to evaluate this gate with constant inputs
        if let Some(eval_result) = try_evaluate_gate_with_constants(graph, gate) {
            match eval_result {
                Ok(output_values) => {
                    let output_wires = {
                        let gate_data = graph.gate_data(gate);
                        let gate_param = gate_data.gate_param();
                        gate_param.outputs.to_vec()
                    };
                    for (i, &output_wire) in output_wires.iter().enumerate() {
                        // Skip if output is already constant
                        if graph.wire_data(output_wire).kind.is_const() {
                            continue;
                        }

                        // Replace the wire with a constant and get only the gates that were
                        // affected.
                        //
                        // Perform sorting to ensure deterministic order.
                        let (_const_wire, num_updates, mut affected_gates) =
                            graph.replace_wire_with_constant(output_wire, output_values[i]);
                        affected_gates.sort();
                        if num_updates > 0 {
                            total_replaced += num_updates;
                            for user_gate in affected_gates {
                                if in_worklist.insert(user_gate) {
                                    worklist.push_back(user_gate);
                                }
                            }
                        }
                    }
                }
                Err(err) => {
                    // TODO: bubble up the error. For now we just panic.
                    panic!("Constant propagation detected an always-failing gate: {err}");
                }
            }
        }
    }

    total_replaced
}

/// Tries to evaluate a gate with constant inputs.
///
/// Returns Some(output_values) if the gate can be constant-evaluated, None otherwise.
/// This consolidates the input checking and evaluation logic.
fn try_evaluate_gate_with_constants(
    graph: &GateGraph,
    gate: Gate,
) -> Option<Result<Vec<binius_core::word::Word>, String>> {
    let gate_data = graph.gate_data(gate);
    let gate_param = gate_data.gate_param();

    let mut input_constants = Vec::new();
    for &input_wire in gate_param.inputs {
        if let WireKind::Constant(val) = graph.wire_data(input_wire).kind {
            input_constants.push(val);
        } else {
            // Not all inputs are constant, can't evaluate.
            return None;
        }
    }

    // Evaluate the gate with constant inputs
    let result = evaluate_gate_constants(graph, gate, &input_constants);
    Some(result)
}

#[cfg(test)]
mod tests {
    use binius_core::word::Word;

    use super::*;
    use crate::compiler::gate::opcode::Opcode;

    #[test]
    fn test_constant_propagation() {
        let mut graph = GateGraph::new();
        let root = graph.path_spec_tree.root();

        // Create constant wires
        let const_5 = graph.add_constant(Word(5));
        let const_3 = graph.add_constant(Word(3));

        // Create a gate with constant inputs
        let xor_out = graph.add_witness();
        let _xor_gate = graph.emit_gate(root, Opcode::Bxor, vec![const_5, const_3], vec![xor_out]);

        // Create another gate that uses the output of the first
        let const_1 = graph.add_constant(Word(1));
        let and_out = graph.add_witness();
        let and_gate = graph.emit_gate(root, Opcode::Band, vec![xor_out, const_1], vec![and_out]);

        // Create a final gate that uses and_out to verify propagation
        let test_out = graph.add_witness();
        let test_gate = graph.emit_gate(root, Opcode::Bxor, vec![and_out, and_out], vec![test_out]);

        // Initially, xor_out and and_out are not constants
        assert!(!matches!(graph.wires[xor_out].kind, WireKind::Constant(_)));
        assert!(!matches!(graph.wires[and_out].kind, WireKind::Constant(_)));

        // Run constant propagation
        let replaced = constant_propagation(&mut graph);

        // We replace: xor_out in and_gate, and_out in test_gate (twice, since both inputs)
        assert_eq!(replaced, 3);

        // The original wires remain as witness wires
        assert!(matches!(graph.wires[xor_out].kind, WireKind::Witness));
        assert!(matches!(graph.wires[and_out].kind, WireKind::Witness));

        // But the gates that used them should now use constant wires
        // Check that and_gate now uses a constant wire with value 6 instead of xor_out
        let and_gate_data = &graph.gates[and_gate];
        let and_inputs = and_gate_data.gate_param().inputs;
        // First input should be a constant with value 6 (5 ^ 3)
        match graph.wires[and_inputs[0]].kind {
            WireKind::Constant(val) => assert_eq!(val, Word(6)),
            _ => panic!("Expected and_gate's first input to be constant 6"),
        }

        // Check that test_gate now uses a constant wire with value 0 instead of and_out
        let test_gate_data = &graph.gates[test_gate];
        let test_inputs = test_gate_data.gate_param().inputs;
        // Both inputs should be constants with value 0 (6 & 1)
        match graph.wires[test_inputs[0]].kind {
            WireKind::Constant(val) => assert_eq!(val, Word(0)),
            _ => panic!("Expected test_gate's input to be constant 0"),
        }
    }

    #[test]
    fn test_constant_propagation_with_shifts() {
        let mut graph = GateGraph::new();
        let root = graph.path_spec_tree.root();

        // Create a constant wire
        let const_16 = graph.add_constant(Word(16));

        // Create a shift right gate
        let shr_out = graph.add_witness();
        let _shr_gate = graph.emit_gate_imm(root, Opcode::Shr, vec![const_16], vec![shr_out], 2);

        // Create a shift left gate using the output
        let shl_out = graph.add_witness();
        let shl_gate = graph.emit_gate_imm(root, Opcode::Shl, vec![shr_out], vec![shl_out], 1);

        // Create a test gate to verify propagation
        let test_out = graph.add_witness();
        let test_gate = graph.emit_gate(root, Opcode::Bxor, vec![shl_out, shl_out], vec![test_out]);

        // Run constant propagation
        let replaced = constant_propagation(&mut graph);
        // We replace: shr_out in shl_gate, shl_out in test_gate (twice)
        assert_eq!(replaced, 3);

        // The original wires remain as witness wires
        assert!(matches!(graph.wires[shr_out].kind, WireKind::Witness));
        assert!(matches!(graph.wires[shl_out].kind, WireKind::Witness));

        // Check that shl_gate now uses a constant wire with value 4 (16 >> 2)
        let shl_gate_data = &graph.gates[shl_gate];
        let shl_inputs = shl_gate_data.gate_param().inputs;
        match graph.wires[shl_inputs[0]].kind {
            WireKind::Constant(val) => assert_eq!(val, Word(4)),
            _ => panic!("Expected shl_gate's input to be constant 4"),
        }

        // Check that test_gate now uses a constant wire with value 8 (4 << 1)
        let test_gate_data = &graph.gates[test_gate];
        let test_inputs = test_gate_data.gate_param().inputs;
        match graph.wires[test_inputs[0]].kind {
            WireKind::Constant(val) => assert_eq!(val, Word(8)),
            _ => panic!("Expected test_gate's input to be constant 8"),
        }
    }

    #[test]
    fn test_constant_propagation_with_hint() {
        let mut graph = GateGraph::new();
        let root = graph.path_spec_tree.root();

        // Test BigUintDivideHint with constants
        // dividend = 100, divisor = 7
        // quotient should be 14, remainder should be 2
        let dividend = graph.add_constant(Word(100));
        let divisor = graph.add_constant(Word(7));

        // BigUintDivideHint has variable shape, so we need to specify dimensions
        // For single-limb division: [dividend_limbs, divisor_limbs] = [1, 1]
        let quotient = graph.add_witness();
        let remainder = graph.add_witness();

        let _hint_gate = graph.emit_gate_generic(
            root,
            Opcode::BigUintDivideHint,
            vec![dividend, divisor],
            vec![quotient, remainder],
            &[1, 1], // [dividend_limbs, divisor_limbs] = [1, 1] for single word division
            &[],
        );

        // Create gates that use the outputs to verify propagation
        let test_q = graph.add_witness();
        let test_r = graph.add_witness();
        let test_q_gate =
            graph.emit_gate(root, Opcode::Bxor, vec![quotient, quotient], vec![test_q]);
        let test_r_gate =
            graph.emit_gate(root, Opcode::Bxor, vec![remainder, remainder], vec![test_r]);

        // Run constant propagation
        let replaced = constant_propagation(&mut graph);
        // We replace: quotient in test_q_gate (twice), remainder in test_r_gate (twice)
        assert_eq!(replaced, 4);

        // The original wires remain as witness wires
        assert!(matches!(graph.wires[quotient].kind, WireKind::Witness));
        assert!(matches!(graph.wires[remainder].kind, WireKind::Witness));

        // Check that test gates now use constant wires
        let test_q_data = &graph.gates[test_q_gate];
        let test_q_inputs = test_q_data.gate_param().inputs;
        match graph.wires[test_q_inputs[0]].kind {
            WireKind::Constant(val) => assert_eq!(val, Word(14)), // 100 / 7 = 14
            _ => panic!("Expected test_q_gate's input to be constant 14"),
        }

        let test_r_data = &graph.gates[test_r_gate];
        let test_r_inputs = test_r_data.gate_param().inputs;
        match graph.wires[test_r_inputs[0]].kind {
            WireKind::Constant(val) => assert_eq!(val, Word(2)), // 100 % 7 = 2
            _ => panic!("Expected test_r_gate's input to be constant 2"),
        }
    }

    #[test]
    #[should_panic(expected = "Constant propagation detected an always-failing gate")]
    fn test_constant_propagation_with_failing_gate() {
        let mut graph = GateGraph::new();
        let root = graph.path_spec_tree.root();

        // Create an Assert0 gate with a non-zero constant input
        // This should fail evaluation because Assert0 expects the input to be zero
        let non_zero_const = graph.add_constant(Word(42)); // Non-zero value
        let _assert_gate = graph.emit_gate(root, Opcode::AssertZero, vec![non_zero_const], vec![]);

        // This should panic when the Assert0 gate fails during evaluation
        // because the constant input (42) is not zero
        constant_propagation(&mut graph);
    }
}
