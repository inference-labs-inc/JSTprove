// Copyright 2025 Irreducible Inc.
use std::collections::{HashMap, HashSet};

use binius_core::word::Word;
use cranelift_entity::{PrimaryMap, SecondaryMap, entity_impl};

use crate::compiler::{
    gate::opcode::{Opcode, OpcodeShape},
    pathspec::{PathSpec, PathSpecTree},
};

#[derive(Default)]
pub struct ConstPool {
    pub pool: HashMap<Word, Wire>,
}

impl ConstPool {
    pub fn new() -> Self {
        ConstPool::default()
    }

    pub fn get(&self, value: Word) -> Option<Wire> {
        self.pool.get(&value).cloned()
    }

    pub fn insert(&mut self, word: Word, wire: Wire) {
        let prev = self.pool.insert(word, wire);
        assert!(prev.is_none());
    }
}

/// A wire through which a value flows in and out of gates.
///
/// The difference from `ValueIndex` is that a wire is abstract. Some wires could be moved during
/// compilation and some wires might be pruned altogether.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Wire(u32);
entity_impl!(Wire);

#[derive(Copy, Clone, Debug)]
pub enum WireKind {
    Constant(Word),
    Inout,
    Witness,
    /// An internal wire is a wire created inside a gate.
    Internal,
    /// A scratch wire is a temporary wire used only during evaluation.
    Scratch,
}
impl WireKind {
    /// Returns `true` if this is a constant wire.
    pub fn is_const(&self) -> bool {
        matches!(self, WireKind::Constant(_))
    }
}

#[derive(Copy, Clone)]
pub struct WireData {
    pub kind: WireKind,
}

/// Gate ID - identifies a gate in the graph
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Gate(u32);

entity_impl!(Gate);

/// A handy struct that allows a more type safe destructure.
pub struct GateParam<'a> {
    pub constants: &'a [Wire],
    pub inputs: &'a [Wire],
    pub outputs: &'a [Wire],
    pub aux: &'a [Wire],
    pub scratch: &'a [Wire],
    pub imm: &'a [u32],
}

/// Describes a particular gate in the gate graph, it's type, input and output wires and
/// immediate parameters.
pub struct GateData {
    /// The code of operation of this gate.
    pub opcode: Opcode,

    /// The input and output wires of this gate.
    ///
    /// They are laid out in the following order:
    ///
    /// - Constants
    /// - Inputs
    /// - Outputs
    /// - Aux
    /// - Scratch
    ///
    /// The number of input and output wires is specified by the opcode's shape.
    pub wires: Vec<Wire>,

    /// The immediate parameters of this gate.
    ///
    /// The immediates contain compile-time parameters of the circuits, such as shift amounts,
    /// byte indices, etc.
    ///
    /// The length of the immediates is specified by the opcode's shape.
    pub immediates: Vec<u32>,

    /// The dimensions of this gate.
    ///
    /// This is empty for gates of constant shape. When the shape is variable, the number of
    /// input, output and internal wires is a function of non-empty `dimensions`. This function is
    /// typically linear.
    pub dimensions: Vec<usize>,
}

impl GateData {
    pub fn gate_param(&self) -> GateParam<'_> {
        let shape = self.shape();
        let start_const = 0;
        let end_const = shape.const_in.len();
        let start_input = end_const;
        let end_input = start_input + shape.n_in;
        let start_output = end_input;
        let end_output = start_output + shape.n_out;
        let start_aux = end_output;
        let end_aux = start_aux + shape.n_aux;
        let start_scratch = end_aux;
        let end_scratch = start_scratch + shape.n_scratch;
        GateParam {
            constants: &self.wires[start_const..end_const],
            inputs: &self.wires[start_input..end_input],
            outputs: &self.wires[start_output..end_output],
            aux: &self.wires[start_aux..end_aux],
            scratch: &self.wires[start_scratch..end_scratch],
            imm: &self.immediates,
        }
    }

    /// The gate shape (takes dimensions into account).
    pub fn shape(&self) -> OpcodeShape {
        self.opcode.shape(&self.dimensions)
    }

    /// Ensures the gate has the right shape.
    pub fn validate_shape(&self) {
        let shape = self.shape();
        let gate_param = self.gate_param();
        assert_eq!(gate_param.inputs.len(), shape.n_in);
        assert_eq!(gate_param.outputs.len(), shape.n_out);
        assert_eq!(gate_param.aux.len(), shape.n_aux);
        assert_eq!(self.immediates.len(), shape.n_imm);
    }
}

/// Gate graph replaces the current Shared struct
pub struct GateGraph {
    // Primary maps
    pub gates: PrimaryMap<Gate, GateData>,
    pub wires: PrimaryMap<Wire, WireData>,

    pub path_spec_tree: PathSpecTree,
    pub gate_origin: SecondaryMap<Gate, PathSpec>,
    pub assertion_names: SecondaryMap<Gate, PathSpec>,

    pub const_pool: ConstPool,
    pub n_witness: usize,
    pub n_inout: usize,

    // Use-def analysis
    /// Maps each wire to the gate that defines it (if any)
    pub wire_def: SecondaryMap<Wire, Option<Gate>>,
    /// Maps each wire to the set of gates that use it
    wire_uses: SecondaryMap<Wire, HashSet<Gate>>,
}

impl GateGraph {
    pub fn new() -> Self {
        let path_spec_tree = PathSpecTree::new();
        let root = path_spec_tree.root();
        Self {
            gates: PrimaryMap::new(),
            wires: PrimaryMap::new(),
            path_spec_tree,
            gate_origin: SecondaryMap::with_default(root),
            assertion_names: SecondaryMap::with_default(root),
            const_pool: ConstPool::new(),
            n_witness: 0,
            n_inout: 0,
            wire_def: SecondaryMap::new(),
            wire_uses: SecondaryMap::new(),
        }
    }

    /// Runs a validation pass ensuring all the invariants hold.
    pub fn validate(&self) {
        // Every gate holds shape.
        for gate in self.gates.values() {
            gate.validate_shape();
        }
    }

    pub fn add_inout(&mut self) -> Wire {
        self.n_inout += 1;
        self.wires.push(WireData {
            kind: WireKind::Inout,
        })
    }

    pub fn add_witness(&mut self) -> Wire {
        self.n_witness += 1;
        self.wires.push(WireData {
            kind: WireKind::Witness,
        })
    }

    pub fn add_internal(&mut self) -> Wire {
        // Internal wires are treated as witnesses for allocation purposes
        self.n_witness += 1;
        self.wires.push(WireData {
            kind: WireKind::Internal,
        })
    }

    pub fn add_scratch(&mut self) -> Wire {
        // Scratch wires are temporary storage, not part of witness
        self.wires.push(WireData {
            kind: WireKind::Scratch,
        })
    }

    pub fn add_constant(&mut self, word: Word) -> Wire {
        if let Some(wire) = self.const_pool.get(word) {
            return wire;
        }
        let wire = self.wires.push(WireData {
            kind: WireKind::Constant(word),
        });
        self.const_pool.insert(word, wire);
        wire
    }

    /// Emits a gate with the given opcode, inputs and outputs.
    pub fn emit_gate(
        &mut self,
        gate_origin: PathSpec,
        opcode: Opcode,
        inputs: impl IntoIterator<Item = Wire>,
        outputs: impl IntoIterator<Item = Wire>,
    ) -> Gate {
        self.emit_gate_generic(gate_origin, opcode, inputs, outputs, &[], &[])
    }

    /// Emits a gate with the given opcode, inputs, outputs and a single immediate argument.
    pub fn emit_gate_imm(
        &mut self,
        gate_origin: PathSpec,
        opcode: Opcode,
        inputs: impl IntoIterator<Item = Wire>,
        outputs: impl IntoIterator<Item = Wire>,
        imm32: u32,
    ) -> Gate {
        self.emit_gate_generic(gate_origin, opcode, inputs, outputs, &[], &[imm32])
    }

    /// Creates a gate inline with the given opcode's shape parametrized with the inputs, outputs
    /// and immediates.
    ///
    /// Panics if the resulting opcode shape is not valid.
    pub fn emit_gate_generic(
        &mut self,
        gate_origin: PathSpec,
        opcode: Opcode,
        inputs: impl IntoIterator<Item = Wire>,
        outputs: impl IntoIterator<Item = Wire>,
        dimensions: &[usize],
        immediates: &[u32],
    ) -> Gate {
        let shape = opcode.shape(dimensions);
        let mut wires: Vec<Wire> = Vec::with_capacity(
            shape.const_in.len() + shape.n_in + shape.n_out + shape.n_aux + shape.n_scratch,
        );
        for c in shape.const_in {
            wires.push(self.add_constant(*c));
        }
        wires.extend(inputs);
        wires.extend(outputs);
        for _ in 0..shape.n_aux {
            // We create internal wires as auxiliary.
            wires.push(self.add_internal());
        }
        for _ in 0..shape.n_scratch {
            wires.push(self.add_scratch());
        }
        let data = GateData {
            opcode,
            wires,
            dimensions: dimensions.to_vec(),
            immediates: immediates.to_vec(),
        };
        data.validate_shape();

        let gate = self.gates.push(data);

        self.gate_origin[gate] = gate_origin;

        gate
    }

    /// Updates use-def information for a newly added gate
    fn update_use_def_for_gate(&mut self, gate: Gate) {
        let gate_data = &self.gates[gate];
        let gate_param = gate_data.gate_param();

        // Record this gate as defining its outputs
        for &output_wire in gate_param.outputs {
            self.wire_def[output_wire] = Some(gate);
        }

        // Record this gate as defining its internal wires
        for &aux_wire in gate_param.aux {
            self.wire_def[aux_wire] = Some(gate);
        }

        // Record this gate as using its inputs
        for &input_wire in gate_param.inputs {
            self.wire_uses[input_wire].insert(gate);
        }

        // Record this gate as using its constants
        for &const_wire in gate_param.constants {
            self.wire_uses[const_wire].insert(gate);
        }
    }

    /// Rebuilds the use-def chains from scratch by analyzing all gates
    pub fn rebuild_use_def_chains(&mut self) {
        // Clear existing use-def information
        self.wire_def.clear();
        self.wire_uses.clear();

        // Rebuild from all gates
        for gate in self.gates.keys() {
            self.update_use_def_for_gate(gate);
        }
    }

    /// Returns all gates that use the given wire
    pub fn get_wire_uses(&self, wire: Wire) -> &HashSet<Gate> {
        &self.wire_uses[wire]
    }

    /// Returns an iterator over all constant wires and their data
    pub fn iter_const_wires(&self) -> impl Iterator<Item = (Wire, &WireData)> {
        self.wires
            .iter()
            .filter(|(wire, _)| self.wire_data(*wire).kind.is_const())
    }

    /// Gets wire data by reference
    pub fn wire_data(&self, wire: Wire) -> &WireData {
        &self.wires[wire]
    }

    /// Gets gate data by reference
    pub fn gate_data(&self, gate: Gate) -> &GateData {
        &self.gates[gate]
    }

    /// Replaces all occurrences of a wire in a gate with another wire
    pub fn replace_gate_wire(&mut self, gate: Gate, old_wire: Wire, new_wire: Wire) {
        let gate_data = &mut self.gates[gate];
        for wire in &mut gate_data.wires {
            if *wire == old_wire {
                *wire = new_wire;
            }
        }
    }

    /// Updates use-def chains when replacing a wire use
    pub fn update_wire_use(&mut self, old_wire: Wire, new_wire: Wire, gate: Gate) {
        self.wire_uses[old_wire].remove(&gate);
        self.wire_uses[new_wire].insert(gate);
    }

    /// Replaces all uses of old_wire with a constant wire containing the given value.
    ///
    /// Returns the constant wire that was used, the number of individual wire replacements,
    /// and the list of gates that were actually affected by this replacement.
    /// This encapsulates both wire replacement and use-def chain updates.
    pub fn replace_wire_with_constant(
        &mut self,
        old_wire: Wire,
        value: Word,
    ) -> (Wire, usize, Vec<Gate>) {
        let const_wire = self.add_constant(value);

        if const_wire == old_wire {
            return (const_wire, 0, Vec::new());
        }

        // Get all users of the old wire (clone to avoid borrow conflicts)
        let users: Vec<Gate> = self.get_wire_uses(old_wire).iter().copied().collect();
        let mut total_replacements = 0;

        // Replace wire references in all user gates
        for user_gate in &users {
            // Count how many times this wire appears in this gate before replacing
            let gate_data = self.gate_data(*user_gate);
            let gate_param = gate_data.gate_param();
            let replacements_in_gate = gate_param.inputs.iter().filter(|&&w| w == old_wire).count()
                + gate_param
                    .outputs
                    .iter()
                    .filter(|&&w| w == old_wire)
                    .count();
            total_replacements += replacements_in_gate;

            self.replace_gate_wire(*user_gate, old_wire, const_wire);
            self.update_wire_use(old_wire, const_wire, *user_gate);
        }

        (const_wire, total_replacements, users)
    }
}

impl Default for GateGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::gate::opcode::Opcode;

    // Test helper functions
    fn get_wire_def(graph: &GateGraph, wire: Wire) -> Option<Gate> {
        graph.wire_def[wire]
    }

    fn wire_use_count(graph: &GateGraph, wire: Wire) -> usize {
        graph.wire_uses[wire].len()
    }

    fn is_wire_single_use(graph: &GateGraph, wire: Wire) -> bool {
        graph.wire_uses[wire].len() == 1
    }

    fn get_wire_single_use(graph: &GateGraph, wire: Wire) -> Option<Gate> {
        let uses = &graph.wire_uses[wire];
        if uses.len() == 1 {
            uses.iter().next().copied()
        } else {
            None
        }
    }

    fn get_gate_inputs(graph: &GateGraph, gate: Gate) -> Vec<Wire> {
        let gate_data = &graph.gates[gate];
        let gate_param = gate_data.gate_param();

        let mut inputs = Vec::new();
        inputs.extend_from_slice(gate_param.constants);
        inputs.extend_from_slice(gate_param.inputs);
        inputs
    }

    fn get_gate_outputs(graph: &GateGraph, gate: Gate) -> Vec<Wire> {
        let gate_data = &graph.gates[gate];
        let gate_param = gate_data.gate_param();

        let mut outputs = Vec::new();
        outputs.extend_from_slice(gate_param.outputs);
        outputs
    }

    #[test]
    fn test_use_def_analysis() {
        let mut graph = GateGraph::new();
        let root = graph.path_spec_tree.root();

        // Create some wires
        let in1 = graph.add_inout();
        let in2 = graph.add_inout();
        let out1 = graph.add_witness();
        let out2 = graph.add_witness();

        // Create a gate that uses in1 and in2, produces out1
        let gate1 = graph.emit_gate(root, Opcode::Bxor, vec![in1, in2], vec![out1]);

        // Create another gate that uses out1 and in1, produces out2
        let gate2 = graph.emit_gate(root, Opcode::Band, vec![out1, in1], vec![out2]);

        // Build use-def chains
        graph.rebuild_use_def_chains();

        // Check that gate1 defines out1
        assert_eq!(get_wire_def(&graph, out1), Some(gate1));

        // Check that gate2 defines out2
        assert_eq!(get_wire_def(&graph, out2), Some(gate2));

        // Check that in1 and in2 are used by gate1
        assert!(graph.get_wire_uses(in1).contains(&gate1));
        assert!(graph.get_wire_uses(in2).contains(&gate1));

        // Check that out1 is used by gate2
        assert!(graph.get_wire_uses(out1).contains(&gate2));

        // Check wire use counts
        assert_eq!(wire_use_count(&graph, in1), 2); // Used by gate1 and gate2
        assert_eq!(wire_use_count(&graph, in2), 1);
        assert_eq!(wire_use_count(&graph, out1), 1);
        assert_eq!(wire_use_count(&graph, out2), 0);

        // Check single use queries
        assert!(!is_wire_single_use(&graph, in1)); // Used twice
        assert!(is_wire_single_use(&graph, in2));
        assert!(is_wire_single_use(&graph, out1));
        assert!(!is_wire_single_use(&graph, out2)); // No uses

        // Check get_wire_single_use
        assert_eq!(get_wire_single_use(&graph, in1), None); // Used twice
        assert_eq!(get_wire_single_use(&graph, out1), Some(gate2));
        assert_eq!(get_wire_single_use(&graph, out2), None); // No uses
    }

    #[test]
    fn test_constant_use_def() {
        let mut graph = GateGraph::new();
        let root = graph.path_spec_tree.root();

        // Create a constant wire
        let const_wire = graph.add_constant(Word(42u64));
        let in_wire = graph.add_inout();
        let out = graph.add_witness();

        // Create a gate that uses the constant and input wire
        let gate = graph.emit_gate(root, Opcode::Bxor, vec![const_wire, in_wire], vec![out]);

        // Build use-def chains
        graph.rebuild_use_def_chains();

        // Constants are not defined by gates
        assert_eq!(get_wire_def(&graph, const_wire), None);

        // But they should be tracked as used
        assert!(graph.get_wire_uses(const_wire).contains(&gate));
        assert_eq!(wire_use_count(&graph, const_wire), 1);
    }

    #[test]
    fn test_rebuild_use_def_chains() {
        let mut graph = GateGraph::new();
        let root = graph.path_spec_tree.root();

        // Create wires and gates
        let in1 = graph.add_inout();
        let in2 = graph.add_inout();
        let out = graph.add_witness();

        graph.emit_gate(root, Opcode::Bxor, vec![in1, in2], vec![out]);

        // Clear use-def info manually (simulating corruption)
        graph.wire_def.clear();
        graph.wire_uses.clear();

        // Verify it's cleared
        assert_eq!(get_wire_def(&graph, out), None);
        assert!(graph.get_wire_uses(in1).is_empty());

        // Rebuild
        graph.rebuild_use_def_chains();

        // Verify it's restored
        assert!(get_wire_def(&graph, out).is_some());
        assert!(!graph.get_wire_uses(in1).is_empty());
        assert!(!graph.get_wire_uses(in2).is_empty());
    }

    #[test]
    fn test_gate_inputs_outputs() {
        let mut graph = GateGraph::new();
        let root = graph.path_spec_tree.root();

        let in1 = graph.add_inout();
        let in2 = graph.add_inout();
        let out = graph.add_witness();

        let gate = graph.emit_gate(root, Opcode::Bxor, vec![in1, in2], vec![out]);

        // No need to rebuild use-def chains for this test
        // as we're just checking the gate structure

        let inputs = get_gate_inputs(&graph, gate);
        // Bxor has 1 constant input (ALL_ONE) + 2 regular inputs
        assert_eq!(inputs.len(), 3);
        assert!(inputs.contains(&in1));
        assert!(inputs.contains(&in2));
        // First input should be the constant wire
        let const_wire = inputs[0];
        match graph.wires[const_wire].kind {
            WireKind::Constant(word) => assert_eq!(word, Word::ALL_ONE),
            _ => panic!("Expected constant wire"),
        }

        let outputs = get_gate_outputs(&graph, gate);
        assert_eq!(outputs.len(), 1);
        assert!(outputs.contains(&out));
    }
}
