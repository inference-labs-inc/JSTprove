// Copyright 2025 Irreducible Inc.
//! linear expression graph.

use cranelift_entity::EntitySet;
use petgraph::{
    dot::Dot,
    graph::{DiGraph, NodeIndex},
};
use rustc_hash::FxHashMap;

use super::Stat;
use crate::compiler::{
    Wire,
    constraint_builder::{ConstraintBuilder, Shift, WireOperand},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstraintRef {
    And { index: usize },
    Mul { index: usize },
    Linear { index: usize },
}

/// Represents the different types of nodes in the Linear Expression Graph.
#[derive(Debug)]
pub enum NodeData {
    /// Root node - represents a use site in a non-linear constraint.
    ///
    /// These nodes are the "sinks" of the graph where linear expressions flow into
    /// non-linear constraints (AND/MUL). They have no outgoing edges and represent
    /// the termination points for inlining decisions.
    ///
    /// # Example
    /// ```text
    /// y = x ^ a       // Linear definition
    /// z = y & b       // AND constraint creates a Root node for y's use
    /// ```
    Root {
        /// Reference to the non-linear constraint using the linear definition.
        constraint: ConstraintRef,
    },

    /// Linear definition node - represents a linear constraint.
    ///
    /// These nodes define a wire as an XOR combination of shifted values.
    /// They are candidates for inlining into their consumers.
    ///
    /// # Example
    /// ```text
    /// y = x ^ (a << 5) ^ (b >> 3)  // Creates a LinDef node for y
    /// ```
    LinDef {
        /// The wire being defined by this linear constraint.
        dst: Wire,
        /// The XOR combination of shifted values that defines dst.
        operand: WireOperand,
        /// Index in the linear constraint list.
        index: usize,
    },

    /// Opaque node - represents a wire not defined by a linear constraint.
    ///
    /// These are wires that come from inputs, constants, or outputs of non-linear
    /// operations. They cannot be inlined and serve as terminal nodes in the
    /// inlining traversal.
    ///
    /// # Example
    /// ```text
    /// x = input()     // Creates an Opaque node for x
    /// y = a * b       // Creates Opaque nodes for y (MUL output)
    /// ```
    Opaque {
        /// The opaque wire that cannot be inlined.
        wire: Wire,
    },
}

impl NodeData {
    fn is_root(&self) -> bool {
        matches!(self, NodeData::Root { .. })
    }

    fn is_opaque(&self) -> bool {
        matches!(self, NodeData::Opaque { .. })
    }
}

/// Data associated with edges in the Linear Expression Graph.
///
/// Each edge represents a use of a wire (producer) by another constraint (consumer),
/// annotated with the shift operation applied to the producer value.
///
/// # Example
/// ```text
/// y = x << 5      // Edge from x to y has shift = Sll(5)
/// z = y ^ a       // Edge from y to z has shift = None
/// w = z >> 3      // Edge from z to w has shift = Srl(3)
/// ```
#[derive(Debug)]
pub struct EdgeData {
    /// The shift operation applied when the producer is used by the consumer.
    pub shift: Shift,
}

/// Linear Expression Graph (LeGraph) - the core data structure for gate fusion optimization.
///
/// This graph represents the data flow relationships between linear constraints (XOR and shift
/// operations) and their uses in non-linear constraints (AND/MUL operations). The graph is used
/// to determine which linear constraints can be inlined into their consumers to reduce the total
/// number of AND constraints in the final circuit.
///
/// # Graph Structure
///
/// The graph consists of three types of nodes:
///
/// 1. **Linear Definition Nodes** (`LinDef`): Represent linear constraints that define a wire as an
///    XOR combination of shifted values. These are candidates for inlining.
///
/// 2. **Root Nodes**: Represent uses of linear definitions in non-linear constraints (AND/MUL).
///    These are the sinks of the graph where inlining decisions terminate.
///
/// 3. **Opaque Nodes**: Represent wires that are not defined by linear constraints (e.g., inputs or
///    outputs of non-linear operations). These cannot be inlined.
///
/// Edges in the graph flow from producers to consumers, with each edge annotated with a shift
/// operation that describes how the producer value is transformed when used by the consumer.
///
/// # Example
///
/// ```text
/// // Circuit:
/// y = a ^ b        // Linear definition
/// z = y >> 5       // Linear definition using y
/// w = z & c        // Non-linear use of z
///
/// // Graph representation:
/// [a] ──┐
///       ├─> [y = a ^ b] ──srl(5)──> [z = y >> 5] ──none──> [AND root]
/// [b] ──┘
/// ```
///
/// In this example, both `y` and `z` can potentially be inlined into the AND constraint,
/// resulting in `w = ((a ^ b) >> 5) & c` without intermediate wire commitments.
pub struct LeGraph {
    pub pg: DiGraph<NodeData, EdgeData>,
    pub wire_to_node: FxHashMap<Wire, NodeIndex>,
    pub lin_def: EntitySet<Wire>,
    pub lin_committed: EntitySet<Wire>,
    pub roots: Vec<NodeIndex>,
    pub opaque: Vec<NodeIndex>,
}

impl LeGraph {
    /// Constructs a new Linear Expression Graph from the constraint builder.
    ///
    /// This method analyzes all constraints in the builder and constructs a graph that captures
    /// the use-def relationships between linear and non-linear constraints.
    ///
    /// # Process
    ///
    /// 1. Identifies all linear constraint definitions
    /// 2. Tracks uses of linear definitions in other linear constraints
    /// 3. Identifies "root" uses where linear definitions flow into non-linear constraints
    /// 4. Builds edges with appropriate shift annotations
    pub fn new(cb: &ConstraintBuilder, stat: &mut Stat) -> Self {
        let mut leg = Self {
            pg: DiGraph::new(),
            wire_to_node: FxHashMap::default(),
            lin_committed: EntitySet::new(),
            lin_def: EntitySet::new(),
            roots: Vec::new(),
            opaque: Vec::new(),
        };
        build_use_def(cb, &mut leg, stat);
        leg
    }

    pub(super) fn is_root(&self, node: NodeIndex) -> bool {
        self.pg.node_weight(node).unwrap().is_root()
    }

    pub(super) fn is_opaque(&self, node: NodeIndex) -> bool {
        self.pg.node_weight(node).unwrap().is_opaque()
    }

    /// Returns the set of wires that must be committed (converted to AND constraints).
    ///
    /// This is populated only after running the commit_set decision pass.
    pub fn commit_set(&self) -> &EntitySet<Wire> {
        &self.lin_committed
    }

    /// Returns the operand (RHS) of a linear definition for the given wire.
    ///
    /// # Panics
    ///
    /// Panics if the wire is not defined by a linear constraint.
    pub fn lin_def(&self, wire: Wire) -> &WireOperand {
        let node = self.wire_to_node[&wire];
        match self.pg.node_weight(node).unwrap() {
            NodeData::LinDef { operand, .. } => operand,
            _ => panic!("supposed to be a linear def"),
        }
    }

    pub fn lin_dst(&self, node: NodeIndex) -> Wire {
        match self.pg.node_weight(node).unwrap() {
            NodeData::LinDef { dst, .. } => *dst,
            _ => panic!("supposed to be a linear assignment"),
        }
    }

    pub fn lin_def_constraint_ref(&self, wire: Wire) -> ConstraintRef {
        let node = self.wire_to_node[&wire];
        match self.pg.node_weight(node).unwrap() {
            NodeData::LinDef { index, .. } => ConstraintRef::Linear { index: *index },
            _ => panic!("supposed to be a linear def"),
        }
    }

    pub fn root_constraint_ref(&self, node: NodeIndex) -> ConstraintRef {
        match self.pg.node_weight(node).unwrap() {
            NodeData::Root { constraint } => *constraint,
            _ => panic!("supposed to be a root"),
        }
    }

    /// Checks if a wire is defined by a linear constraint.
    ///
    /// Returns `true` if the wire is the output of a linear constraint (XOR combination
    /// of shifted values), `false` if it's an opaque wire or not in the graph at all.
    pub fn is_lin_def(&self, wire: Wire) -> bool {
        self.lin_def.contains(wire)
    }

    /// Add a linear definition node to the graph.
    ///
    /// Creates a new node representing a linear constraint that defines `dst` as
    /// the XOR combination specified by `operand`.
    fn add_lin_def(&mut self, dst: Wire, operand: WireOperand, index: usize) {
        let lin_node = self.pg.add_node(NodeData::LinDef {
            dst,
            operand,
            index,
        });
        let prev = self.wire_to_node.insert(dst, lin_node);
        assert!(prev.is_none(), "wire already has a node");
        self.lin_def.insert(dst);
    }

    /// Notes a use of a wire by a linear user.
    ///
    /// `shift` is how much the producer is shifted by the consumer expression.
    ///
    /// Note:
    ///
    /// 1. directionality matters, the value flows from the producer into the consumer.
    /// 2. a single consumer possibly can refer the same producer multiple times. In that case there
    ///    are going to be multiple edges.
    fn note_lin_use(&mut self, producer: Wire, shift: Shift, consumer: Wire) {
        let node_c = self.wire_to_node[&consumer];
        if self.is_lin_def(producer) {
            let node_p = self.wire_to_node[&producer];
            self.pg.add_edge(node_p, node_c, EdgeData { shift });
        } else {
            // This is a use of a wire that is not defined by a linear. That means it's opaque!
            let opaque_node = *self.wire_to_node.entry(producer).or_insert_with(|| {
                let t = self.pg.add_node(NodeData::Opaque { wire: producer });
                self.opaque.push(t);
                t
            });
            self.pg.add_edge(opaque_node, node_c, EdgeData { shift });
        }
    }

    /// Notes a use of a wire of a linear producer by a non-linear user.
    fn note_nonlinear_use(&mut self, producer: Wire, shift: Shift, constraint: ConstraintRef) {
        let node_p = self.wire_to_node[&producer];
        let root_node = self.pg.add_node(NodeData::Root { constraint });
        self.roots.push(root_node);
        self.pg.add_edge(node_p, root_node, EdgeData { shift });
    }

    /// A pure diagnostic tool which can come in handy if you are in trouble.
    #[doc(hidden)]
    #[allow(dead_code)]
    pub fn render_graphviz(&self) -> String {
        let fmt_edge =
            |_g: &petgraph::Graph<NodeData, EdgeData>,
             _edge: petgraph::graph::EdgeReference<'_, EdgeData>| { String::new() };
        let fmt_node = |_g: &petgraph::Graph<NodeData, EdgeData>,
                        (_node, data): (NodeIndex, &NodeData)| {
            match data {
                NodeData::Root { .. } => {
                    let label = "root".to_string();
                    format!("shape=box color=red label=\"{label}\"")
                }
                NodeData::LinDef { dst, operand, .. } => {
                    let fmt_operand = if operand.len() == 1 {
                        format_shifted_wire(&operand[0])
                    } else {
                        operand
                            .iter()
                            .map(format_shifted_wire)
                            .collect::<Vec<String>>()
                            .join("^")
                    };

                    let label = format!("{:?} = {fmt_operand}", dst);
                    let shape = if self.lin_committed.contains(*dst) {
                        "shape=octogon"
                    } else {
                        "shape=box"
                    };
                    format!("{shape} label=\"{label}\"")
                }
                NodeData::Opaque { wire } => {
                    let label = format!("opaque: {:?}", wire);
                    format!("shape=box color=blue label=\"{label}\"")
                }
            }
        };
        let dot = Dot::with_attr_getters(&self.pg, &[], &fmt_edge, &fmt_node);
        return format!("{:?}", dot);

        fn format_shifted_wire(t: &crate::compiler::constraint_builder::ShiftedWire) -> String {
            match t.shift {
                Shift::None => format!("{:?}", t.wire),
                Shift::Sll(amount) => format!("{:?} << {:?}", t.wire, amount),
                Shift::Sll32(amount) => format!("{:?} <<32 {:?}", t.wire, amount),
                Shift::Srl(amount) => format!("{:?} >> {:?}", t.wire, amount),
                Shift::Srl32(amount) => format!("{:?} >>32 {:?}", t.wire, amount),
                Shift::Sar(amount) => format!("{:?} sar {:?}", t.wire, amount),
                Shift::Sra32(amount) => format!("{:?} sar32 {:?}", t.wire, amount),
                Shift::Rotr(amount) => format!("{:?} rotr {:?}", t.wire, amount),
                Shift::Rotr32(amount) => format!("{:?} rotr32 {:?}", t.wire, amount),
            }
        }
    }
}

fn build_use_def(cb: &ConstraintBuilder, leg: &mut LeGraph, _stat: &mut Stat) {
    // Collect defs from linear constraints.
    //
    // Linear constraints are simple definitions. We assert that this is the case here.
    // In future we should actually define `linear_constraints`.
    for (index, lin) in cb.linear_constraints.iter().enumerate() {
        leg.add_lin_def(lin.dst, lin.rhs.clone(), index);
    }

    for lin in &cb.linear_constraints {
        let consumer = lin.dst;
        for term in &lin.rhs {
            leg.note_lin_use(term.wire, term.shift, consumer);
        }
    }

    for (index, and) in cb.and_constraints.iter().enumerate() {
        harvest_nonlin_uses(&and.a, leg, ConstraintRef::And { index });
        harvest_nonlin_uses(&and.b, leg, ConstraintRef::And { index });
        harvest_nonlin_uses(&and.c, leg, ConstraintRef::And { index });
    }

    for (index, mul) in cb.mul_constraints.iter().enumerate() {
        harvest_nonlin_uses(&mul.a, leg, ConstraintRef::Mul { index });
        harvest_nonlin_uses(&mul.b, leg, ConstraintRef::Mul { index });
        harvest_nonlin_uses(&mul.hi, leg, ConstraintRef::Mul { index });
        harvest_nonlin_uses(&mul.lo, leg, ConstraintRef::Mul { index });
    }
}

fn harvest_nonlin_uses(operand: &WireOperand, leg: &mut LeGraph, constraint: ConstraintRef) {
    for term in operand {
        if leg.is_lin_def(term.wire) {
            leg.note_nonlinear_use(term.wire, term.shift, constraint);
        }
    }
}
