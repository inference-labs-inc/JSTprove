// Copyright 2025 Irreducible Inc.
use std::iter;

use petgraph::{
    Direction,
    graph::EdgeIndex,
    visit::{DfsPostOrder, EdgeRef},
};
use rustc_hash::FxHashMap;

use super::{LeGraph, Stat};
use crate::compiler::constraint_builder::Shift;

pub const MAX_DEPTH: usize = 6;

struct CommitSetCx {
    /// Every shift type that was used on the path to reach this node.
    set: Vec<Shift>,
    /// Number of nodes we should visit from the current node to get back to one of the roots (or
    /// committed linear expression)
    ///
    /// This is used as a proxy to estimate the impact of inlining.
    depth: usize,
}

impl CommitSetCx {
    /// Create a new context for an edge with depth 0.
    fn new(seed_shift: Shift) -> Self {
        let mut set = Vec::with_capacity(8);
        set.push(seed_shift);
        Self { set, depth: 0 }
    }

    /// Returns if every shift is composable with the given one.
    fn composable(&self, shift: Shift) -> bool {
        self.set.iter().all(|s| Shift::compose(*s, shift).is_some())
    }

    /// Merge multiple contexts into a single one.
    fn join<'a>(iter: impl Iterator<Item = &'a CommitSetCx>) -> Self {
        let mut set = Vec::with_capacity(8);
        let mut depth = 0;
        for cx in iter {
            set.extend(cx.set.iter().copied());
            depth = depth.max(cx.depth);
        }
        set.sort_unstable();
        set.dedup();
        Self { set, depth }
    }

    /// Create a new context by adding a new shift and incrementing depth.
    fn add(&self, out_shift: Shift) -> CommitSetCx {
        let mut set = self
            .set
            .iter()
            .copied()
            .chain(iter::once(out_shift))
            .collect::<Vec<_>>();
        set.sort_unstable();
        set.dedup();
        Self {
            set,
            depth: self.depth + 1,
        }
    }
}

/// Traverse the linear expression graph and decide which linear expressions to commit.
///
/// There are two cases where we might commit a linear expression:
///
/// 1. When inlining a linear expression is not possible because it does not fit into a single AND
///    constraint. For example, an expression that uses a shift right operator cannot be inlined
///    into a user that uses shift left operator.
///
/// 2. Inlining is prone to term explosion. To prevent that we avoid inlining expressions that lie
///    past a certain depth.
///
/// Note that this is all-or-nothing decision: if at least one user cannot inline an expression
/// then no users should inline it.
pub fn run_decide_commit_set(leg: &mut LeGraph, stat: &mut Stat) {
    let mut per_edge: FxHashMap<EdgeIndex, CommitSetCx> = FxHashMap::default();
    per_edge.reserve(leg.pg.edge_count());

    // Iterate the graph in the postorder. That is, we iterate the producers before their consumers.
    // IOW, when visiting a node all of its children have been already visited.
    //
    // Remember that Linear Expression Graph (legraph) is a directed graph where edges point towards
    // the consumers. We propagate information along the edges from the consumers up to the
    // producers.
    //
    // We seed iteration from the "sources" of graph. A source is a node with no incoming edges and
    // those are the opaque wires in our legraph. However, this is a postorder iteration and that
    // means that we start processing at the "sinks", ie. the first node to be popped out from
    // `next` is a sink. A sink is a node that does not have any outgoing edges. In legraph
    // sinks are our roots, ie. non-linear constraints.
    //
    // The information is captured by the `CommitSetCx` which represent the relevant data for the
    // inlining process.
    //
    // With all of that, what we are doing is examining every linear expression node and see if
    // every user's shifts compose with the current node shifts which are stored in the incoming
    // edges and additionally the node does not lie too deep in the graph for any of the users.
    let mut postorder = DfsPostOrder::empty(&leg.pg);
    for source in &leg.opaque {
        postorder.move_to(*source);
        while let Some(node) = postorder.next(&leg.pg) {
            if leg.is_root(node) {
                // Special handling for the root nodes.
                //
                // Just create a new context for each root node with the seed shift.
                for in_edge in leg.pg.edges_directed(node, Direction::Incoming) {
                    let shift = in_edge.weight().shift;
                    per_edge.insert(in_edge.id(), CommitSetCx::new(shift));
                }
                continue;
            }
            if leg.is_opaque(node) {
                // Special handling for opaque nodes, or lack of there of.
                continue;
            }

            // Must be a linear definition then.
            //
            // Check whether the incoming edges are composing with every outcoming edges.
            let lin_def_wire = leg.lin_dst(node);
            let incoming = leg.pg.edges_directed(node, Direction::Incoming);
            let outcoming = leg.pg.edges_directed(node, Direction::Outgoing);

            let mut composable = true;
            let mut depth = 0;

            'out: for out_edge in outcoming.clone() {
                let out_edge_cx = &per_edge[&out_edge.id()];
                depth = out_edge_cx.depth.max(depth);
                for in_edge in incoming.clone() {
                    let in_shift = in_edge.weight().shift;
                    if !out_edge_cx.composable(in_shift) {
                        composable = false;
                        break 'out;
                    }
                }
            }

            if depth > MAX_DEPTH || !composable {
                // Decision: commit.
                //
                // Every incoming edge context is going to be a brand new one seeded with the
                // current shift.
                for in_edge in incoming {
                    let in_shift = in_edge.weight().shift;
                    per_edge.insert(in_edge.id(), CommitSetCx::new(in_shift));
                }

                // Insert into the committed set verifying that this wire was not inserted before.
                assert!(leg.lin_committed.insert(lin_def_wire));

                stat.note_committed();
                if depth > MAX_DEPTH {
                    stat.note_committed_linear_depth();
                }
            } else {
                // Decision: inline.
                //
                // This node will beget a new context by joining outcoming contexts. Then every
                // incoming edge will get combined with the outcoming shift type.
                //
                // TODO: note that we've already visited every child, so we could free up memory
                // required for their context.
                let join_cx = CommitSetCx::join(outcoming.map(|edge| &per_edge[&edge.id()]));
                for in_edge in incoming {
                    let in_shift = in_edge.weight().shift;
                    per_edge.insert(in_edge.id(), join_cx.add(in_shift));
                }
            }

            stat.note_visited();
        }
    }
}
