// Copyright 2025 Irreducible Inc.
use std::fmt;

use crate::compiler::constraint_builder::ConstraintBuilder;

#[derive(Default)]
pub struct Stat {
    /// The initial number of linear defs.
    pre_linear_def: usize,
    /// The initial number of and constraints.
    pre_and_constraints: usize,
    /// The number of linear defs we realized we must commit.
    committed_linear: usize,
    /// Committed due to reaching the maximum depth.
    committed_linear_depth: usize,
    /// The number of linears visited in legraph.
    legraph_visited: usize,
}

impl fmt::Debug for Stat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stat")
            .field("pre_linear_def", &self.pre_linear_def)
            .field("pre_and_constraints", &self.pre_and_constraints)
            .field("committed_linear", &self.committed_linear)
            .field("committed_linear_depth", &self.committed_linear_depth)
            .field("legraph_visited", &self.legraph_visited)
            .finish()
    }
}

impl Stat {
    pub fn new(cb: &ConstraintBuilder) -> Self {
        Self {
            pre_linear_def: cb.linear_constraints.len(),
            pre_and_constraints: cb.and_constraints.len(),
            committed_linear: 0,
            committed_linear_depth: 0,
            legraph_visited: 0,
        }
    }

    pub fn note_committed(&mut self) {
        self.committed_linear += 1;
    }

    pub fn note_visited(&mut self) {
        self.legraph_visited += 1;
    }

    pub fn note_committed_linear_depth(&mut self) {
        self.committed_linear_depth += 1;
    }
}
