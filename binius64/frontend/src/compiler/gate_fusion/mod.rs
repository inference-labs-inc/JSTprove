// Copyright 2025 Irreducible Inc.
//! Gate fusion optimization.
//!
//! The main cost of our system is coming from the number of AND constraints. The less we have the
//! cheaper it is.
//!
//! Our AND constraints are powerful construct construct. They can handle a single AND of two XOR
//! combinations where each of the values could be shifted.
//!
//! `ConstraintBuilder` which this pass operates consists of AND, MUL and linear constraints. Linear
//! constraints are basically are constraints that define a single wire using a XOR combination
//! and/or shifts. Since our system does not suppose standalone linear combinations they will have
//! to be promoted to AND constraints.
//!
//! BUT we have a chance of avoiding that if we manage to inline that wire into every consumer
//! constraint which means we don't have to commit that value and thus we don't need an AND
//! constraint!

use cranelift_entity::EntitySet;
use legraph::LeGraph;

use crate::compiler::{Wire, constraint_builder::ConstraintBuilder};

mod commit_set;
mod legraph;
mod patch;

mod stat;
#[cfg(test)]
mod tests;

use stat::Stat;

pub fn run_pass(cb: &mut ConstraintBuilder, pinned_wires: &EntitySet<Wire>, all_one: Wire) {
    let mut stat = Stat::new(cb);

    let mut leg = LeGraph::new(cb, &mut stat);
    commit_set::run_decide_commit_set(&mut leg, &mut stat);
    leg.lin_committed.extend(pinned_wires.iter());
    let patches = patch::build(cb, &leg, all_one);
    patch::apply_patches(cb, patches);
}
