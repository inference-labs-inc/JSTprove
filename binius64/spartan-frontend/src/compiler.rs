// Copyright 2025 Irreducible Inc.

use crate::{
    circuit_builder::ConstraintBuilder,
    constraint_system::{ConstraintSystem, WitnessLayout},
    wire_elimination::{CostModel, run_wire_elimination},
};

pub fn compile(builder: ConstraintBuilder) -> (ConstraintSystem, WitnessLayout) {
    let ir = builder.build();
    let ir = run_wire_elimination(CostModel::default(), ir);
    ir.finalize()
}
