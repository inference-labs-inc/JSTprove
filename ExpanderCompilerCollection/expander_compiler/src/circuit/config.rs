use std::{fmt::Debug, hash::Hash};

use crate::field::Field;

pub trait Config: Default + Clone + Ord + Debug + Hash + Copy + 'static {
    type CircuitField: Field;

    const CONFIG_ID: usize;

    const COST_INPUT: usize = 1000;
    const COST_VARIABLE: usize = 100;
    const COST_MUL: usize = 10;
    const COST_ADD: usize = 3;
    const COST_CONST: usize = 3;

    const ENABLE_RANDOM_COMBINATION: bool = true;
}

#[derive(Default, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct M31Config {}

impl Config for M31Config {
    type CircuitField = crate::field::M31;

    const CONFIG_ID: usize = 1;
}

#[derive(Default, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct BN254Config {}

impl Config for BN254Config {
    type CircuitField = crate::field::BN254;

    const CONFIG_ID: usize = 2;
}

#[derive(Default, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct GF2Config {}

impl Config for GF2Config {
    type CircuitField = crate::field::GF2;

    const CONFIG_ID: usize = 3;

    // temporary fix for Keccak_GF2
    // TODO: measure actual costs
    const COST_MUL: usize = 200;

    const ENABLE_RANDOM_COMBINATION: bool = false;
}
