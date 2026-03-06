use std::collections::HashMap;

use arith::Field;
use gkr_engine::FieldEngine;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum CoefType {
    #[default]
    Constant,
    Random,
    PublicInput(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GateKind {
    Mul,
    Add,
    Const,
    Uni,
}

#[derive(Debug, Clone, Copy)]
pub struct RndCoefIdx {
    pub layer: usize,
    pub kind: GateKind,
    pub gate: usize,
}

#[derive(Debug, Clone)]
pub struct Gate<C: FieldEngine, const INPUT_NUM: usize> {
    pub i_ids: [usize; INPUT_NUM],
    pub o_id: usize,
    pub coef_type: CoefType,
    pub coef: C::CircuitField,
    pub gate_type: usize,
}

pub type GateMul<C> = Gate<C, 2>;
pub type GateAdd<C> = Gate<C, 1>;
pub type GateUni<C> = Gate<C, 1>;
pub type GateConst<C> = Gate<C, 0>;

impl<C: FieldEngine, const INPUT_NUM: usize> Copy for Gate<C, INPUT_NUM> {}

pub struct RndCoefMap<F: Field> {
    pub map: HashMap<(usize, GateKind, usize), F>,
}

impl<F: Field> RndCoefMap<F> {
    #[inline]
    pub fn get(&self, layer: usize, kind: GateKind, gate: usize) -> Option<F> {
        self.map.get(&(layer, kind, gate)).copied()
    }
}
