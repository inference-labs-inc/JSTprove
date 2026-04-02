// Copyright 2025 Irreducible Inc.
use std::collections::BTreeMap;

use binius_core::{
    constraint_system::{
        AndConstraint, ConstraintSystem, MulConstraint, Operand, ShiftedValueIndex, ValueIndex,
        ValueVecLayout,
    },
    word::Word,
};

use crate::ast::{Constraint, ConstraintSet, OperandExpr, ShiftOp, Term};

struct Assignment {
    value_index: Option<ValueIndex>,
}

impl Assignment {
    pub fn no_assignment() -> Self {
        Self { value_index: None }
    }
}

pub struct Context {
    ids: BTreeMap<String, Assignment>,
    literals: BTreeMap<u64, Assignment>,
    value_vec_layout: Option<ValueVecLayout>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            ids: BTreeMap::new(),
            literals: BTreeMap::new(),
            value_vec_layout: None,
        }
    }

    pub fn preprocess(&mut self, cs: &ConstraintSet) {
        for constraint in &cs.constraints {
            match constraint {
                Constraint::And { a, b, c } => {
                    self.preprocess_operand(a);
                    self.preprocess_operand(b);
                    self.preprocess_operand(c);
                }
                Constraint::Mul { a, b, hi, lo } => {
                    self.preprocess_operand(a);
                    self.preprocess_operand(b);
                    self.preprocess_operand(hi);
                    self.preprocess_operand(lo);
                }
            }
        }
    }

    fn preprocess_operand(&mut self, operand: &OperandExpr) {
        match operand {
            OperandExpr::Term(term) => self.preprocess_term(term),
            OperandExpr::Xor(xor_expr) => {
                for term in &xor_expr.terms {
                    self.preprocess_term(term);
                }
            }
        }
    }

    fn preprocess_term(&mut self, term: &Term) {
        match term {
            Term::Literal(val) => {
                self.literals.insert(*val, Assignment::no_assignment());
            }
            Term::Wire(wire) | Term::Shifted { wire, .. } => {
                self.ids.insert(wire.clone(), Assignment::no_assignment());
            }
        }
    }

    pub fn perform_witness_assignment(&mut self) {
        let n_const = self.literals.len();
        let n_witness = self.ids.len();

        let mut cur_index = 0;
        for (_, assignment) in self.literals.iter_mut() {
            assignment.value_index = Some(ValueIndex(cur_index));
            cur_index += 1;
        }
        let offset_inout = cur_index as usize;
        cur_index = cur_index.next_power_of_two();
        let offset_witness = cur_index as usize;
        for (_, assignment) in self.ids.iter_mut() {
            assignment.value_index = Some(ValueIndex(cur_index));
            cur_index += 1;
        }
        cur_index = cur_index.next_power_of_two();
        let committed_total_len = cur_index as usize;

        let value_vec_layout = ValueVecLayout {
            n_const,
            n_inout: 0,
            n_witness,
            n_internal: 0,
            offset_inout,
            offset_witness,
            committed_total_len,
            n_scratch: 0,
        };
        // For ceck, we only need basic validation, not full protocol compliance
        assert!(
            value_vec_layout.committed_total_len.is_power_of_two(),
            "total length must be a power-of-two"
        );
        assert!(
            value_vec_layout.offset_witness.is_power_of_two(),
            "witness parameters must start at a power-of-two offset",
        );
        self.value_vec_layout = Some(value_vec_layout);
    }

    pub fn build(&self, cs: &ConstraintSet) -> ConstraintSystem {
        let mut and_constraints = Vec::new();
        let mut mul_constraints = Vec::new();

        for constraint in &cs.constraints {
            match constraint {
                Constraint::And { a, b, c } => {
                    let and_constraint = AndConstraint {
                        a: self.convert_operand(a),
                        b: self.convert_operand(b),
                        c: self.convert_operand(c),
                    };
                    and_constraints.push(and_constraint);
                }
                Constraint::Mul { a, b, hi, lo } => {
                    let mul_constraint = MulConstraint {
                        a: self.convert_operand(a),
                        b: self.convert_operand(b),
                        hi: self.convert_operand(hi),
                        lo: self.convert_operand(lo),
                    };
                    mul_constraints.push(mul_constraint);
                }
            }
        }

        let constants = self.literals.keys().map(|&lit| Word(lit)).collect();

        ConstraintSystem::new(
            constants,
            self.value_vec_layout.clone().unwrap(),
            and_constraints,
            mul_constraints,
        )
    }

    fn convert_term(&self, term: &Term) -> ShiftedValueIndex {
        match term {
            Term::Literal(val) => ShiftedValueIndex::plain(self.lookup_literal_assignment(*val)),
            Term::Wire(id) => ShiftedValueIndex::plain(self.lookup_id_assignment(id)),
            Term::Shifted { wire, op, amount } => {
                let index = self.lookup_id_assignment(wire);
                match op {
                    ShiftOp::Sll => ShiftedValueIndex::sll(index, *amount),
                    ShiftOp::Slr => ShiftedValueIndex::srl(index, *amount),
                    ShiftOp::Sar => ShiftedValueIndex::sar(index, *amount),
                    ShiftOp::Ror => ShiftedValueIndex::rotr(index, *amount),
                }
            }
        }
    }

    fn convert_operand(&self, operand: &OperandExpr) -> Operand {
        match operand {
            OperandExpr::Term(term) => vec![self.convert_term(term)],
            OperandExpr::Xor(xor_expr) => xor_expr
                .terms
                .iter()
                .map(|term| self.convert_term(term))
                .collect(),
        }
    }

    fn lookup_literal_assignment(&self, lit: u64) -> ValueIndex {
        let Some(Assignment { value_index }) = self.literals.get(&lit) else {
            panic!("literal {lit} hasn't been processed")
        };
        let Some(value_index) = *value_index else {
            panic!("value_index has not been assigned for literal {lit}")
        };
        value_index
    }

    fn lookup_id_assignment(&self, id: &str) -> ValueIndex {
        let Some(Assignment { value_index }) = self.ids.get(id) else {
            panic!("{id} hasn't been processed")
        };
        let Some(value_index) = *value_index else {
            panic!("value_index has not been assigned for {id}")
        };
        value_index
    }
}
