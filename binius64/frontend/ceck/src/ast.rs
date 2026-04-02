// Copyright 2025 Irreducible Inc.
use binius_core::constraint_system::ShiftVariant;

#[derive(Debug, Clone, PartialEq)]
pub enum ShiftOp {
    Sll,
    Slr,
    Sar,
    Ror,
}

impl ShiftOp {
    #[allow(dead_code)]
    pub fn to_shift_variant(&self) -> ShiftVariant {
        match self {
            ShiftOp::Sll => ShiftVariant::Sll,
            ShiftOp::Slr => ShiftVariant::Slr,
            ShiftOp::Sar => ShiftVariant::Sar,
            ShiftOp::Ror => ShiftVariant::Rotr,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Term {
    Literal(u64),
    Wire(String),
    Shifted {
        wire: String,
        op: ShiftOp,
        amount: usize,
    },
}

#[derive(Debug, Clone)]
pub struct XorExpr {
    pub terms: Vec<Term>,
}

#[derive(Debug, Clone)]
pub enum OperandExpr {
    Xor(XorExpr),
    Term(Term),
}

#[derive(Debug, Clone)]
pub enum Constraint {
    And {
        a: OperandExpr,
        b: OperandExpr,
        c: OperandExpr,
    },
    Mul {
        a: OperandExpr,
        b: OperandExpr,
        hi: OperandExpr,
        lo: OperandExpr,
    },
}

#[derive(Debug, Clone)]
pub struct ConstraintSet {
    pub constraints: Vec<Constraint>,
}

#[derive(Debug, Clone)]
pub struct AssertEqv {
    pub lhs: ConstraintSet,
    pub rhs: ConstraintSet,
}

#[derive(Debug, Clone)]
pub struct AssertNotEqv {
    pub lhs: ConstraintSet,
    pub rhs: ConstraintSet,
}

#[derive(Debug, Clone)]
pub enum TestItem {
    AssertEqv(AssertEqv),
    AssertNotEqv(AssertNotEqv),
}

#[derive(Debug)]
pub struct TestFile {
    pub assertions: Vec<TestItem>,
}
