// Copyright 2025 Irreducible Inc.
use binius_core::constraint_system::{AndConstraint, MulConstraint, ShiftedValueIndex, ValueIndex};
use cranelift_entity::{EntitySet, SecondaryMap};
use smallvec::{SmallVec, smallvec};

use crate::compiler::Wire;

/// Builder for creating constraints using Wire references
pub struct ConstraintBuilder {
    pub and_constraints: Vec<WireAndConstraint>,
    pub mul_constraints: Vec<WireMulConstraint>,
    pub linear_constraints: Vec<WireLinearConstraint>,
}

impl ConstraintBuilder {
    pub fn new() -> Self {
        Self {
            and_constraints: Vec::new(),
            mul_constraints: Vec::new(),
            linear_constraints: Vec::new(),
        }
    }

    /// Build an AND constraint: A ' B = C
    pub fn and(&mut self) -> AndConstraintBuilder<'_> {
        AndConstraintBuilder::new(self)
    }

    /// Build a MUL constraint: A * B = (HI << 64) | LO
    pub fn mul(&mut self) -> MulConstraintBuilder<'_> {
        MulConstraintBuilder::new(self)
    }

    /// Build a linear constraint: RHS = DST
    /// (where RHS is XOR of shifted values and DST is a
    /// single wire)
    pub fn linear(&mut self) -> LinearConstraintBuilder<'_> {
        LinearConstraintBuilder::new(self)
    }

    /// Convert all wire-based constraints to ValueIndex-based constraints.
    pub fn build(
        self,
        wire_mapping: &SecondaryMap<Wire, ValueIndex>,
        all_one: Wire,
    ) -> (Vec<AndConstraint>, Vec<MulConstraint>) {
        let mut and_constraints = self
            .and_constraints
            .into_iter()
            .map(|c| c.into_constraint(wire_mapping))
            .collect::<Vec<_>>();

        let mul_constraints = self
            .mul_constraints
            .into_iter()
            .map(|c| c.into_constraint(wire_mapping))
            .collect();

        // Convert linear constraints to AND constraints (rhs & all_one = dst)
        if !self.linear_constraints.is_empty() {
            let all_one = wire_mapping[all_one];
            for linear_constraint in self.linear_constraints {
                let and_constraint = linear_constraint.into_and_constraint(wire_mapping, all_one);
                and_constraints.push(and_constraint);
            }
        }

        (and_constraints, mul_constraints)
    }

    pub fn mark_used_wires(&self) -> EntitySet<Wire> {
        let mut used_set = EntitySet::new();
        for ac in &self.and_constraints {
            ac.mark_used(&mut used_set);
        }
        for mc in &self.mul_constraints {
            mc.mark_used(&mut used_set);
        }
        for lc in &self.linear_constraints {
            lc.mark_used(&mut used_set);
        }
        used_set
    }
}

impl Default for ConstraintBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to convert operand to ShiftedValueIndex
fn expand_and_convert_operand(
    operand: WireOperand,
    wire_mapping: &SecondaryMap<Wire, ValueIndex>,
) -> Vec<ShiftedValueIndex> {
    operand
        .into_iter()
        .map(|sw| sw.to_shifted_value_index(wire_mapping))
        .collect()
}

/// AND constraint using Wire references
pub struct WireAndConstraint {
    pub a: WireOperand,
    pub b: WireOperand,
    pub c: WireOperand,
}

impl WireAndConstraint {
    fn into_constraint(self, wire_mapping: &SecondaryMap<Wire, ValueIndex>) -> AndConstraint {
        AndConstraint {
            a: expand_and_convert_operand(self.a, wire_mapping),
            b: expand_and_convert_operand(self.b, wire_mapping),
            c: expand_and_convert_operand(self.c, wire_mapping),
        }
    }

    fn mark_used(&self, used_set: &mut EntitySet<Wire>) {
        mark_used(&self.a, used_set);
        mark_used(&self.b, used_set);
        mark_used(&self.c, used_set);
    }
}

/// MUL constraint using Wire references
pub struct WireMulConstraint {
    pub a: WireOperand,
    pub b: WireOperand,
    pub hi: WireOperand,
    pub lo: WireOperand,
}

/// LINEAR constraint using Wire references
pub struct WireLinearConstraint {
    pub rhs: WireOperand,
    pub dst: Wire,
}

impl WireLinearConstraint {
    fn into_and_constraint(
        self,
        wire_mapping: &SecondaryMap<Wire, ValueIndex>,
        all_ones: ValueIndex,
    ) -> AndConstraint {
        let dst = wire_mapping[self.dst];
        AndConstraint {
            a: expand_and_convert_operand(self.rhs, wire_mapping),
            b: vec![ShiftedValueIndex::plain(all_ones)],
            c: vec![ShiftedValueIndex::plain(dst)],
        }
    }

    fn mark_used(&self, used_set: &mut EntitySet<Wire>) {
        mark_used(&self.rhs, used_set);
        used_set.insert(self.dst);
    }
}

impl WireMulConstraint {
    fn into_constraint(self, wire_mapping: &SecondaryMap<Wire, ValueIndex>) -> MulConstraint {
        MulConstraint {
            a: expand_and_convert_operand(self.a, wire_mapping),
            b: expand_and_convert_operand(self.b, wire_mapping),
            hi: expand_and_convert_operand(self.hi, wire_mapping),
            lo: expand_and_convert_operand(self.lo, wire_mapping),
        }
    }

    fn mark_used(&self, used_set: &mut EntitySet<Wire>) {
        mark_used(&self.a, used_set);
        mark_used(&self.b, used_set);
        mark_used(&self.hi, used_set);
        mark_used(&self.lo, used_set);
    }
}

/// Operand built from wire expressions
pub type WireOperand = Vec<ShiftedWire>;

fn mark_used(operand: &WireOperand, used_set: &mut EntitySet<Wire>) {
    for shifted_wire in operand {
        used_set.insert(shifted_wire.wire);
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ShiftedWire {
    pub wire: Wire,
    pub shift: Shift,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum Shift {
    None,
    Sll(u32),
    Sll32(u32),
    Srl(u32),
    Srl32(u32),
    Sar(u32),
    Sra32(u32),
    Rotr(u32),
    Rotr32(u32),
}

impl Shift {
    /// Try to compose two shift operations.
    ///
    /// Returns None if the shifts are incompatible.
    pub fn compose(lhs: Shift, rhs: Shift) -> Option<Self> {
        match (lhs, rhs) {
            (Shift::None, shift) | (shift, Shift::None) => Some(shift),
            (Shift::Sll(a), Shift::Sll(b)) => {
                // Left shift composition: shl(shl(x, a), b) = shl(x, a + b)
                let combined = a + b;
                if combined < 64 {
                    Some(Shift::Sll(combined))
                } else {
                    None
                }
            }
            (Shift::Sll32(a), Shift::Sll32(b)) => {
                // 32-bit half-wise left shift composition
                let combined = a + b;
                if combined < 32 {
                    Some(Shift::Sll32(combined))
                } else {
                    None
                }
            }
            (Shift::Srl(a), Shift::Srl(b)) => {
                // Logical right shift composition: shr(shr(x, a), b) = shr(x, a + b)
                let combined = a + b;
                if combined < 64 {
                    Some(Shift::Srl(combined))
                } else {
                    None
                }
            }
            (Shift::Srl32(a), Shift::Srl32(b)) => {
                // 32-bit half-wise logical right shift composition
                let combined = a + b;
                if combined < 32 {
                    Some(Shift::Srl32(combined))
                } else {
                    None
                }
            }
            (Shift::Sar(a), Shift::Sar(b)) => {
                // Arithmetic right shift composition
                let combined = a + b;
                if combined < 64 {
                    Some(Shift::Sar(combined))
                } else {
                    None
                }
            }
            (Shift::Sra32(a), Shift::Sra32(b)) => {
                // 32-bit half-wise arithmetic right shift composition
                let combined = a + b;
                if combined < 32 {
                    Some(Shift::Sra32(combined))
                } else {
                    None
                }
            }
            (Shift::Rotr(a), Shift::Rotr(b)) => {
                // Rotate right composition: rotr(rotr(x, a), b) = rotr(x, (a + b) % 64)
                let combined = (a + b) % 64;
                Some(Shift::Rotr(combined))
            }
            (Shift::Rotr32(a), Shift::Rotr32(b)) => {
                // 32-bit half-wise rotate right composition
                let combined = (a + b) % 32;
                Some(Shift::Rotr32(combined))
            }
            _ => None, // Different shift types are not composable
        }
    }
}

impl ShiftedWire {
    fn to_shifted_value_index(
        self,
        wire_mapping: &SecondaryMap<Wire, ValueIndex>,
    ) -> ShiftedValueIndex {
        let idx = wire_mapping[self.wire];
        match self.shift {
            Shift::None => ShiftedValueIndex::plain(idx),
            Shift::Sll(n) => {
                if n == 0 {
                    ShiftedValueIndex::plain(idx)
                } else {
                    ShiftedValueIndex::sll(idx, n as usize)
                }
            }
            Shift::Sll32(n) => {
                if n == 0 {
                    ShiftedValueIndex::plain(idx)
                } else {
                    ShiftedValueIndex::sll32(idx, n as usize)
                }
            }
            Shift::Srl(n) => {
                if n == 0 {
                    ShiftedValueIndex::plain(idx)
                } else {
                    ShiftedValueIndex::srl(idx, n as usize)
                }
            }
            Shift::Srl32(n) => {
                if n == 0 {
                    ShiftedValueIndex::plain(idx)
                } else {
                    ShiftedValueIndex::srl32(idx, n as usize)
                }
            }
            Shift::Sar(n) => {
                if n == 0 {
                    ShiftedValueIndex::plain(idx)
                } else {
                    ShiftedValueIndex::sar(idx, n as usize)
                }
            }
            Shift::Sra32(n) => {
                if n == 0 {
                    ShiftedValueIndex::plain(idx)
                } else {
                    ShiftedValueIndex::sra32(idx, n as usize)
                }
            }
            Shift::Rotr(n) => {
                if n == 0 {
                    ShiftedValueIndex::plain(idx)
                } else {
                    ShiftedValueIndex::rotr(idx, n as usize)
                }
            }
            Shift::Rotr32(n) => {
                if n == 0 {
                    ShiftedValueIndex::plain(idx)
                } else {
                    ShiftedValueIndex::rotr32(idx, n as usize)
                }
            }
        }
    }
}

pub struct AndConstraintBuilder<'a> {
    builder: &'a mut ConstraintBuilder,
    a: WireOperand,
    b: WireOperand,
    c: WireOperand,
}

impl<'a> AndConstraintBuilder<'a> {
    fn new(builder: &'a mut ConstraintBuilder) -> Self {
        Self {
            builder,
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
        }
    }

    /// Set the A operand
    pub fn a(mut self, expr: impl Into<WireExpr>) -> Self {
        self.a = expr.into().to_operand();
        self
    }

    /// Set the B operand
    pub fn b(mut self, expr: impl Into<WireExpr>) -> Self {
        self.b = expr.into().to_operand();
        self
    }

    /// Set the C operand
    pub fn c(mut self, expr: impl Into<WireExpr>) -> Self {
        self.c = expr.into().to_operand();
        self
    }

    /// Finalize and add the constraint
    pub fn build(self) {
        self.builder.and_constraints.push(WireAndConstraint {
            a: self.a,
            b: self.b,
            c: self.c,
        });
    }
}

pub struct MulConstraintBuilder<'a> {
    builder: &'a mut ConstraintBuilder,
    a: WireOperand,
    b: WireOperand,
    hi: WireOperand,
    lo: WireOperand,
}

pub struct LinearConstraintBuilder<'a> {
    builder: &'a mut ConstraintBuilder,
    rhs: WireOperand,
    dst: Option<Wire>,
}

impl<'a> MulConstraintBuilder<'a> {
    fn new(builder: &'a mut ConstraintBuilder) -> Self {
        Self {
            builder,
            a: Vec::new(),
            b: Vec::new(),
            hi: Vec::new(),
            lo: Vec::new(),
        }
    }

    pub fn a(mut self, expr: impl Into<WireExpr>) -> Self {
        self.a = expr.into().to_operand();
        self
    }

    pub fn b(mut self, expr: impl Into<WireExpr>) -> Self {
        self.b = expr.into().to_operand();
        self
    }

    pub fn hi(mut self, expr: impl Into<WireExpr>) -> Self {
        self.hi = expr.into().to_operand();
        self
    }

    pub fn lo(mut self, expr: impl Into<WireExpr>) -> Self {
        self.lo = expr.into().to_operand();
        self
    }

    pub fn build(self) {
        self.builder.mul_constraints.push(WireMulConstraint {
            a: self.a,
            b: self.b,
            hi: self.hi,
            lo: self.lo,
        });
    }
}

impl<'a> LinearConstraintBuilder<'a> {
    fn new(builder: &'a mut ConstraintBuilder) -> Self {
        Self {
            builder,
            rhs: Vec::new(),
            dst: None,
        }
    }

    /// Set the RHS operand (XOR combination of shifted values)
    pub fn rhs(mut self, expr: impl Into<WireExpr>) -> Self {
        self.rhs = expr.into().to_operand();
        self
    }

    /// Set the DST operand (destination wire)
    pub fn dst(mut self, wire: Wire) -> Self {
        self.dst = Some(wire);
        self
    }

    /// Finalize and add the linear constraint.
    ///
    /// Panics if `dst` wasn't assigned.
    pub fn build(self) {
        self.builder.linear_constraints.push(WireLinearConstraint {
            rhs: self.rhs,
            dst: self.dst.expect("dst wire must be assigned"),
        });
    }
}

/// Expression for building wire operands as an XOR accumulation of terms.
#[derive(Clone)]
pub struct WireExpr(SmallVec<[WireExprTerm; 4]>);

/// Individual term in XOR expression
#[derive(Copy, Clone)]
pub enum WireExprTerm {
    Wire(Wire),
    Shifted(Wire, ShiftOp),
}

#[derive(Copy, Clone)]
pub enum ShiftOp {
    Sll(u32),
    Sll32(u32),
    Srl(u32),
    Srl32(u32),
    Sar(u32),
    Sra32(u32),
    Rotr(u32),
    Rotr32(u32),
}

impl WireExpr {
    #[allow(clippy::wrong_self_convention)]
    fn to_operand(self) -> WireOperand {
        let mut result = Vec::new();
        for term in self.0 {
            let shifted_wire = term.to_shifted_wire();

            result.push(shifted_wire);
        }
        result
    }
}

impl WireExprTerm {
    fn to_shifted_wire(self) -> ShiftedWire {
        match self {
            WireExprTerm::Wire(w) => ShiftedWire {
                wire: w,
                shift: Shift::None,
            },
            WireExprTerm::Shifted(w, op) => ShiftedWire {
                wire: w,
                shift: match op {
                    ShiftOp::Sll(n) => Shift::Sll(n),
                    ShiftOp::Sll32(n) => Shift::Sll32(n),
                    ShiftOp::Srl(n) => Shift::Srl(n),
                    ShiftOp::Srl32(n) => Shift::Srl32(n),
                    ShiftOp::Sar(n) => Shift::Sar(n),
                    ShiftOp::Sra32(n) => Shift::Sra32(n),
                    ShiftOp::Rotr(n) => Shift::Rotr(n),
                    ShiftOp::Rotr32(n) => Shift::Rotr32(n),
                },
            },
        }
    }
}

// Convenience functions
pub fn wire(w: Wire) -> WireExpr {
    WireExpr(smallvec![w.into()])
}

pub fn sll(w: Wire, n: u32) -> WireExprTerm {
    WireExprTerm::Shifted(w, ShiftOp::Sll(n))
}

pub fn sll32(w: Wire, n: u32) -> WireExprTerm {
    WireExprTerm::Shifted(w, ShiftOp::Sll32(n))
}

pub fn srl(w: Wire, n: u32) -> WireExprTerm {
    WireExprTerm::Shifted(w, ShiftOp::Srl(n))
}

pub fn srl32(w: Wire, n: u32) -> WireExprTerm {
    WireExprTerm::Shifted(w, ShiftOp::Srl32(n))
}

pub fn sar(w: Wire, n: u32) -> WireExprTerm {
    WireExprTerm::Shifted(w, ShiftOp::Sar(n))
}

pub fn sra32(w: Wire, n: u32) -> WireExprTerm {
    WireExprTerm::Shifted(w, ShiftOp::Sra32(n))
}

pub fn rotr(w: Wire, n: u32) -> WireExprTerm {
    WireExprTerm::Shifted(w, ShiftOp::Rotr(n))
}

pub fn rotr32(w: Wire, n: u32) -> WireExprTerm {
    WireExprTerm::Shifted(w, ShiftOp::Rotr32(n))
}

// XOR helpers for common cases
pub fn xor2(a: impl Into<WireExprTerm>, b: impl Into<WireExprTerm>) -> WireExpr {
    WireExpr(smallvec![a.into(), b.into()])
}

pub fn xor3(
    a: impl Into<WireExprTerm>,
    b: impl Into<WireExprTerm>,
    c: impl Into<WireExprTerm>,
) -> WireExpr {
    WireExpr(smallvec![a.into(), b.into(), c.into()])
}

pub fn xor4(
    a: impl Into<WireExprTerm>,
    b: impl Into<WireExprTerm>,
    c: impl Into<WireExprTerm>,
    d: impl Into<WireExprTerm>,
) -> WireExpr {
    WireExpr(smallvec![a.into(), b.into(), c.into(), d.into()])
}

pub fn xor_multi(terms: impl IntoIterator<Item = WireExprTerm>) -> WireExpr {
    WireExpr(terms.into_iter().collect())
}

// Empty operand helper
pub fn empty() -> WireExpr {
    WireExpr(smallvec![])
}

// Implement conversions
impl From<Wire> for WireExpr {
    fn from(w: Wire) -> Self {
        wire(w)
    }
}

impl From<Wire> for WireExprTerm {
    fn from(w: Wire) -> Self {
        WireExprTerm::Wire(w)
    }
}

impl From<WireExprTerm> for WireExpr {
    fn from(expr: WireExprTerm) -> Self {
        WireExpr(smallvec![expr])
    }
}

#[cfg(test)]
mod tests {
    use binius_core::constraint_system::ShiftVariant;
    use cranelift_entity::EntityRef;

    use super::*;

    #[test]
    fn test_rotr_zero_optimization_with_builder() {
        // Test that rotr(w, 0) is optimized to plain(w) using the builder API
        // and produces the expected final ConstraintSystem

        // Setup wire mapping
        let mut wire_mapping = SecondaryMap::new();
        let wire_a = Wire::new(0);
        let wire_b = Wire::new(1);
        let wire_c = Wire::new(2);
        let all_one_wire = Wire::new(3);

        wire_mapping[wire_a] = ValueIndex(0);
        wire_mapping[wire_b] = ValueIndex(1);
        wire_mapping[wire_c] = ValueIndex(2);
        wire_mapping[all_one_wire] = ValueIndex(3);

        // Test case 1: Linear constraint with rotr(0)
        // c = rotr(a, 0) ⊕ b
        {
            let mut builder = ConstraintBuilder::new();

            // Build: c = rotr(a, 0) ⊕ b
            builder
                .linear()
                .rhs(xor2(rotr(wire_a, 0), wire_b))
                .dst(wire_c)
                .build();

            let (and_constraints, mul_constraints) = builder.build(&wire_mapping, all_one_wire);

            // rotr(0) should be optimized to plain wire, so we expect:
            // (a ⊕ b) & all_one = c
            assert_eq!(and_constraints.len(), 1);
            assert_eq!(mul_constraints.len(), 0);

            let and_c = &and_constraints[0];

            // Check operand a: should have plain(0) and plain(1)
            assert_eq!(and_c.a.len(), 2);
            assert!(
                and_c
                    .a
                    .iter()
                    .any(|svi| svi.value_index == ValueIndex(0) && svi.amount == 0)
            );
            assert!(
                and_c
                    .a
                    .iter()
                    .any(|svi| svi.value_index == ValueIndex(1) && svi.amount == 0)
            );

            // Check operand b: should be all_one
            assert_eq!(and_c.b.len(), 1);
            assert_eq!(and_c.b[0].value_index, ValueIndex(3));
            assert_eq!(and_c.b[0].amount, 0);

            // Check operand c: should be wire_c
            assert_eq!(and_c.c.len(), 1);
            assert_eq!(and_c.c[0].value_index, ValueIndex(2));
            assert_eq!(and_c.c[0].amount, 0);
        }

        // Test case 2: Linear constraint with rotr(n) where n > 0
        // c = rotr(a, 5) ⊕ b
        {
            let mut builder = ConstraintBuilder::new();

            // Build: c = rotr(a, 5) ⊕ b
            builder
                .linear()
                .rhs(xor2(rotr(wire_a, 5), wire_b))
                .dst(wire_c)
                .build();

            let (and_constraints, mul_constraints) = builder.build(&wire_mapping, all_one_wire);

            assert_eq!(and_constraints.len(), 1);
            assert_eq!(mul_constraints.len(), 0);

            let and_c = &and_constraints[0];

            // Operand a should have: ror(a, 5), plain(b)
            assert_eq!(and_c.a.len(), 2);

            // Check for native ror(a, 5)
            assert!(and_c.a.iter().any(|svi| {
                svi.value_index == ValueIndex(0)
                    && svi.amount == 5
                    && matches!(svi.shift_variant, ShiftVariant::Rotr)
            }));

            // Check for plain(b)
            assert!(
                and_c
                    .a
                    .iter()
                    .any(|svi| svi.value_index == ValueIndex(1) && svi.amount == 0)
            );
        }
    }

    #[test]
    fn test_rotr_in_and_constraint() {
        // Test rotr in AND constraints: (a & rotr(b, 0)) ⊕ c = 0

        let mut wire_mapping = SecondaryMap::new();
        let wire_a = Wire::new(0);
        let wire_b = Wire::new(1);
        let wire_c = Wire::new(2);
        let all_one_wire = Wire::new(3);

        wire_mapping[wire_a] = ValueIndex(0);
        wire_mapping[wire_b] = ValueIndex(1);
        wire_mapping[wire_c] = ValueIndex(2);
        wire_mapping[all_one_wire] = ValueIndex(3);

        // Test with rotr(0)
        {
            let mut builder = ConstraintBuilder::new();

            // Build: a & rotr(b, 0) ⊕ c = 0
            builder.and().a(wire_a).b(rotr(wire_b, 0)).c(wire_c).build();

            let (and_constraints, _) = builder.build(&wire_mapping, all_one_wire);

            assert_eq!(and_constraints.len(), 1);
            let and_c = &and_constraints[0];

            // Check operand a: plain wire_a
            assert_eq!(and_c.a.len(), 1);
            assert_eq!(and_c.a[0].value_index, ValueIndex(0));
            assert_eq!(and_c.a[0].amount, 0);

            // Check operand b: should be plain wire_b (rotr(0) optimized)
            assert_eq!(and_c.b.len(), 1);
            assert_eq!(and_c.b[0].value_index, ValueIndex(1));
            assert_eq!(and_c.b[0].amount, 0);

            // Check operand c: plain wire_c
            assert_eq!(and_c.c.len(), 1);
            assert_eq!(and_c.c[0].value_index, ValueIndex(2));
            assert_eq!(and_c.c[0].amount, 0);
        }

        // Test with rotr(8) - should expand
        {
            let mut builder = ConstraintBuilder::new();

            // Build: a & rotr(b, 8) ⊕ c = 0
            builder.and().a(wire_a).b(rotr(wire_b, 8)).c(wire_c).build();

            let (and_constraints, _) = builder.build(&wire_mapping, all_one_wire);

            assert_eq!(and_constraints.len(), 1);
            let and_c = &and_constraints[0];

            // Check operand b: should have native ror(b, 8)
            assert_eq!(and_c.b.len(), 1);

            assert!(and_c.b.iter().any(|svi| {
                svi.value_index == ValueIndex(1)
                    && svi.amount == 8
                    && matches!(svi.shift_variant, ShiftVariant::Rotr)
            }));
        }
    }

    #[test]
    fn test_complex_expression_with_rotr() {
        // Test a more complex expression: c = rotr(a, 0) ⊕ sll(b, 5) ⊕ rotr(a, 12)

        let mut wire_mapping = SecondaryMap::new();
        let wire_a = Wire::new(0);
        let wire_b = Wire::new(1);
        let wire_c = Wire::new(2);
        let all_one_wire = Wire::new(3);

        wire_mapping[wire_a] = ValueIndex(0);
        wire_mapping[wire_b] = ValueIndex(1);
        wire_mapping[wire_c] = ValueIndex(2);
        wire_mapping[all_one_wire] = ValueIndex(3);

        let mut builder = ConstraintBuilder::new();

        // Build complex expression
        builder
            .linear()
            .rhs(xor3(rotr(wire_a, 0), sll(wire_b, 5), rotr(wire_a, 12)))
            .dst(wire_c)
            .build();

        let (and_constraints, mul_constraints) = builder.build(&wire_mapping, all_one_wire);

        assert_eq!(and_constraints.len(), 1);
        assert_eq!(mul_constraints.len(), 0);

        let and_c = &and_constraints[0];

        // Expected operand a components:
        // - plain(a) from rotr(a, 0)
        // - sll(b, 5)
        // - ror(a, 12)
        assert_eq!(and_c.a.len(), 3);

        // Check for plain(a) from rotr(0)
        assert!(
            and_c
                .a
                .iter()
                .any(|svi| svi.value_index == ValueIndex(0) && svi.amount == 0),
            "Should have plain(a) from rotr(a, 0)"
        );

        // Check for sll(b, 5)
        assert!(
            and_c.a.iter().any(|svi| {
                svi.value_index == ValueIndex(1)
                    && svi.amount == 5
                    && matches!(svi.shift_variant, ShiftVariant::Sll)
            }),
            "Should have sll(b, 5)"
        );

        // Check for ror(a, 12)
        assert!(
            and_c.a.iter().any(|svi| {
                svi.value_index == ValueIndex(0)
                    && svi.amount == 12
                    && matches!(svi.shift_variant, ShiftVariant::Rotr)
            }),
            "Should have native ror(a, 12)"
        );
    }
}
