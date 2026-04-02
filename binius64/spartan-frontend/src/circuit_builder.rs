// Copyright 2025 Irreducible Inc.

use std::{array, backtrace::Backtrace, collections::HashMap, mem};

use binius_field::{BinaryField128bGhash as B128, Field};
use bytemuck::zeroed_vec;
use smallvec::{SmallVec, smallvec};

use crate::constraint_system::{
    ConstraintSystem, ConstraintWire, MulConstraint, Operand, WireKind, WitnessIndex, WitnessLayout,
};

/// Common interface for circuit construction and witness generation.
///
/// This trait enables writing circuit logic once that works for both abstract circuit
/// expression ([`ConstraintBuilder`]) and concrete witness generation ([`WitnessGenerator`]).
/// The same function can build constraints symbolically or evaluate them with concrete values.
pub trait CircuitBuilder {
    type Wire: Copy;

    fn assert_zero(&mut self, wire: Self::Wire);

    fn assert_eq(&mut self, lhs: Self::Wire, rhs: Self::Wire) {
        let diff = self.add(lhs, rhs);
        self.assert_zero(diff);
    }

    fn constant(&mut self, val: B128) -> Self::Wire;

    fn add(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire;

    // Addition is subtraction in characteristic 2.
    fn sub(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
        self.add(lhs, rhs)
    }

    fn mul(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire;

    fn hint<F: Fn([B128; IN]) -> [B128; OUT], const IN: usize, const OUT: usize>(
        &mut self,
        inputs: [Self::Wire; IN],
        f: F,
    ) -> [Self::Wire; OUT];
}

#[derive(Debug)]
pub struct WireAllocator {
    n_wires: u32,
    kind: WireKind,
}

impl WireAllocator {
    pub fn new(kind: WireKind) -> Self {
        WireAllocator { n_wires: 0, kind }
    }

    pub fn alloc(&mut self) -> ConstraintWire {
        let wire = ConstraintWire {
            kind: self.kind,
            id: self.n_wires,
        };
        self.n_wires += 1;
        wire
    }
}

// TODO: Add string labels for constraints to make validation easier.

// Witness values are a permuted subset of the wire values.
// Need a way to fingerprint a constraint system.

/// Status of a private wire during optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireStatus {
    /// Wire is unknown - not yet analyzed for elimination
    Unknown,
    /// Wire is pinned - explicitly kept alive
    Pinned,
    /// Wire has been pruned (eliminated)
    Pruned,
}

/// Intermediate representation of a constraint system that can be manipulated and optimized.
///
/// This IR is used during circuit construction and optimization passes like wire elimination.
/// It tracks wire allocators, constants, and constraints, along with metadata about which
/// private wires are still alive (not eliminated by optimization).
#[derive(Debug)]
pub struct ConstraintSystemIR {
    pub(crate) constant_alloc: WireAllocator,
    pub(crate) public_alloc: WireAllocator,
    pub(crate) private_alloc: WireAllocator,
    pub(crate) constants: HashMap<B128, u32>,
    pub(crate) zero_constraints: Vec<Operand<ConstraintWire>>,
    pub(crate) mul_constraints: Vec<MulConstraint<ConstraintWire>>,
    /// Tracks the status of private wires (Unknown, Pinned, or Pruned).
    /// Index corresponds to private wire ID. Initially all wires are Unknown.
    pub(crate) private_wires_status: Vec<WireStatus>,
}

impl ConstraintSystemIR {
    /// Finalize the IR into a ConstraintSystem and WitnessLayout by converting remaining
    /// zero constraints to MulConstraints, computing the final witness layout, and mapping all
    /// ConstraintWires to WitnessIndices.
    ///
    /// Internally looks up or allocates a constant wire with value 1, used to convert
    /// zero constraints of the form `A = 0` into MulConstraints `A * 1 = 0`.
    pub fn finalize(mut self) -> (ConstraintSystem, WitnessLayout) {
        // Look up or allocate a constant wire for ONE
        let one_id = self
            .constants
            .entry(B128::ONE)
            .or_insert_with(|| self.constant_alloc.alloc().id);
        let one_wire = ConstraintWire {
            kind: WireKind::Constant,
            id: *one_id,
        };

        // Convert constants HashMap to Vec
        let mut constants = zeroed_vec(self.constant_alloc.n_wires as usize);
        for (val, id) in self.constants {
            constants[id as usize] = val;
        }

        // Replace all remaining zero constraints with mul constraints
        let one_operand = Operand::from(one_wire);
        let zero_operand = Operand::default();
        for operand in mem::take(&mut self.zero_constraints) {
            if !operand.is_empty() {
                self.mul_constraints.push(MulConstraint {
                    a: operand,
                    b: one_operand.clone(),
                    c: zero_operand.clone(),
                });
            }
        }

        // Create private_alive array from wire status (invert pruned logic)
        let private_alive: Vec<bool> = self
            .private_wires_status
            .iter()
            .map(|&status| !matches!(status, WireStatus::Pruned))
            .collect();

        // Create WitnessLayout
        let layout =
            WitnessLayout::sparse(constants.clone(), self.public_alloc.n_wires, &private_alive);

        // Map all ConstraintWire to WitnessIndex
        let map_operand = |operand: &Operand<ConstraintWire>| -> Operand<WitnessIndex> {
            let indices: SmallVec<[WitnessIndex; 4]> = operand
                .wires()
                .iter()
                .filter_map(|wire| layout.get(wire))
                .collect();
            Operand::new(indices)
        };

        let mul_constraints = self
            .mul_constraints
            .iter()
            .map(|constraint| MulConstraint {
                a: map_operand(&constraint.a),
                b: map_operand(&constraint.b),
                c: map_operand(&constraint.c),
            })
            .collect();

        // Map one_wire to WitnessIndex
        let one_wire_index = layout
            .get(&one_wire)
            .expect("one_wire constant should exist in layout");

        let cs = ConstraintSystem::new(
            constants,
            layout.n_inout() as u32,
            layout.n_private() as u32,
            layout.log_public(),
            mul_constraints,
            one_wire_index,
        );

        (cs, layout)
    }
}

/// Builds constraint systems symbolically by recording operations as constraints.
///
/// Implements [`CircuitBuilder`] with [`ConstraintWire`] as the wire type. Operations like
/// `add` and `mul` allocate new wires and record constraints without evaluating values.
#[derive(Debug)]
pub struct ConstraintBuilder {
    ir: ConstraintSystemIR,
}

impl ConstraintBuilder {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        ConstraintBuilder {
            ir: ConstraintSystemIR {
                constant_alloc: WireAllocator::new(WireKind::Constant),
                public_alloc: WireAllocator::new(WireKind::InOut),
                private_alloc: WireAllocator::new(WireKind::Private),
                constants: HashMap::new(),
                zero_constraints: Vec::new(),
                mul_constraints: Vec::new(),
                private_wires_status: Vec::new(),
            },
        }
    }

    pub fn alloc_inout(&mut self) -> ConstraintWire {
        self.ir.public_alloc.alloc()
    }

    pub fn build(self) -> ConstraintSystemIR {
        self.ir
    }
}

impl CircuitBuilder for ConstraintBuilder {
    type Wire = ConstraintWire;

    fn assert_zero(&mut self, wire: Self::Wire) {
        self.ir.zero_constraints.push(wire.into())
    }

    fn assert_eq(&mut self, lhs: Self::Wire, rhs: Self::Wire) {
        self.ir
            .zero_constraints
            .push(Operand::new(smallvec![lhs, rhs]));
    }

    fn constant(&mut self, val: B128) -> Self::Wire {
        let id = self
            .ir
            .constants
            .entry(val)
            .or_insert_with(|| self.ir.constant_alloc.alloc().id);
        ConstraintWire {
            kind: WireKind::Constant,
            id: *id,
        }
    }

    fn add(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
        let out = self.ir.private_alloc.alloc();
        self.ir.private_wires_status.push(WireStatus::Unknown);
        self.ir
            .zero_constraints
            .push(Operand::new(smallvec![lhs, rhs, out]));
        out
    }

    fn mul(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
        let out = self.ir.private_alloc.alloc();
        self.ir.private_wires_status.push(WireStatus::Unknown);
        self.ir.mul_constraints.push(MulConstraint {
            a: lhs.into(),
            b: rhs.into(),
            c: out.into(),
        });
        out
    }

    fn hint<F: Fn([B128; IN]) -> [B128; OUT], const IN: usize, const OUT: usize>(
        &mut self,
        _inputs: [Self::Wire; IN],
        _f: F,
    ) -> [Self::Wire; OUT] {
        array::from_fn(|_| {
            let wire = self.ir.private_alloc.alloc();
            self.ir.private_wires_status.push(WireStatus::Unknown);
            wire
        })
    }
}

#[derive(Debug)]
pub struct WitnessError {
    pub backtrace: Backtrace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessWire(B128);

impl WitnessWire {
    #[inline]
    pub fn val(self) -> B128 {
        self.0
    }
}

/// Generates witness values by evaluating circuit operations with concrete field elements.
///
/// Implements [`CircuitBuilder`] with [`WitnessWire`] as the wire type. Operations like
/// `add` and `mul` compute actual field values and populate the witness array. Captures
/// the first constraint violation as an error for debugging.
#[derive(Debug)]
pub struct WitnessGenerator<'a> {
    alloc: WireAllocator,
    witness: Vec<B128>,
    layout: &'a WitnessLayout,
    first_error: Option<Backtrace>,
}

impl<'a> WitnessGenerator<'a> {
    pub fn new(layout: &'a WitnessLayout) -> Self {
        let witness_size = layout.size();

        let mut witness = zeroed_vec(witness_size);
        witness[..layout.constants.len()].copy_from_slice(&layout.constants);

        Self {
            alloc: WireAllocator::new(WireKind::Private),
            witness,
            layout,
            first_error: None,
        }
    }

    fn alloc_value(&mut self, value: B128) -> WitnessWire {
        let wire = self.alloc.alloc();
        self.write_value(wire, value)
    }

    fn write_value(&mut self, wire: ConstraintWire, value: B128) -> WitnessWire {
        if let Some(index) = self.layout.get(&wire) {
            self.witness[index.0 as usize] = value;
        }
        WitnessWire(value)
    }

    pub fn write_inout(&mut self, wire: ConstraintWire, value: B128) -> WitnessWire {
        assert_eq!(wire.kind, WireKind::InOut);
        self.write_value(wire, value)
    }

    pub fn build(self) -> Result<Vec<B128>, WitnessError> {
        if let Some(backtrace) = self.first_error {
            Err(WitnessError { backtrace })
        } else {
            Ok(self.witness)
        }
    }

    pub fn error(&self) -> Option<&Backtrace> {
        self.first_error.as_ref()
    }

    fn record_error(&mut self) {
        if self.first_error.is_none() {
            self.first_error = Some(Backtrace::capture());
        }
    }
}

impl<'a> CircuitBuilder for WitnessGenerator<'a> {
    type Wire = WitnessWire;

    fn assert_zero(&mut self, wire: Self::Wire) {
        if wire.val() != B128::ZERO {
            self.record_error();
        }
    }

    fn assert_eq(&mut self, lhs: Self::Wire, rhs: Self::Wire) {
        if lhs != rhs {
            self.record_error();
        }
    }

    fn constant(&mut self, val: B128) -> Self::Wire {
        WitnessWire(val)
    }

    fn add(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
        self.alloc_value(lhs.val() + rhs.val())
    }

    fn mul(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
        self.alloc_value(lhs.val() * rhs.val())
    }

    fn hint<F: Fn([B128; IN]) -> [B128; OUT], const IN: usize, const OUT: usize>(
        &mut self,
        inputs: [Self::Wire; IN],
        f: F,
    ) -> [Self::Wire; OUT] {
        f(inputs.map(WitnessWire::val)).map(|value| self.alloc_value(value))
    }
}

#[cfg(test)]
mod tests {
    use std::iter::successors;

    use binius_field::{Field, PackedField};

    use super::*;

    fn fibonacci<Builder: CircuitBuilder>(
        builder: &mut Builder,
        x0: Builder::Wire,
        x1: Builder::Wire,
        n: usize,
    ) -> Builder::Wire {
        if n == 0 {
            return x0;
        }

        let (_xnsub1, xn) = successors(Some((x0, x1)), |&(a, b)| {
            let next = builder.mul(a, b);
            Some((b, next))
        })
        .nth(n - 1)
        .expect("closure always returns Some");

        xn
    }

    #[test]
    fn test_fibonacci() {
        let mut constraint_builder = ConstraintBuilder::new();
        let x0 = constraint_builder.alloc_inout();
        let x1 = constraint_builder.alloc_inout();
        let xn = constraint_builder.alloc_inout();
        let out = fibonacci(&mut constraint_builder, x0, x1, 20);
        constraint_builder.assert_eq(out, xn);
        let ir = constraint_builder.build();
        let (constraint_system, layout) = ir.finalize();

        let mut witness_generator = WitnessGenerator::new(&layout);
        let x0 = witness_generator.write_inout(x0, B128::ONE);
        let x1 = witness_generator.write_inout(x1, B128::MULTIPLICATIVE_GENERATOR);
        let xn = witness_generator.write_inout(xn, B128::MULTIPLICATIVE_GENERATOR.pow(6765));
        let out = fibonacci(&mut witness_generator, x0, x1, 20);
        witness_generator.assert_eq(out, xn);
        let witness = witness_generator.build().unwrap();

        constraint_system.validate(&witness);
    }

    #[test]
    fn test_assertion_failure_captured() {
        let mut constraint_builder = ConstraintBuilder::new();
        let x = constraint_builder.alloc_inout();
        let y = constraint_builder.alloc_inout();
        let sum = constraint_builder.add(x, y);
        let expected = constraint_builder.alloc_inout();
        constraint_builder.assert_eq(sum, expected);
        let ir = constraint_builder.build();
        let (_constraint_system, layout) = ir.finalize();

        let mut witness_generator = WitnessGenerator::new(&layout);
        let x_val = B128::new(5);
        let y_val = B128::new(7);
        let wrong_expected = B128::new(99); // Incorrect: should be 5 + 7 = 2 in binary field

        let x_wire = witness_generator.write_inout(x, x_val);
        let y_wire = witness_generator.write_inout(y, y_val);
        let sum_wire = witness_generator.add(x_wire, y_wire);
        let expected_wire = witness_generator.write_inout(expected, wrong_expected);
        witness_generator.assert_eq(sum_wire, expected_wire);

        assert!(witness_generator.build().is_err());
    }
}
