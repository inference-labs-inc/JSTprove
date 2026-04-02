// Copyright 2025 Irreducible Inc.

use std::{cmp::Ordering, collections::HashMap, mem};

use binius_field::{BinaryField128bGhash as B128, Field};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use smallvec::{SmallVec, smallvec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WireKind {
    Constant,
    InOut,
    Private,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConstraintWire {
    pub(crate) kind: WireKind,
    pub(crate) id: u32,
}

#[derive(Debug, Clone)]
pub struct Operand<W>(SmallVec<[W; 4]>);

impl<W> Default for Operand<W> {
    fn default() -> Self {
        Operand(SmallVec::new())
    }
}

impl<W: Copy + Ord> Operand<W> {
    pub fn new(mut term: SmallVec<[W; 4]>) -> Self {
        term.sort_unstable();

        let has_duplicate_wire = term.windows(2).any(|w| w[0] == w[1]);
        let term = if has_duplicate_wire {
            term.chunk_by(|a, b| a == b)
                .flat_map(|group| {
                    // Group is a slice of wires that are all equal. We want to return an empty
                    // iterator if the group is even length and a singleton iterator otherwise.
                    let last_even_idx = group.len() / 2 * 2;
                    group[last_even_idx..].iter().copied()
                })
                .collect()
        } else {
            term
        };

        Self(term)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn wires(&self) -> &[W] {
        &self.0
    }

    pub fn merge(&mut self, rhs: &Self) -> (Operand<W>, Operand<W>) {
        // Classic merge algorithm for sorted vectors, but where duplicate items cancel out.
        let lhs = mem::take(&mut self.0);
        let dst = &mut self.0;

        let mut lhs_iter = lhs.into_iter().peekable();
        let mut rhs_iter = rhs.0.iter().copied().peekable();

        let mut additions = Operand::default();
        let mut removals = Operand::default();

        loop {
            match (lhs_iter.peek(), rhs_iter.peek()) {
                (Some(next_lhs), Some(next_rhs)) => {
                    match next_lhs.cmp(next_rhs) {
                        Ordering::Equal => {
                            // Advance both iterators, but don't push the wires because they cancel.
                            let wire = lhs_iter.next().expect("peek returned Some");
                            let _ = rhs_iter.next().expect("peek returned Some");

                            removals.0.push(wire);
                        }
                        Ordering::Less => dst.push(lhs_iter.next().expect("peek returned Some")),
                        Ordering::Greater => {
                            let wire = rhs_iter.next().expect("peek returned Some");
                            additions.0.push(wire);
                            dst.push(wire);
                        }
                    }
                }
                (Some(_), None) => dst.push(lhs_iter.next().expect("peek returned Some")),
                (None, Some(_)) => {
                    let wire = rhs_iter.next().expect("peek returned Some");
                    additions.0.push(wire);
                    dst.push(wire);
                }
                (None, None) => break,
            }
        }

        (additions, removals)
    }
}

impl<W> From<W> for Operand<W> {
    fn from(value: W) -> Self {
        Operand(smallvec![value])
    }
}

#[derive(Debug, Clone)]
pub struct MulConstraint<W> {
    pub a: Operand<W>,
    pub b: Operand<W>,
    pub c: Operand<W>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct WitnessIndex(pub u32);

/// A constraint system with multiplication constraints over witness indices.
///
/// Contains multiplication constraints of the form `A * B = C` where A, B, C are operands
/// (XOR combinations of witness values). Constraints directly reference [`WitnessIndex`]
/// positions in the witness array.
///
/// This struct does not guarantee power-of-two constraint counts or witness size. Use
/// [`ConstraintSystemPadded`] for a version with blinding and power-of-two padding.
#[derive(Debug, Clone)]
pub struct ConstraintSystem<F: Field = B128> {
    constants: Vec<F>,
    n_inout: u32,
    n_private: u32,
    log_public: u32,
    mul_constraints: Vec<MulConstraint<WitnessIndex>>,
    one_wire: WitnessIndex,
}

impl<F: Field> ConstraintSystem<F> {
    /// Create a new constraint system.
    pub fn new(
        constants: Vec<F>,
        n_inout: u32,
        n_private: u32,
        log_public: u32,
        mul_constraints: Vec<MulConstraint<WitnessIndex>>,
        one_wire: WitnessIndex,
    ) -> Self {
        Self {
            constants,
            n_inout,
            n_private,
            log_public,
            mul_constraints,
            one_wire,
        }
    }

    pub fn constants(&self) -> &[F] {
        &self.constants
    }

    pub fn n_inout(&self) -> u32 {
        self.n_inout
    }

    pub fn n_private(&self) -> u32 {
        self.n_private
    }

    pub fn log_public(&self) -> u32 {
        self.log_public
    }

    pub fn n_public(&self) -> u32 {
        1 << self.log_public
    }

    pub fn mul_constraints(&self) -> &[MulConstraint<WitnessIndex>] {
        &self.mul_constraints
    }

    pub fn one_wire(&self) -> WitnessIndex {
        self.one_wire
    }

    /// Validate that a witness satisfies all multiplication constraints.
    pub fn validate(&self, witness: &[B128]) {
        let operand_val = |operand: &Operand<WitnessIndex>| {
            operand
                .wires()
                .iter()
                .map(|idx| witness[idx.0 as usize])
                .sum::<B128>()
        };

        for MulConstraint { a, b, c } in &self.mul_constraints {
            assert_eq!(operand_val(a) * operand_val(b), operand_val(c));
        }
    }
}

/// A constraint system with blinding and power-of-two padding.
///
/// Wraps a [`ConstraintSystem`], adds dummy constraints for blinding, and pads the total
/// number of constraints to a power of two (required by the prover's multilinear extension
/// protocol).
#[derive(Debug, Clone)]
pub struct ConstraintSystemPadded<F: Field = B128> {
    inner: ConstraintSystem<F>,
    log_size: u32,
    blinding_info: BlindingInfo,
    mul_constraints: Vec<MulConstraint<WitnessIndex>>,
}

impl<F: Field> ConstraintSystemPadded<F> {
    /// Create a new padded constraint system with blinding.
    ///
    /// This:
    /// 1. Adds dummy multiplication constraints for blinding (3 wires each: A * B = C)
    /// 2. Pads the total constraint count to a power of two with `one * one = one` constraints
    /// 3. Calculates the log_size based on witness requirements
    pub fn new(cs: ConstraintSystem<F>, blinding_info: BlindingInfo) -> Self {
        let mut mul_constraints = cs.mul_constraints.clone();

        // Calculate witness size and log_size
        let n_public = cs.n_public() as usize;
        let n_private = cs.n_private as usize;
        let total_witness_size = n_public
            + n_private
            + blinding_info.n_dummy_wires
            + 3 * blinding_info.n_dummy_constraints;
        let log_size = log2_ceil_usize(total_witness_size) as u32;

        // Add dummy constraints for blinding
        // Each dummy constraint uses 3 consecutive wires starting after n_dummy_wires
        let dummy_constraint_wire_base = n_public + n_private + blinding_info.n_dummy_wires;
        for i in 0..blinding_info.n_dummy_constraints {
            let a = WitnessIndex((dummy_constraint_wire_base + 3 * i) as u32);
            let b = WitnessIndex((dummy_constraint_wire_base + 3 * i + 1) as u32);
            let c = WitnessIndex((dummy_constraint_wire_base + 3 * i + 2) as u32);

            mul_constraints.push(MulConstraint {
                a: Operand::from(a),
                b: Operand::from(b),
                c: Operand::from(c),
            });
        }

        // Pad to next power of two with `one * one = one` constraints
        let one_operand = Operand::from(cs.one_wire);
        let current_len = mul_constraints.len();
        mul_constraints.resize(
            current_len.next_power_of_two(),
            MulConstraint {
                a: one_operand.clone(),
                b: one_operand.clone(),
                c: one_operand.clone(),
            },
        );

        Self {
            inner: cs,
            log_size,
            blinding_info,
            mul_constraints,
        }
    }

    pub fn constants(&self) -> &[F] {
        self.inner.constants()
    }

    pub fn n_inout(&self) -> u32 {
        self.inner.n_inout()
    }

    pub fn n_private(&self) -> u32 {
        self.inner.n_private()
    }

    pub fn log_public(&self) -> u32 {
        self.inner.log_public()
    }

    pub fn n_public(&self) -> u32 {
        self.inner.n_public()
    }

    pub fn one_wire(&self) -> WitnessIndex {
        self.inner.one_wire()
    }

    pub fn log_size(&self) -> u32 {
        self.log_size
    }

    pub fn size(&self) -> usize {
        1 << self.log_size as usize
    }

    pub fn blinding_info(&self) -> &BlindingInfo {
        &self.blinding_info
    }

    pub fn mul_constraints(&self) -> &[MulConstraint<WitnessIndex>] {
        &self.mul_constraints
    }

    pub fn validate(&self, witness: &[B128]) {
        assert_eq!(witness.len(), self.size());

        let operand_val = |operand: &Operand<WitnessIndex>| {
            operand
                .wires()
                .iter()
                .map(|idx| witness[idx.0 as usize])
                .sum::<B128>()
        };

        for MulConstraint { a, b, c } in &self.mul_constraints {
            assert_eq!(operand_val(a) * operand_val(b), operand_val(c));
        }
    }
}

#[derive(Debug, Clone)]
pub struct WitnessLayout {
    pub(crate) constants: Vec<B128>,
    n_inout: u32,
    n_private: u32,
    log_public: u32,
    log_size: u32,
    private_index_map: HashMap<u32, u32>,
}

impl WitnessLayout {
    pub fn sparse(constants: Vec<B128>, n_inout: u32, private_alive: &[bool]) -> Self {
        let n_constants = constants.len() as u32;
        let n_public = n_constants + n_inout;
        let log_public = log2_ceil_usize(n_public as usize) as u32;

        let private_offset = 1 << log_public;
        let private_index_map = private_alive
            .iter()
            .enumerate()
            .filter(|(_, alive)| **alive)
            .enumerate()
            .map(|(new_idx, (id, _))| (id as u32, private_offset + new_idx as u32))
            .collect::<HashMap<_, _>>();

        let n_private = private_index_map.len() as u32;
        let log_size = log2_ceil_usize((private_offset + n_private) as usize) as u32;

        Self {
            constants,
            n_inout,
            n_private,
            log_public,
            log_size,
            private_index_map,
        }
    }

    pub fn with_blinding(self, info: BlindingInfo) -> Self {
        let log_public = self.log_public;
        let n_private = self.n_private as usize;

        let private_offset = 1 << log_public as usize;
        let total_size =
            private_offset + n_private + info.n_dummy_wires + 3 * info.n_dummy_constraints;
        let log_size = log2_ceil_usize(total_size) as u32;

        Self { log_size, ..self }
    }

    pub fn size(&self) -> usize {
        1 << self.log_size as usize
    }

    pub fn n_constants(&self) -> usize {
        self.constants.len()
    }

    pub fn n_inout(&self) -> usize {
        self.n_inout as usize
    }

    pub fn n_private(&self) -> usize {
        self.n_private as usize
    }

    pub fn log_public(&self) -> u32 {
        self.log_public
    }

    pub fn log_size(&self) -> u32 {
        self.log_size
    }

    /// Returns the first index of the inout
    pub fn inout_offset(&self) -> WitnessIndex {
        WitnessIndex(self.constants.len() as u32)
    }

    pub fn private_offset(&self) -> WitnessIndex {
        WitnessIndex(1 << self.log_public)
    }

    pub fn get(&self, wire: &ConstraintWire) -> Option<WitnessIndex> {
        match wire.kind {
            WireKind::Constant => {
                assert!((wire.id as usize) < self.constants.len());
                Some(WitnessIndex(wire.id))
            }
            WireKind::InOut => {
                assert!(wire.id < self.n_inout);
                Some(WitnessIndex(self.inout_offset().0 + wire.id))
            }
            WireKind::Private => self
                .private_index_map
                .get(&wire.id)
                .map(|&id| WitnessIndex(id)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BlindingInfo {
    /// The number of random dummy wires that must be added.
    pub n_dummy_wires: usize,
    /// The number of random dummy multiplication constraints that must be added.
    pub n_dummy_constraints: usize,
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;

    #[test]
    fn test_wires_added_mod2() {
        // Create 4 wires with different kinds to ensure proper sorting
        let w = [
            ConstraintWire {
                kind: WireKind::Constant,
                id: 0,
            },
            ConstraintWire {
                kind: WireKind::InOut,
                id: 0,
            },
            ConstraintWire {
                kind: WireKind::Private,
                id: 0,
            },
            ConstraintWire {
                kind: WireKind::Private,
                id: 1,
            },
        ];

        // Input sequence: w[0], w[2], w[2], w[3], w[3], w[1], w[2], w[1], w[3], w[3]
        // Counts: w[0]=1, w[1]=2, w[2]=3, w[3]=4
        // After mod 2: w[0]=1, w[1]=0, w[2]=1, w[3]=0
        let input = smallvec![w[0], w[2], w[2], w[3], w[3], w[1], w[2], w[1], w[3], w[3]];
        let operand = Operand::new(input);

        // Expected result: w[0], w[2] (sorted)
        assert_eq!(operand.wires(), &[w[0], w[2]]);
    }

    #[test]
    fn test_sorting_when_no_duplicates() {
        // Create 4 wires with different kinds to ensure proper sorting
        let w = [
            ConstraintWire {
                kind: WireKind::Constant,
                id: 0,
            },
            ConstraintWire {
                kind: WireKind::InOut,
                id: 0,
            },
            ConstraintWire {
                kind: WireKind::Private,
                id: 0,
            },
            ConstraintWire {
                kind: WireKind::Private,
                id: 1,
            },
        ];

        // Input sequence: w[2], w[3], w[0], w[1]
        let input = smallvec![w[2], w[3], w[0], w[1]];
        let operand = Operand::new(input);

        // Expected result: w[0], w[1], w[2], w[3] (sorted by WireKind then ID)
        assert_eq!(operand.wires(), &[w[0], w[1], w[2], w[3]]);
    }

    #[test]
    fn test_merge() {
        // Create 4 wires with different kinds to ensure proper sorting
        let w = [
            ConstraintWire {
                kind: WireKind::Constant,
                id: 0,
            },
            ConstraintWire {
                kind: WireKind::InOut,
                id: 0,
            },
            ConstraintWire {
                kind: WireKind::Private,
                id: 0,
            },
            ConstraintWire {
                kind: WireKind::Private,
                id: 1,
            },
        ];

        // Test case 1: merge([w[0]], [])
        let mut lhs = Operand(smallvec![w[0]]);
        let rhs = Operand(smallvec![]);
        let (additions, removals) = lhs.merge(&rhs);
        assert_eq!(lhs.wires(), &[w[0]]);
        assert_eq!(additions.wires(), &[]);
        assert_eq!(removals.wires(), &[]);

        // Test case 2: merge([], [w[0]])
        let mut lhs = Operand(smallvec![]);
        let rhs = Operand(smallvec![w[0]]);
        let (additions, removals) = lhs.merge(&rhs);
        assert_eq!(lhs.wires(), &[w[0]]);
        assert_eq!(additions.wires(), &[w[0]]);
        assert_eq!(removals.wires(), &[]);

        // Test case 3: merge([w[0]], [w[0]])
        let mut lhs = Operand(smallvec![w[0]]);
        let rhs = Operand(smallvec![w[0]]);
        let (additions, removals) = lhs.merge(&rhs);
        assert_eq!(lhs.wires(), &[]);
        assert_eq!(additions.wires(), &[]);
        assert_eq!(removals.wires(), &[w[0]]);

        // Test case 4: merge([w[0]], [w[1]])
        let mut lhs = Operand(smallvec![w[0]]);
        let rhs = Operand(smallvec![w[1]]);
        let (additions, removals) = lhs.merge(&rhs);
        assert_eq!(lhs.wires(), &[w[0], w[1]]);
        assert_eq!(additions.wires(), &[w[1]]);
        assert_eq!(removals.wires(), &[]);

        // Test case 5: merge([w[0]], [w[0], w[1]])
        let mut lhs = Operand(smallvec![w[0]]);
        let rhs = Operand(smallvec![w[0], w[1]]);
        let (additions, removals) = lhs.merge(&rhs);
        assert_eq!(lhs.wires(), &[w[1]]);
        assert_eq!(additions.wires(), &[w[1]]);
        assert_eq!(removals.wires(), &[w[0]]);

        // Test case 6: merge([w[0], w[2]], [w[1], w[3]])
        let mut lhs = Operand(smallvec![w[0], w[2]]);
        let rhs = Operand(smallvec![w[1], w[3]]);
        let (additions, removals) = lhs.merge(&rhs);
        assert_eq!(lhs.wires(), &[w[0], w[1], w[2], w[3]]);
        assert_eq!(additions.wires(), &[w[1], w[3]]);
        assert_eq!(removals.wires(), &[]);

        // Test case 7: merge([w[0], w[2]], [w[0], w[1], w[2], w[3]])
        let mut lhs = Operand(smallvec![w[0], w[2]]);
        let rhs = Operand(smallvec![w[0], w[1], w[2], w[3]]);
        let (additions, removals) = lhs.merge(&rhs);
        assert_eq!(lhs.wires(), &[w[1], w[3]]);
        assert_eq!(additions.wires(), &[w[1], w[3]]);
        assert_eq!(removals.wires(), &[w[0], w[2]]);
    }
}
