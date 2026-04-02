// Copyright 2025 Irreducible Inc.
use rustc_hash::FxHashSet;

use super::legraph::LeGraph;
use crate::compiler::{
    Wire,
    constraint_builder::{
        ConstraintBuilder, Shift, ShiftedWire, WireAndConstraint, WireMulConstraint, WireOperand,
    },
    gate_fusion::legraph::ConstraintRef,
};

/// A patch is a description of a change to the constraint system.
///
/// It specifies a list of constraints that are going to be removed and a list of constraints that
/// are going to be added.
///
/// The lhs and rhs MUST be equivalent to preserve soundness.
pub struct Patch {
    /// The constraint set that is going to be replaced with this one.
    subsumes: Vec<ConstraintRef>,
    /// The new constraints that is going to be added to the graph.
    added: AndOrMulConstraint,
}

enum AndOrMulConstraint {
    And(WireAndConstraint),
    Mul(WireMulConstraint),
}

/// Apply the given patches to the constraint builder given.
pub fn apply_patches(cb: &mut ConstraintBuilder, patches: Vec<Patch>) {
    let mut subsumes: FxHashSet<ConstraintRef> = FxHashSet::default();
    subsumes.reserve(patches.len());
    let mut new_and_constraints = Vec::new();
    let mut new_mul_constraints = Vec::new();

    // Collect all subsumed constraints and new constraints to add
    for patch in patches {
        subsumes.extend(patch.subsumes);
        match patch.added {
            AndOrMulConstraint::And(and_constraint) => new_and_constraints.push(and_constraint),
            AndOrMulConstraint::Mul(mul_constraint) => new_mul_constraints.push(mul_constraint),
        }
    }

    // Filter out subsumed constraints from each vector
    // Use std::mem::take to take ownership of the vectors
    let old_and_constraints = std::mem::take(&mut cb.and_constraints);
    cb.and_constraints = old_and_constraints
        .into_iter()
        .enumerate()
        .filter_map(|(index, constraint)| {
            if subsumes.contains(&ConstraintRef::And { index }) {
                None
            } else {
                Some(constraint)
            }
        })
        .collect();

    let old_mul_constraints = std::mem::take(&mut cb.mul_constraints);
    cb.mul_constraints = old_mul_constraints
        .into_iter()
        .enumerate()
        .filter_map(|(index, constraint)| {
            if subsumes.contains(&ConstraintRef::Mul { index }) {
                None
            } else {
                Some(constraint)
            }
        })
        .collect();

    let old_linear_constraints = std::mem::take(&mut cb.linear_constraints);
    cb.linear_constraints = old_linear_constraints
        .into_iter()
        .enumerate()
        .filter_map(|(index, constraint)| {
            if subsumes.contains(&ConstraintRef::Linear { index }) {
                None
            } else {
                Some(constraint)
            }
        })
        .collect();

    // Add the new constraints
    cb.and_constraints.extend(new_and_constraints);
    cb.mul_constraints.extend(new_mul_constraints);
}

/// Builds a list of patches that would remove the inlined linear definitions and potentially
/// AND constraints.
///
/// NB: patches may have overlapping subsumes.
pub fn build(cb: &ConstraintBuilder, leg: &LeGraph, all_one: Wire) -> Vec<Patch> {
    let mut patches = vec![];
    build_non_linear_patches(cb, leg, &mut patches);
    for committed in leg.commit_set().iter() {
        let patch = build_committed_lin_def_patch(cb, leg, all_one, committed);
        patches.push(patch);
    }
    patches
}

/// Collect patches for the non-linear constraints that inline linear definitions.
fn build_non_linear_patches(cb: &ConstraintBuilder, leg: &LeGraph, patches: &mut Vec<Patch>) {
    // Collect *distinct* constraint references for each root constraint.
    let mut constraints = Vec::with_capacity(leg.roots.len());
    for root in leg.roots.iter() {
        constraints.push(leg.root_constraint_ref(*root));
    }
    constraints.sort_unstable();
    constraints.dedup();

    // Create a patch for each distinct constraint.
    for constraint_ref in constraints {
        let patch = build_non_lin_patch(cb, leg, constraint_ref);
        patches.push(patch);
    }
}

fn build_non_lin_patch(
    cb: &ConstraintBuilder,
    leg: &LeGraph,
    constraint_ref: ConstraintRef,
) -> Patch {
    let mut subsumes = vec![constraint_ref];

    let new_constraint = match constraint_ref {
        ConstraintRef::And { index } => {
            let a = process_operand(leg, &mut subsumes, &cb.and_constraints[index].a);
            let b = process_operand(leg, &mut subsumes, &cb.and_constraints[index].b);
            let c = process_operand(leg, &mut subsumes, &cb.and_constraints[index].c);
            AndOrMulConstraint::And(WireAndConstraint { a, b, c })
        }
        ConstraintRef::Mul { index } => {
            let a = process_operand(leg, &mut subsumes, &cb.mul_constraints[index].a);
            let b = process_operand(leg, &mut subsumes, &cb.mul_constraints[index].b);
            let lo = process_operand(leg, &mut subsumes, &cb.mul_constraints[index].lo);
            let hi = process_operand(leg, &mut subsumes, &cb.mul_constraints[index].hi);
            AndOrMulConstraint::Mul(WireMulConstraint { a, b, lo, hi })
        }
        ConstraintRef::Linear { .. } => unreachable!(),
    };

    subsumes.sort_unstable();
    subsumes.dedup();

    Patch {
        subsumes,
        added: new_constraint,
    }
}

/// Build a patch for a committed linear definition.
///
/// Given the wire that defines a linear definition build a patch that replaces the original linear
/// definition and all definitions that could be inlined into it. Therefore, the returned
/// patch will replace the given linear definition and the cone of linear definitions it used.
fn build_committed_lin_def_patch(
    _cb: &ConstraintBuilder,
    leg: &LeGraph,
    all_one: Wire,
    root: Wire,
) -> Patch {
    // `subsumes` is a list of constraints that become redundant with application of this patch.
    // The first redundant constraint is the linear definition that's being committed.
    let mut subsumes = vec![leg.lin_def_constraint_ref(root)];

    let old_operand = leg.lin_def(root);
    let new_operand = process_operand(leg, &mut subsumes, old_operand);

    // Create an AND constraint that enforces: root = new_operand
    Patch {
        subsumes,
        added: AndOrMulConstraint::And(WireAndConstraint {
            a: new_operand,
            b: vec![ShiftedWire {
                wire: all_one,
                shift: Shift::None,
            }],
            c: vec![ShiftedWire {
                wire: root,
                shift: Shift::None,
            }],
        }),
    }
}

fn process_operand(
    leg: &LeGraph,
    subsumes: &mut Vec<ConstraintRef>,
    old_operand: &WireOperand,
) -> WireOperand {
    let mut new_operand = WireOperand::new();
    for term in old_operand {
        process_term(leg, &mut new_operand, subsumes, term.wire, term.shift);
    }
    new_operand
}

/// Recursively process a term, inlining non-committed linear definitions.
fn process_term(
    leg: &LeGraph,
    new_operand: &mut WireOperand,
    subsumes: &mut Vec<ConstraintRef>,
    wire: Wire,
    shift: Shift,
) {
    // Check if this wire is committed or not a linear def (i.e., opaque)
    if leg.commit_set().contains(wire) || !leg.is_lin_def(wire) {
        // This is a terminal or committed wire - add it to the result with the accumulated shift
        new_operand.push(ShiftedWire { wire, shift });
    } else {
        // This is a non-committed linear def - we need to inline it!
        let inner_operand = leg.lin_def(wire);
        let constraint_ref = leg.lin_def_constraint_ref(wire);
        subsumes.push(constraint_ref);

        // Distribute the current shift over all terms in the inner operand
        // This is crucial for correctness: shift(a ^ b) = shift(a) ^ shift(b)
        for inner_term in inner_operand {
            // Compose shifts: we're applying 'shift' to 'inner_term'
            // So we need Shift::compose(inner_term.shift, shift)
            match Shift::compose(inner_term.shift, shift) {
                Some(composed_shift) => {
                    // Recursively process this term with the composed shift
                    process_term(leg, new_operand, subsumes, inner_term.wire, composed_shift);
                }
                None => {
                    // Incompatible shifts - this shouldn't happen if commit set is correct
                    panic!(
                        "Incompatible shifts during inlining: {:?} followed by {:?} for wire {:?}",
                        inner_term.shift, shift, inner_term.wire
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::compiler::{
        Wire,
        constraint_builder::{rotr, rotr32, sll, sll32, sra32, srl, srl32, xor2, xor3},
        gate_fusion::Stat,
    };

    /// Test helper to create a Wire with a given ID
    fn w(id: u32) -> Wire {
        Wire::from_u32(id)
    }

    /// Concise test helper that builds a circuit and verifies both commit decisions and expressions
    fn test_inlining(
        build_constraints: impl FnOnce(&mut ConstraintBuilder),
        expected_committed: &[Wire],
        expected_expressions: &[(Wire, Vec<ShiftedWire>)],
    ) {
        let mut cb = ConstraintBuilder::new();
        build_constraints(&mut cb);

        let mut stat = Stat::default();
        let mut leg = LeGraph::new(&cb, &mut stat);
        crate::compiler::gate_fusion::commit_set::run_decide_commit_set(&mut leg, &mut stat);

        // Verify commit set
        for &wire in expected_committed {
            assert!(
                leg.commit_set().contains(wire),
                "Wire {:?} should be committed but wasn't. Commit set: {:?}",
                wire,
                leg.commit_set()
            );
        }

        // Verify expressions expand correctly
        for &(wire, ref expected_expansion) in expected_expressions {
            let actual = expand_expression(&leg, wire);

            // Convert to BTreeMap for easier comparison (order-independent)
            let expected_map: BTreeMap<(Wire, Shift), usize> =
                expected_expansion
                    .iter()
                    .fold(BTreeMap::new(), |mut map, term| {
                        *map.entry((term.wire, term.shift)).or_insert(0) += 1;
                        map
                    });

            let actual_map: BTreeMap<(Wire, Shift), usize> =
                actual.iter().fold(BTreeMap::new(), |mut map, term| {
                    *map.entry((term.wire, term.shift)).or_insert(0) += 1;
                    map
                });

            assert_eq!(
                actual_map, expected_map,
                "Wire {:?} expanded incorrectly.\nExpected: {:?}\nActual: {:?}",
                wire, expected_expansion, actual
            );
        }
    }

    #[test]
    fn test_rotr_identity_nested() {
        // y = a ^ b
        // z = rotr(y, 20)
        // t = rotr(z, 44)  // total 64 -> identity
        // Expect t expands to: a ^ b (no shifts)
        test_inlining(
            |cb| {
                cb.linear().rhs(xor2(w(0), w(1))).dst(w(2)).build(); // y
                cb.linear().rhs(rotr(w(2), 20)).dst(w(3)).build(); // z
                cb.linear().rhs(rotr(w(3), 44)).dst(w(4)).build(); // t
                cb.and().a(w(4)).b(w(5)).c(w(6)).build();
            },
            &[],
            &[(
                w(4),
                vec![
                    ShiftedWire {
                        wire: w(0),
                        shift: Shift::Rotr(0),
                    },
                    ShiftedWire {
                        wire: w(1),
                        shift: Shift::Rotr(0),
                    },
                ],
            )],
        );
    }

    #[test]
    fn test_committed_linear_of_constant_all_ones_shape() {
        // Directly exercise build_committed_lin_def_patch for a constant RHS.
        fn w(id: u32) -> Wire {
            Wire::from_u32(id)
        }

        let mut cb = ConstraintBuilder::new();
        // x = const (opaque wire in this builder-level test)
        // y = x  (linear def of a single opaque term)
        cb.linear().rhs(xor2(w(0), w(0))).dst(w(1)).build();
        // Above uses xor2(w0,w0) which cancels logically, but at builder-level it's two terms.
        // Use a simpler single-term variant as well
        let mut cb_single = ConstraintBuilder::new();
        cb_single.linear().rhs(xor2(w(0), w(2))).dst(w(3)).build();

        let mut stat = Stat::default();
        let leg = LeGraph::new(&cb_single, &mut stat);
        let all_one = w(9);
        let patch = super::build_committed_lin_def_patch(&cb_single, &leg, all_one, w(3));
        match patch.added {
            AndOrMulConstraint::And(ref andc) => {
                // b must be exactly [all_one]
                assert_eq!(andc.b.len(), 1);
                assert_eq!(andc.b[0].wire, all_one);
                assert!(!andc.a.is_empty());
                assert_eq!(andc.c.len(), 1);
                assert_eq!(andc.c[0].wire, w(3));
            }
            _ => panic!("expected AND constraint in committed patch"),
        }
    }

    #[test]
    fn test_mul_distinct_linears_all_fields() {
        // a, b, hi, lo each reference distinct linear defs; all inlinable (no shifts)
        fn w(id: u32) -> Wire {
            Wire::from_u32(id)
        }

        let mut cb = ConstraintBuilder::new();
        cb.linear().rhs(xor2(w(0), w(1))).dst(w(10)).build(); // a_src
        cb.linear().rhs(xor2(w(2), w(3))).dst(w(11)).build(); // b_src
        cb.linear().rhs(xor2(w(4), w(5))).dst(w(12)).build(); // hi_src
        cb.linear().rhs(xor2(w(6), w(7))).dst(w(13)).build(); // lo_src

        cb.mul().a(w(10)).b(w(11)).hi(w(12)).lo(w(13)).build();

        let mut stat = Stat::default();
        let mut leg = LeGraph::new(&cb, &mut stat);
        crate::compiler::gate_fusion::commit_set::run_decide_commit_set(&mut leg, &mut stat);

        let patches = super::build(&cb, &leg, w(9));
        let mut cb2 = cb;
        super::apply_patches(&mut cb2, patches);

        assert_eq!(cb2.mul_constraints.len(), 1);
        let m = &cb2.mul_constraints[0];
        assert_eq!(m.a.len(), 2);
        assert_eq!(m.b.len(), 2);
        assert_eq!(m.hi.len(), 2);
        assert_eq!(m.lo.len(), 2);
    }

    #[test]
    fn test_stress_shift_combinations_no_panic() {
        // Iterate over a small set of shift pairs; ensure commit_set + build/apply don't panic
        use crate::compiler::constraint_builder::Shift;
        fn w(id: u32) -> Wire {
            Wire::from_u32(id)
        }

        let shifts = [
            Shift::None,
            Shift::Sll(5),
            Shift::Sll32(5),
            Shift::Srl(5),
            Shift::Srl32(5),
            Shift::Sar(5),
            Shift::Sra32(5),
            Shift::Rotr(13),
            Shift::Rotr32(13),
        ];

        for (i, s1) in shifts.iter().enumerate() {
            for (j, s2) in shifts.iter().enumerate() {
                let mut cb = ConstraintBuilder::new();
                // y = shift1(x)
                match s1 {
                    Shift::None => cb.linear().rhs(xor2(w(0), w(1))).dst(w(2)).build(),
                    Shift::Sll(n) => cb.linear().rhs(sll(w(0), *n)).dst(w(2)).build(),
                    Shift::Sll32(n) => cb.linear().rhs(sll32(w(0), *n)).dst(w(2)).build(),
                    Shift::Srl(n) => cb.linear().rhs(srl(w(0), *n)).dst(w(2)).build(),
                    Shift::Srl32(n) => cb.linear().rhs(srl32(w(0), *n)).dst(w(2)).build(),
                    Shift::Sar(n) => cb
                        .linear()
                        .rhs(crate::compiler::constraint_builder::sar(w(0), *n))
                        .dst(w(2))
                        .build(),
                    Shift::Sra32(n) => cb.linear().rhs(sra32(w(0), *n)).dst(w(2)).build(),
                    Shift::Rotr(n) => cb.linear().rhs(rotr(w(0), *n)).dst(w(2)).build(),
                    Shift::Rotr32(n) => cb.linear().rhs(rotr32(w(0), *n)).dst(w(2)).build(),
                }
                // z = shift2(y)
                match s2 {
                    Shift::None => cb.linear().rhs(xor2(w(2), w(3))).dst(w(4)).build(),
                    Shift::Sll(n) => cb.linear().rhs(sll(w(2), *n)).dst(w(4)).build(),
                    Shift::Sll32(n) => cb.linear().rhs(sll32(w(2), *n)).dst(w(4)).build(),
                    Shift::Srl(n) => cb.linear().rhs(srl(w(2), *n)).dst(w(4)).build(),
                    Shift::Srl32(n) => cb.linear().rhs(srl32(w(2), *n)).dst(w(4)).build(),
                    Shift::Sar(n) => cb
                        .linear()
                        .rhs(crate::compiler::constraint_builder::sar(w(2), *n))
                        .dst(w(4))
                        .build(),
                    Shift::Sra32(n) => cb.linear().rhs(sra32(w(2), *n)).dst(w(4)).build(),
                    Shift::Rotr(n) => cb.linear().rhs(rotr(w(2), *n)).dst(w(4)).build(),
                    Shift::Rotr32(n) => cb.linear().rhs(rotr32(w(2), *n)).dst(w(4)).build(),
                }
                cb.and().a(w(4)).b(w(5)).c(w(6)).build();

                let mut stat = Stat::default();
                let mut leg = LeGraph::new(&cb, &mut stat);
                crate::compiler::gate_fusion::commit_set::run_decide_commit_set(
                    &mut leg, &mut stat,
                );
                let patches = super::build(&cb, &leg, w(7));
                let mut cb2 = cb;
                super::apply_patches(&mut cb2, patches);

                // Basic sanity: we should have at least one AND constraint after patches
                assert!(
                    !cb2.and_constraints.is_empty(),
                    "empty AND set for pair ({},{})",
                    i,
                    j
                );
            }
        }
    }

    /// Helper to expand an expression fully (for testing)
    fn expand_expression(leg: &LeGraph, wire: Wire) -> Vec<ShiftedWire> {
        use crate::compiler::constraint_builder::WireOperand;

        let mut result = WireOperand::new();

        if !leg.is_lin_def(wire) {
            // Not a linear def - return as is
            result.push(ShiftedWire {
                wire,
                shift: Shift::None,
            });
            return result;
        }

        let operand = leg.lin_def(wire);
        for term in operand {
            expand_term_recursive(leg, &mut result, term.wire, term.shift);
        }
        result
    }

    fn expand_term_recursive(
        leg: &LeGraph,
        result: &mut Vec<ShiftedWire>,
        wire: Wire,
        shift: Shift,
    ) {
        // Check if this wire is committed OR not a linear def (terminal)
        if !leg.is_lin_def(wire) || leg.commit_set().contains(wire) {
            // Terminal or committed - add it as is
            result.push(ShiftedWire { wire, shift });
        } else {
            // This is a non-committed linear def - expand recursively
            let inner = leg.lin_def(wire);
            for term in inner {
                let composed = Shift::compose(term.shift, shift).unwrap();
                expand_term_recursive(leg, result, term.wire, composed);
            }
        }
    }

    /// Build a frequency map for an operand for order-independent comparison.
    fn operand_count_map(ops: &[ShiftedWire]) -> std::collections::BTreeMap<(Wire, Shift), usize> {
        let mut map = std::collections::BTreeMap::new();
        for t in ops {
            *map.entry((t.wire, t.shift)).or_insert(0) += 1;
        }
        map
    }

    fn assert_operand_eq(actual: &[ShiftedWire], expected: Vec<ShiftedWire>, ctx: &str) {
        let am = operand_count_map(actual);
        let em = operand_count_map(&expected);
        assert_eq!(
            am, em,
            "operand mismatch for {}\nexpected: {:?}\nactual:   {:?}",
            ctx, expected, actual
        );
    }

    // Now let's copy some test cases from plan_tests.rs and verify both commits and expressions

    #[test]
    fn test_simple_xor_inlining() {
        // Test: y = x ^ a, z = y ^ b, then z is used in AND
        // Both y and z should be inlinable into the AND constraint
        test_inlining(
            |cb| {
                // y = x ^ a
                cb.linear().rhs(xor2(w(0), w(1))).dst(w(2)).build();
                // z = y ^ b
                cb.linear().rhs(xor2(w(2), w(3))).dst(w(4)).build();
                // Use z in AND constraint (creates the root)
                cb.and().a(w(4)).b(w(5)).c(w(6)).build();
            },
            &[],
            &[
                // z should expand to: x ^ a ^ b
                (
                    w(4),
                    vec![
                        ShiftedWire {
                            wire: w(0),
                            shift: Shift::None,
                        },
                        ShiftedWire {
                            wire: w(1),
                            shift: Shift::None,
                        },
                        ShiftedWire {
                            wire: w(3),
                            shift: Shift::None,
                        },
                    ],
                ),
                // y should expand to: x ^ a
                (
                    w(2),
                    vec![
                        ShiftedWire {
                            wire: w(0),
                            shift: Shift::None,
                        },
                        ShiftedWire {
                            wire: w(1),
                            shift: Shift::None,
                        },
                    ],
                ),
            ],
        );
    }

    #[test]
    fn test_shift_composition_same_type() {
        // Test: y = x << 10, z = y << 20
        // Shifts should compose: z = x << 30
        test_inlining(
            |cb| {
                // y = x << 10
                cb.linear().rhs(sll(w(0), 10)).dst(w(1)).build();
                // z = y << 20
                cb.linear().rhs(sll(w(1), 20)).dst(w(2)).build();
                // Use z in an AND constraint so it becomes a root
                cb.and().a(w(2)).b(w(3)).c(w(4)).build();
            },
            &[],
            &[
                // z should expand to: x << 30
                (
                    w(2),
                    vec![ShiftedWire {
                        wire: w(0),
                        shift: Shift::Sll(30),
                    }],
                ),
                // y should expand to: x << 10
                (
                    w(1),
                    vec![ShiftedWire {
                        wire: w(0),
                        shift: Shift::Sll(10),
                    }],
                ),
            ],
        );
    }

    #[test]
    fn test_rotr_distributes_over_xor() {
        // Test: y = a ^ b, z = rotr(y, 5)
        // Should distribute: z = rotr(a, 5) ^ rotr(b, 5)
        test_inlining(
            |cb| {
                // y = a ^ b
                cb.linear().rhs(xor2(w(0), w(1))).dst(w(2)).build();
                // z = rotr(y, 5)
                cb.linear().rhs(rotr(w(2), 5)).dst(w(3)).build();
                // Use z in an AND constraint
                cb.and().a(w(3)).b(w(4)).c(w(5)).build();
            },
            &[],
            &[
                // z should expand to: rotr(a, 5) ^ rotr(b, 5)
                (
                    w(3),
                    vec![
                        ShiftedWire {
                            wire: w(0),
                            shift: Shift::Rotr(5),
                        },
                        ShiftedWire {
                            wire: w(1),
                            shift: Shift::Rotr(5),
                        },
                    ],
                ),
            ],
        );
    }

    #[test]
    fn test_shift_composition_different_types() {
        // Test: y = x << 10, z = y >> 20
        // Different shift types cannot compose, y must be committed
        test_inlining(
            |cb| {
                // y = x << 10
                cb.linear().rhs(sll(w(0), 10)).dst(w(1)).build();
                // z = y >> 20
                cb.linear().rhs(srl(w(1), 20)).dst(w(2)).build();
                // Use z in an AND constraint
                cb.and().a(w(2)).b(w(3)).c(w(4)).build();
            },
            &[w(1)], // y must be committed (incompatible shifts)
            &[
                // z should expand to: y >> 20 (y is committed, not inlined)
                (
                    w(2),
                    vec![ShiftedWire {
                        wire: w(1),
                        shift: Shift::Srl(20),
                    }],
                ),
            ],
        );
    }

    #[test]
    fn test_complex_xor_chain() {
        // Test: y = a ^ b ^ c, z = y ^ d ^ e
        // Both should be inlinable
        test_inlining(
            |cb| {
                // y = a ^ b ^ c
                cb.linear().rhs(xor3(w(0), w(1), w(2))).dst(w(3)).build();
                // z = y ^ d ^ e
                cb.linear().rhs(xor3(w(3), w(4), w(5))).dst(w(6)).build();
                // Use z in AND constraint
                cb.and().a(w(6)).b(w(7)).c(w(8)).build();
            },
            &[],
            &[
                // z should expand to: a ^ b ^ c ^ d ^ e
                (
                    w(6),
                    vec![
                        ShiftedWire {
                            wire: w(0),
                            shift: Shift::None,
                        },
                        ShiftedWire {
                            wire: w(1),
                            shift: Shift::None,
                        },
                        ShiftedWire {
                            wire: w(2),
                            shift: Shift::None,
                        },
                        ShiftedWire {
                            wire: w(4),
                            shift: Shift::None,
                        },
                        ShiftedWire {
                            wire: w(5),
                            shift: Shift::None,
                        },
                    ],
                ),
            ],
        );
    }

    #[test]
    fn test_apply_patches() {
        use super::ConstraintRef;

        // Create a constraint builder with various constraints
        let mut cb = ConstraintBuilder::new();

        // Add some AND constraints
        cb.and().a(w(0)).b(w(1)).c(w(2)).build(); // index 0
        cb.and().a(w(3)).b(w(4)).c(w(5)).build(); // index 1
        cb.and().a(w(6)).b(w(7)).c(w(8)).build(); // index 2

        // Add some MUL constraints
        cb.mul().a(w(9)).b(w(10)).lo(w(11)).hi(w(12)).build(); // index 0
        cb.mul().a(w(13)).b(w(14)).lo(w(15)).hi(w(16)).build(); // index 1

        // Add some LINEAR constraints
        cb.linear().rhs(xor2(w(17), w(18))).dst(w(19)).build(); // index 0
        cb.linear().rhs(xor2(w(20), w(21))).dst(w(22)).build(); // index 1

        // Create patches that:
        // 1. Replace AND constraint at index 1 with a new one
        // 2. Replace LINEAR constraint at index 0 with an AND constraint
        // 3. Replace MUL constraint at index 0 with a new one
        let patches = vec![
            Patch {
                subsumes: vec![ConstraintRef::And { index: 1 }],
                added: AndOrMulConstraint::And(WireAndConstraint {
                    a: vec![ShiftedWire {
                        wire: w(30),
                        shift: Shift::None,
                    }],
                    b: vec![ShiftedWire {
                        wire: w(31),
                        shift: Shift::None,
                    }],
                    c: vec![ShiftedWire {
                        wire: w(32),
                        shift: Shift::None,
                    }],
                }),
            },
            Patch {
                subsumes: vec![ConstraintRef::Linear { index: 0 }],
                added: AndOrMulConstraint::And(WireAndConstraint {
                    a: vec![ShiftedWire {
                        wire: w(33),
                        shift: Shift::None,
                    }],
                    b: vec![ShiftedWire {
                        wire: w(34),
                        shift: Shift::None,
                    }],
                    c: vec![ShiftedWire {
                        wire: w(35),
                        shift: Shift::None,
                    }],
                }),
            },
            Patch {
                subsumes: vec![ConstraintRef::Mul { index: 0 }],
                added: AndOrMulConstraint::Mul(WireMulConstraint {
                    a: vec![ShiftedWire {
                        wire: w(36),
                        shift: Shift::None,
                    }],
                    b: vec![ShiftedWire {
                        wire: w(37),
                        shift: Shift::None,
                    }],
                    lo: vec![ShiftedWire {
                        wire: w(38),
                        shift: Shift::None,
                    }],
                    hi: vec![ShiftedWire {
                        wire: w(39),
                        shift: Shift::None,
                    }],
                }),
            },
        ];

        // Apply patches
        apply_patches(&mut cb, patches);

        // Verify results
        // AND constraints: originally 3, removed index 1, added 2 new ones = 4 total
        assert_eq!(cb.and_constraints.len(), 4);
        // Check that original constraints at indices 0 and 2 are preserved
        assert_eq!(cb.and_constraints[0].a[0].wire, w(0));
        assert_eq!(cb.and_constraints[1].a[0].wire, w(6)); // was index 2, now index 1
        // Check new constraints are added at the end
        assert_eq!(cb.and_constraints[2].a[0].wire, w(30));
        assert_eq!(cb.and_constraints[3].a[0].wire, w(33));

        // MUL constraints: originally 2, removed index 0, added 1 new one = 2 total
        assert_eq!(cb.mul_constraints.len(), 2);
        // Check that original constraint at index 1 is preserved (now at index 0)
        assert_eq!(cb.mul_constraints[0].a[0].wire, w(13));
        // Check new constraint is added at the end
        assert_eq!(cb.mul_constraints[1].a[0].wire, w(36));

        // LINEAR constraints: originally 2, removed index 0 = 1 total
        assert_eq!(cb.linear_constraints.len(), 1);
        // Check that original constraint at index 1 is preserved (now at index 0)
        assert_eq!(cb.linear_constraints[0].rhs[0].wire, w(20));
    }

    #[test]
    fn test_patch_overlap_committed_and_non_linear() {
        // Build a scenario where a committed-linear patch and a non-linear patch both subsume
        // the same inner linear (overlap), and ensure apply_patches handles it correctly.
        //
        // t = a ^ b                    // linear (index 0)
        // y = srl(t, 10)               // linear (index 1), will be committed due to incompatible
        // use z = sll(y, 5)                // linear (index 2)
        // AND1 uses z (root)           // non-linear patch inlines z, subsumes z
        // AND2 uses t (root)           // non-linear patch inlines t, subsumes t
        // y committed patch subsumes y and also t (overlap with AND2 patch)
        let mut cb = ConstraintBuilder::new();

        // Inputs
        fn w(id: u32) -> Wire {
            Wire::from_u32(id)
        }
        // t = a ^ b
        cb.linear().rhs(xor2(w(0), w(1))).dst(w(2)).build();
        // y = srl(t, 10)
        cb.linear().rhs(srl(w(2), 10)).dst(w(3)).build();
        // z = sll(y, 5)
        cb.linear().rhs(sll(w(3), 5)).dst(w(4)).build();

        // AND1: use z
        cb.and().a(w(4)).b(w(5)).c(w(6)).build();
        // AND2: use t
        cb.and().a(w(2)).b(w(7)).c(w(8)).build();

        let mut stat = Stat::default();
        let mut leg = LeGraph::new(&cb, &mut stat);
        crate::compiler::gate_fusion::commit_set::run_decide_commit_set(&mut leg, &mut stat);

        // Sanity: y should be committed; t and z should not be
        assert!(leg.commit_set().contains(w(3)), "y should be committed");
        assert!(
            !leg.commit_set().contains(w(2)),
            "t should not be committed"
        );
        assert!(
            !leg.commit_set().contains(w(4)),
            "z should not be committed"
        );

        let patches = super::build(&cb, &leg, w(9));
        let mut cb2 = cb; // clone-by-move and apply patches
        super::apply_patches(&mut cb2, patches);

        // Expectations:
        // - AND constraints: start 2, both subsumed and replaced (2), plus 1 from committed y => 3
        assert_eq!(cb2.and_constraints.len(), 3);
        // - Linear constraints: t, y, z all subsumed => 0 remaining
        assert_eq!(cb2.linear_constraints.len(), 0);
    }

    #[test]
    fn test_mul_operand_duplicate_inlining() {
        // Build MUL where the same linear def appears twice in a single operand:
        // y = x ^ c
        // MUL.a = y ^ y ^ z  (duplicate y)
        // After inlining, expect: a = x ^ c ^ x ^ c ^ z (5 terms, preserving duplicates)
        fn w(id: u32) -> Wire {
            Wire::from_u32(id)
        }

        let mut cb = ConstraintBuilder::new();
        // y = x ^ c
        cb.linear().rhs(xor2(w(0), w(1))).dst(w(2)).build();
        // MUL: a = y ^ y ^ z; b = u; hi, lo are outputs
        cb.mul()
            .a(crate::compiler::constraint_builder::xor3(w(2), w(2), w(3)))
            .b(w(4))
            .hi(w(5))
            .lo(w(6))
            .build();

        let mut stat = Stat::default();
        let mut leg = LeGraph::new(&cb, &mut stat);
        crate::compiler::gate_fusion::commit_set::run_decide_commit_set(&mut leg, &mut stat);

        let patches = super::build(&cb, &leg, w(7));
        let mut cb2 = cb;
        super::apply_patches(&mut cb2, patches);

        // Verify results: one MUL remains, and its operand a equals x ^ c ^ x ^ c ^ z
        assert_eq!(cb2.mul_constraints.len(), 1);
        let a = &cb2.mul_constraints[0].a;
        assert_operand_eq(
            a,
            vec![
                ShiftedWire {
                    wire: w(0),
                    shift: Shift::None,
                },
                ShiftedWire {
                    wire: w(1),
                    shift: Shift::None,
                },
                ShiftedWire {
                    wire: w(0),
                    shift: Shift::None,
                },
                ShiftedWire {
                    wire: w(1),
                    shift: Shift::None,
                },
                ShiftedWire {
                    wire: w(3),
                    shift: Shift::None,
                },
            ],
            "mul.a",
        );
    }

    #[test]
    fn test_mul_mixed_inlinable_and_committed() {
        // Scenario: committed linear feeds one MUL operand; inlinable linear feeds the other.
        // t_committed = sll(a, 40)
        // y = sll(t_committed, 30)  // uses committed
        // u = x ^ c                  // inlinable
        // MUL: a=y, b=u
        fn w(id: u32) -> Wire {
            Wire::from_u32(id)
        }

        let mut cb = ConstraintBuilder::new();
        cb.linear().rhs(sll(w(0), 40)).dst(w(1)).build(); // t_committed
        cb.linear().rhs(sll(w(1), 30)).dst(w(2)).build(); // y
        cb.linear().rhs(xor2(w(3), w(4))).dst(w(5)).build(); // u
        cb.mul().a(w(2)).b(w(5)).hi(w(6)).lo(w(7)).build();

        let mut stat = Stat::default();
        let mut leg = LeGraph::new(&cb, &mut stat);
        crate::compiler::gate_fusion::commit_set::run_decide_commit_set(&mut leg, &mut stat);

        assert!(
            leg.commit_set().contains(w(1)),
            "t_committed should be committed"
        );
        let patches = super::build(&cb, &leg, w(8));
        let mut cb2 = cb;
        super::apply_patches(&mut cb2, patches);

        assert_eq!(cb2.mul_constraints.len(), 1);
        let m = &cb2.mul_constraints[0];
        assert_operand_eq(
            &m.a,
            vec![ShiftedWire {
                wire: w(1),
                shift: Shift::Sll(30),
            }],
            "mul.a (committed)",
        );
        assert_operand_eq(
            &m.b,
            vec![
                ShiftedWire {
                    wire: w(3),
                    shift: Shift::None,
                },
                ShiftedWire {
                    wire: w(4),
                    shift: Shift::None,
                },
            ],
            "mul.b (inlinable)",
        );
    }

    #[test]
    fn test_mul_hi_lo_mixed_inlinable_committed() {
        // hi comes from a committed path; lo from an inlinable XOR
        fn w(id: u32) -> Wire {
            Wire::from_u32(id)
        }

        let mut cb = ConstraintBuilder::new();
        // committed producer
        cb.linear().rhs(sll(w(0), 48)).dst(w(1)).build(); // t
        cb.linear().rhs(sll(w(1), 20)).dst(w(2)).build(); // hi_src (should commit t)
        // inlinable lo_src = x ^ c
        cb.linear().rhs(xor2(w(3), w(4))).dst(w(5)).build();
        // build MUL: a,b plain; hi=hi_src; lo=lo_src
        cb.mul().a(w(6)).b(w(7)).hi(w(2)).lo(w(5)).build();

        let mut stat = Stat::default();
        let mut leg = LeGraph::new(&cb, &mut stat);
        crate::compiler::gate_fusion::commit_set::run_decide_commit_set(&mut leg, &mut stat);

        assert!(
            leg.commit_set().contains(w(1)),
            "inner t should be committed"
        );
        let patches = super::build(&cb, &leg, w(8));
        let mut cb2 = cb;
        super::apply_patches(&mut cb2, patches);

        assert_eq!(cb2.mul_constraints.len(), 1);
        let m = &cb2.mul_constraints[0];
        assert_operand_eq(
            &m.hi,
            vec![ShiftedWire {
                wire: w(1),
                shift: Shift::Sll(20),
            }],
            "mul.hi (committed)",
        );
        assert_operand_eq(
            &m.lo,
            vec![
                ShiftedWire {
                    wire: w(3),
                    shift: Shift::None,
                },
                ShiftedWire {
                    wire: w(4),
                    shift: Shift::None,
                },
            ],
            "mul.lo (inlinable)",
        );
    }
}
