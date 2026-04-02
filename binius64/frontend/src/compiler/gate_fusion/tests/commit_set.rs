// Copyright 2025 Irreducible Inc.
use crate::compiler::{
    Wire,
    constraint_builder::{ConstraintBuilder, rotr, sar, sll, srl, xor2, xor3},
    gate_fusion::{Stat, commit_set, legraph::LeGraph},
};

/// Test helper to create a Wire with a given ID
fn w(id: u32) -> Wire {
    Wire::from_u32(id)
}

/// Test helper to build a simple test circuit and verify the commit set
fn test_commit_set(
    build_constraints: impl FnOnce(&mut ConstraintBuilder),
    expected_committed: &[Wire],
    expected_not_committed: &[Wire],
) {
    let mut cb = ConstraintBuilder::new();
    build_constraints(&mut cb);

    let mut stat = Stat::default();
    let mut leg = LeGraph::new(&cb, &mut stat);
    commit_set::run_decide_commit_set(&mut leg, &mut stat);
    let commit_set = leg.commit_set();

    // Verify expected wires are committed
    for &wire in expected_committed {
        assert!(
            commit_set.contains(wire),
            "Wire {:?} should be committed but wasn't. Commit set: {:?}",
            wire,
            commit_set
        );
    }

    // Verify expected wires are NOT committed (i.e., can be inlined)
    for &wire in expected_not_committed {
        assert!(
            !commit_set.contains(wire),
            "Wire {:?} should NOT be committed but was. Commit set: {:?}",
            wire,
            commit_set
        );
    }
}

#[test]
fn test_simple_xor_inlining() {
    // Test: y = x ^ a, z = y ^ b, then z is used in AND
    // Both y and z should be inlinable into the AND constraint
    test_commit_set(
        |cb| {
            // y = x ^ a
            cb.linear().rhs(xor2(w(0), w(1))).dst(w(2)).build();
            // z = y ^ b
            cb.linear().rhs(xor2(w(2), w(3))).dst(w(4)).build();
            // Use z in AND constraint (creates the root)
            cb.and().a(w(4)).b(w(5)).c(w(6)).build();
        },
        &[],           // Nothing should be committed
        &[w(2), w(4)], // Both linear defs can be inlined
    );
}

#[test]
fn test_xor_used_in_and_constraint() {
    // Test: y = x ^ a, and(y, b, c)
    // y should be inlinable into the AND constraint
    test_commit_set(
        |cb| {
            // y = x ^ a
            cb.linear().rhs(xor2(w(0), w(1))).dst(w(2)).build();
            // and(y, b, c)
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[],     // y can be inlined
        &[w(2)], // y should not be committed
    );
}

#[test]
fn test_shift_composition_same_type() {
    // Test: y = x << 10, z = y << 20
    // Shifts should compose: z = x << 30
    test_commit_set(
        |cb| {
            // y = x << 10
            cb.linear().rhs(sll(w(0), 10)).dst(w(1)).build();
            // z = y << 20
            cb.linear().rhs(sll(w(1), 20)).dst(w(2)).build();
            // Use z in an AND constraint so it becomes a root
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[], // Both shifts can be composed and inlined
        &[w(1), w(2)],
    );
}

#[test]
fn test_srl_composition() {
    // Test: y = x >> 10, z = y >> 20
    // Shifts should compose: z = x >> 30
    test_commit_set(
        |cb| {
            // y = x >> 10
            cb.linear().rhs(srl(w(0), 10)).dst(w(1)).build();
            // z = y >> 20
            cb.linear().rhs(srl(w(1), 20)).dst(w(2)).build();
            // Use z in an AND constraint so it becomes a root
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[], // Both shifts can be composed and inlined
        &[w(1), w(2)],
    );
}

#[test]
fn test_sar_composition() {
    // y = sar(x, 31), z = sar(y, 1) -> compose to sar(x, 32), within range
    test_commit_set(
        |cb| {
            cb.linear().rhs(sar(w(0), 31)).dst(w(1)).build();
            cb.linear().rhs(sar(w(1), 1)).dst(w(2)).build();
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[], // inlinable
        &[w(1), w(2)],
    );
}

#[test]
fn test_sar_incompatible_with_srl() {
    // y = sar(x, 7), z = srl(y, 1) -> incompatible; y must be committed
    test_commit_set(
        |cb| {
            cb.linear().rhs(sar(w(0), 7)).dst(w(1)).build();
            cb.linear().rhs(srl(w(1), 1)).dst(w(2)).build();
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[w(1)], // commit producer
        &[w(2)], // z can inline its (now committed) input
    );
}

#[test]
fn test_sar_incompatible_with_sll() {
    // y = sar(x, 5), z = sll(y, 2) -> incompatible; y must be committed
    test_commit_set(
        |cb| {
            cb.linear().rhs(sar(w(0), 5)).dst(w(1)).build();
            cb.linear().rhs(sll(w(1), 2)).dst(w(2)).build();
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[w(1)], // commit producer
        &[w(2)],
    );
}

#[test]
fn test_sar_incompatible_with_rotr() {
    // y = sar(x, 5), z = rotr(y, 10) -> incompatible types; y must be committed
    test_commit_set(
        |cb| {
            cb.linear().rhs(sar(w(0), 5)).dst(w(1)).build();
            cb.linear().rhs(rotr(w(1), 10)).dst(w(2)).build();
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[w(1)],
        &[w(2)],
    );
}

#[test]
fn test_all_or_nothing_across_and_and_mul() {
    // x = sll(a, 20)
    // y = x ^ c (OK to inline)
    // z = srl(x, 5) (incompatible with sll)
    // Use y in AND and z in MUL -> x must be committed due to mixed uses
    test_commit_set(
        |cb| {
            cb.linear().rhs(sll(w(0), 20)).dst(w(1)).build(); // x
            cb.linear().rhs(xor2(w(1), w(2))).dst(w(3)).build(); // y = x ^ c
            cb.linear().rhs(srl(w(1), 5)).dst(w(4)).build(); // z = x >> 5
            cb.and().a(w(3)).b(w(5)).c(w(6)).build();
            cb.mul().a(w(4)).b(w(7)).hi(w(8)).lo(w(9)).build();
        },
        &[w(1)],       // x must be committed
        &[w(3), w(4)], // y and z can inline their inputs (subject to x being committed)
    );
}

#[test]
fn test_sll_boundary_63_vs_64() {
    // Compose to 63 -> OK; compose to 64 -> commit
    // Case 1: 32 + 31 = 63
    test_commit_set(
        |cb| {
            cb.linear().rhs(sll(w(0), 32)).dst(w(1)).build();
            cb.linear().rhs(sll(w(1), 31)).dst(w(2)).build();
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[],
        &[w(1), w(2)],
    );

    // Case 2: 32 + 32 = 64 -> commit first
    test_commit_set(
        |cb| {
            cb.linear().rhs(sll(w(0), 32)).dst(w(5)).build();
            cb.linear().rhs(sll(w(5), 32)).dst(w(6)).build();
            cb.and().a(w(6)).b(w(7)).c(w(8)).build();
        },
        &[w(5)],
        &[w(6)],
    );
}

#[test]
fn test_srl_boundary_63_vs_64() {
    // Case 1: 16 + 47 = 63 -> OK
    test_commit_set(
        |cb| {
            cb.linear().rhs(srl(w(0), 16)).dst(w(1)).build();
            cb.linear().rhs(srl(w(1), 47)).dst(w(2)).build();
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[],
        &[w(1), w(2)],
    );

    // Case 2: 48 + 16 = 64 -> commit first
    test_commit_set(
        |cb| {
            cb.linear().rhs(srl(w(0), 48)).dst(w(5)).build();
            cb.linear().rhs(srl(w(5), 16)).dst(w(6)).build();
            cb.and().a(w(6)).b(w(7)).c(w(8)).build();
        },
        &[w(5)],
        &[w(6)],
    );
}

#[test]
fn test_sar_boundary_63_vs_64() {
    // Case 1: 40 + 23 = 63 -> OK
    test_commit_set(
        |cb| {
            cb.linear().rhs(sar(w(0), 40)).dst(w(1)).build();
            cb.linear().rhs(sar(w(1), 23)).dst(w(2)).build();
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[],
        &[w(1), w(2)],
    );

    // Case 2: 32 + 32 = 64 -> commit first
    test_commit_set(
        |cb| {
            cb.linear().rhs(sar(w(0), 32)).dst(w(5)).build();
            cb.linear().rhs(sar(w(5), 32)).dst(w(6)).build();
            cb.and().a(w(6)).b(w(7)).c(w(8)).build();
        },
        &[w(5)],
        &[w(6)],
    );
}

#[test]
fn test_zero_shift_composition() {
    // Zero shifts still carry their type; mixed types do not compose → commit
    test_commit_set(
        |cb| {
            // y = sll(x, 0)
            cb.linear().rhs(sll(w(0), 0)).dst(w(1)).build();
            // z = srl(y, 0)
            cb.linear().rhs(srl(w(1), 0)).dst(w(2)).build();
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[w(1)], // y must be committed (Sll vs Srl are incompatible)
        &[w(2)],
    );

    // Same-type zero shifts compose trivially
    test_commit_set(
        |cb| {
            // y = srl(x, 0)
            cb.linear().rhs(srl(w(0), 0)).dst(w(1)).build();
            // z = srl(y, 0)
            cb.linear().rhs(srl(w(1), 0)).dst(w(2)).build();
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[],
        &[w(1), w(2)],
    );
}

#[test]
fn test_rotr_zero_inlining() {
    // y = a ^ b; z = rotr(y, 0); use z in AND. Both y and z should be inlinable.
    test_commit_set(
        |cb| {
            cb.linear().rhs(xor2(w(0), w(1))).dst(w(2)).build(); // y
            cb.linear().rhs(rotr(w(2), 0)).dst(w(3)).build(); // z
            cb.and().a(w(3)).b(w(4)).c(w(5)).build();
        },
        &[],
        &[w(2), w(3)],
    );
}

#[test]
fn test_diamond_fanout_inlining() {
    // P = x ^ y
    // Q = P ^ c
    // R = P ^ d
    // S = Q ^ R
    // Use S in AND -> Expect P,Q,R,S all inlinable (no shifts)
    test_commit_set(
        |cb| {
            cb.linear().rhs(xor2(w(0), w(1))).dst(w(2)).build(); // P
            cb.linear().rhs(xor2(w(2), w(3))).dst(w(4)).build(); // Q
            cb.linear().rhs(xor2(w(2), w(5))).dst(w(6)).build(); // R
            cb.linear().rhs(xor2(w(4), w(6))).dst(w(7)).build(); // S
            cb.and().a(w(7)).b(w(8)).c(w(9)).build();
        },
        &[],
        &[w(2), w(4), w(6), w(7)],
    );
}

#[test]
fn test_shift_composition_different_types() {
    // Test: y = x << 10, z = y >> 20
    // Different shift types cannot compose, y must be committed
    test_commit_set(
        |cb| {
            // y = x << 10
            cb.linear().rhs(sll(w(0), 10)).dst(w(1)).build();
            // z = y >> 20
            cb.linear().rhs(srl(w(1), 20)).dst(w(2)).build();
            // Use z in an AND constraint
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[w(1)], // y must be committed (incompatible shifts)
        &[w(2)], // z can still be inlined
    );
}

#[test]
fn test_rotr_distributes_over_xor() {
    // Test: y = a ^ b, z = rotr(y, 5)
    // Should distribute: z = rotr(a, 5) ^ rotr(b, 5)
    test_commit_set(
        |cb| {
            // y = a ^ b
            cb.linear().rhs(xor2(w(0), w(1))).dst(w(2)).build();
            // z = rotr(y, 5)
            cb.linear().rhs(rotr(w(2), 5)).dst(w(3)).build();
            // Use z in an AND constraint
            cb.and().a(w(3)).b(w(4)).c(w(5)).build();
        },
        &[], // Both can be inlined (rotr distributes over xor)
        &[w(2), w(3)],
    );
}

#[test]
fn test_rotr_distributes_over_multi_xor() {
    // Test: y = a ^ b ^ c, z = rotr(y, 7)
    // Should distribute: z = rotr(a, 7) ^ rotr(b, 7) ^ rotr(c, 7)
    test_commit_set(
        |cb| {
            // y = a ^ b ^ c
            cb.linear().rhs(xor3(w(0), w(1), w(2))).dst(w(3)).build();
            // z = rotr(y, 7)
            cb.linear().rhs(rotr(w(3), 7)).dst(w(4)).build();
            // Use z in an AND constraint
            cb.and().a(w(4)).b(w(5)).c(w(6)).build();
        },
        &[], // Both can be inlined (rotr distributes over xor)
        &[w(3), w(4)],
    );
}

#[test]
fn test_incompatible_shift_sequence() {
    // Test: y = a >> 10, z = y << 5
    // Different shift types in sequence cannot compose (srl then sll)
    test_commit_set(
        |cb| {
            // y = a >> 10
            cb.linear().rhs(srl(w(0), 10)).dst(w(2)).build();
            // z = y << 5
            cb.linear().rhs(sll(w(2), 5)).dst(w(3)).build();
            // Use z in an AND constraint
            cb.and().a(w(3)).b(w(4)).c(w(5)).build();
        },
        &[w(2)], // y must be committed (incompatible shift sequence)
        &[w(3)], // z can still be inlined
    );
}

#[test]
fn test_multiple_uses_all_or_nothing() {
    // Test: x = a << 20, y = x ^ c, z = x >> 5
    // x is used in both y and z. Since we have x shifted left,
    // and z tries to shift it right, these are incompatible shift types.
    // Therefore x must be committed (all-or-nothing principle)
    test_commit_set(
        |cb| {
            // x = a << 20
            cb.linear().rhs(sll(w(0), 20)).dst(w(2)).build();
            // y = x ^ c (composable - shift can distribute over XOR)
            cb.linear().rhs(xor2(w(2), w(3))).dst(w(4)).build();
            // z = x >> 5 (incompatible - can't compose sll with srl)
            cb.linear().rhs(srl(w(2), 5)).dst(w(5)).build();
            // Use y and z in AND constraints
            cb.and().a(w(4)).b(w(6)).c(w(7)).build();
            cb.and().a(w(5)).b(w(8)).c(w(9)).build();
        },
        &[w(2)],       // x must be committed (incompatible shift types)
        &[w(4), w(5)], // y and z can be inlined
    );
}

#[test]
fn test_fixed_point_iteration() {
    // Test: a = input1 ^ input2
    //       b = a >> 10 (srl shift)
    //       c = b ^ input4
    //       d = b << 5 (sll shift - incompatible with srl)
    // b has incompatible uses (used with both XOR and incompatible shift)
    test_commit_set(
        |cb| {
            // a = input1 ^ input2
            cb.linear().rhs(xor2(w(0), w(1))).dst(w(2)).build();
            // b = a >> 10
            cb.linear().rhs(srl(w(2), 10)).dst(w(4)).build();
            // c = b ^ input4
            cb.linear().rhs(xor2(w(4), w(5))).dst(w(6)).build();
            // d = b << 5 (incompatible - can't compose srl with sll)
            cb.linear().rhs(sll(w(4), 5)).dst(w(7)).build();
            // Use c and d in AND constraints
            cb.and().a(w(6)).b(w(8)).c(w(9)).build();
            cb.and().a(w(7)).b(w(10)).c(w(11)).build();
        },
        &[w(4)],             // b must be committed (incompatible shift types)
        &[w(2), w(6), w(7)], // a, c, and d can be inlined
    );
}

#[test]
fn test_rotr_composition() {
    // Test: y = rotr(x, 10), z = rotr(y, 15)
    // Should compose to: z = rotr(x, 25)
    test_commit_set(
        |cb| {
            // y = rotr(x, 10)
            cb.linear().rhs(rotr(w(0), 10)).dst(w(1)).build();
            // z = rotr(y, 15)
            cb.linear().rhs(rotr(w(1), 15)).dst(w(2)).build();
            // Use z in AND constraint
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[], // Both rotations can compose
        &[w(1), w(2)],
    );
}

#[test]
fn test_complex_xor_chain() {
    // Test: y = a ^ b ^ c, z = y ^ d ^ e
    // Both should be inlinable
    test_commit_set(
        |cb| {
            // y = a ^ b ^ c
            cb.linear().rhs(xor3(w(0), w(1), w(2))).dst(w(3)).build();
            // z = y ^ d ^ e
            cb.linear().rhs(xor3(w(3), w(4), w(5))).dst(w(6)).build();
            // Use z in AND constraint
            cb.and().a(w(6)).b(w(7)).c(w(8)).build();
        },
        &[], // All can be inlined
        &[w(3), w(6)],
    );
}

#[test]
fn test_wire_used_in_mul_constraint() {
    // Test: y = x ^ a, mul(y, b) = hi:lo
    // y should be inlinable into the MUL constraint
    test_commit_set(
        |cb| {
            // y = x ^ a
            cb.linear().rhs(xor2(w(0), w(1))).dst(w(2)).build();
            // mul(y, b) = hi:lo
            cb.mul().a(w(2)).b(w(3)).hi(w(4)).lo(w(5)).build();
        },
        &[],     // y can be inlined
        &[w(2)], // y should not be committed
    );
}

#[test]
fn test_shifted_wire_in_non_linear_use() {
    // Test: y = x >> 5, and(y, a, b)
    // Since y is already shifted and used in non-linear constraint,
    // we need to be careful about inlining
    test_commit_set(
        |cb| {
            // y = x >> 5
            cb.linear().rhs(srl(w(0), 5)).dst(w(1)).build();
            // and(y, a, b)
            cb.and().a(w(1)).b(w(2)).c(w(3)).build();
        },
        &[], // Simple shift can be inlined
        &[w(1)],
    );
}

#[test]
fn test_multiple_non_linear_uses() {
    // Test: y = x ^ a, and(y, b, c), and(y, d, e)
    // y used in multiple AND constraints - should still be inlinable
    test_commit_set(
        |cb| {
            // y = x ^ a
            cb.linear().rhs(xor2(w(0), w(1))).dst(w(2)).build();
            // and(y, b, c)
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
            // and(y, d, e)
            cb.and().a(w(2)).b(w(5)).c(w(6)).build();
        },
        &[], // y can be inlined into both AND constraints
        &[w(2)],
    );
}

#[test]
fn test_deep_xor_tree() {
    // Test a deeper tree of XOR operations
    // a = x ^ y
    // b = z ^ w
    // c = a ^ b
    // All should be inlinable
    test_commit_set(
        |cb| {
            // a = x ^ y
            cb.linear().rhs(xor2(w(0), w(1))).dst(w(2)).build();
            // b = z ^ w
            cb.linear().rhs(xor2(w(3), w(4))).dst(w(5)).build();
            // c = a ^ b
            cb.linear().rhs(xor2(w(2), w(5))).dst(w(6)).build();
            // Use c in AND constraint
            cb.and().a(w(6)).b(w(7)).c(w(8)).build();
        },
        &[], // All can be inlined
        &[w(2), w(5), w(6)],
    );
}

#[test]
fn test_shift_overflow_prevention() {
    // Test: y = x << 40, z = y << 30
    // Combined shift would be 70, which exceeds 64 bits
    // y must be committed
    test_commit_set(
        |cb| {
            // y = x << 40
            cb.linear().rhs(sll(w(0), 40)).dst(w(1)).build();
            // z = y << 30
            cb.linear().rhs(sll(w(1), 30)).dst(w(2)).build();
            // Use z in AND constraint
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[w(1)], // y must be committed (shift overflow)
        &[w(2)], // z can still be inlined
    );
}

#[test]
fn test_rotr_wraps_correctly() {
    // Test: y = rotr(x, 50), z = rotr(y, 30)
    // Combined rotation should be (50 + 30) % 64 = 16
    // Both should be inlinable
    test_commit_set(
        |cb| {
            // y = rotr(x, 50)
            cb.linear().rhs(rotr(w(0), 50)).dst(w(1)).build();
            // z = rotr(y, 30)
            cb.linear().rhs(rotr(w(1), 30)).dst(w(2)).build();
            // Use z in AND constraint
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[], // Both can be composed (rotation wraps)
        &[w(1), w(2)],
    );
}

#[test]
fn test_rotr_large_composition() {
    // Test: y = rotr(x, 63), z = rotr(y, 63)
    // Combined rotation should be (63 + 63) % 64 = 62
    // Both should be inlinable
    test_commit_set(
        |cb| {
            // y = rotr(x, 63)
            cb.linear().rhs(rotr(w(0), 63)).dst(w(1)).build();
            // z = rotr(y, 63)
            cb.linear().rhs(rotr(w(1), 63)).dst(w(2)).build();
            // Use z in AND constraint
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[], // Both can be composed (rotation wraps at 64)
        &[w(1), w(2)],
    );
}

#[test]
fn test_no_linear_defs() {
    // Test with only AND constraints, no linear constraints
    test_commit_set(
        |cb| {
            // Just AND constraints, no linear defs
            cb.and().a(w(0)).b(w(1)).c(w(2)).build();
            cb.and().a(w(3)).b(w(4)).c(w(5)).build();
        },
        &[], // Nothing to commit
        &[], // No linear defs to inline
    );
}

#[test]
fn test_linear_def_no_uses() {
    // Test: y = x ^ a, but y is never used
    // Unused linear defs don't need to be committed
    test_commit_set(
        |cb| {
            // y = x ^ a (unused)
            cb.linear().rhs(xor2(w(0), w(1))).dst(w(2)).build();
            // Some other AND constraint
            cb.and().a(w(3)).b(w(4)).c(w(5)).build();
        },
        &[],     // Unused def doesn't need committing
        &[w(2)], // Not committed
    );
}

#[test]
fn test_mixed_shift_in_xor() {
    // Test: y = (x << 5) ^ (z >> 3), used in AND
    // The operand has mixed shifts, but they're at the term level
    test_commit_set(
        |cb| {
            // y = (x << 5) ^ (z >> 3)
            cb.linear()
                .rhs(xor2(sll(w(0), 5), srl(w(1), 3)))
                .dst(w(2))
                .build();
            // and(y, a, b)
            cb.and().a(w(2)).b(w(3)).c(w(4)).build();
        },
        &[], // Can be inlined (shifts are on individual terms)
        &[w(2)],
    );
}

#[test]
fn test_recursive_commit_propagation() {
    // Test: a = input >> 15 (srl)
    //       b = a ^ input3
    //       c = a << 10 (sll - incompatible with srl)
    // a has incompatible uses (XOR in b, incompatible shift in c)
    test_commit_set(
        |cb| {
            // a = input >> 15
            cb.linear().rhs(srl(w(0), 15)).dst(w(2)).build();
            // b = a ^ input3
            cb.linear().rhs(xor2(w(2), w(3))).dst(w(4)).build();
            // c = a << 10 (incompatible - can't compose srl with sll)
            cb.linear().rhs(sll(w(2), 10)).dst(w(5)).build();
            // d = b ^ input4
            cb.linear().rhs(xor2(w(4), w(6))).dst(w(7)).build();
            // Use c and d in AND constraints
            cb.and().a(w(5)).b(w(8)).c(w(9)).build();
            cb.and().a(w(7)).b(w(10)).c(w(11)).build();
        },
        &[w(2)],             // a must be committed (incompatible uses)
        &[w(4), w(5), w(7)], // b, c, and d can be inlined
    );
}

#[test]
fn test_rotr_with_unshifted_xor_terms() {
    // Test the specific bug we fixed: rotr(a ^ b, n) where a and b are unshifted
    // This tests that Rotr(n) composes correctly with None (unshifted terms)
    test_commit_set(
        |cb| {
            // y = a ^ b (both unshifted)
            cb.linear().rhs(xor2(w(0), w(1))).dst(w(2)).build();
            // z = rotr(y, 63)
            cb.linear().rhs(rotr(w(2), 63)).dst(w(3)).build();
            // Use z in an AND constraint
            cb.and().a(w(3)).b(w(4)).c(w(5)).build();
        },
        &[], // Everything should be inlinable - rotr distributes over unshifted XOR
        &[w(2), w(3)],
    );
}

#[test]
fn test_rotr_with_mixed_shift_xor() {
    // Test: y = a ^ (b << 5), z = rotr(y, 10)
    // When we try to inline with rotr(10), the Sll(5) is incompatible
    test_commit_set(
        |cb| {
            // b_shifted = b << 5
            cb.linear().rhs(sll(w(1), 5)).dst(w(6)).build();
            // y = a ^ b_shifted (a is unshifted, b_shifted has Sll)
            cb.linear().rhs(xor2(w(0), w(6))).dst(w(2)).build();
            // z = rotr(y, 10)
            cb.linear().rhs(rotr(w(2), 10)).dst(w(3)).build();
            // Use z in an AND constraint
            cb.and().a(w(3)).b(w(4)).c(w(5)).build();
        },
        &[w(6)],       // b_shifted must be committed (can't compose Rotr with Sll)
        &[w(2), w(3)], // y and z can still be inlined
    );
}
