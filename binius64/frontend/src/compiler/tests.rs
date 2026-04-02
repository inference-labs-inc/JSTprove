// Copyright 2025 Irreducible Inc.
use binius_core::{verify::verify_constraints, word::Word};
use proptest::prelude::*;
use rand::{Rng, SeedableRng as _, rngs::StdRng};

use super::*;
use crate::util::num_biguint_from_u64_limbs;

#[test]
fn test_icmp_ult() {
    // Build a circuit with only two inputs and check c = a < b.
    let builder = CircuitBuilder::new();
    let a = builder.add_inout();
    let b = builder.add_inout();
    let actual = builder.icmp_ult(a, b);
    let expected = builder.add_inout();
    builder.assert_false("lt", builder.bxor(actual, expected));
    let circuit = builder.build();

    // check that it actually works.
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..10000 {
        let mut w = circuit.new_witness_filler();
        w[a] = Word(rng.random::<u64>());
        w[b] = Word(rng.random::<u64>());
        w[expected] = Word(if w[a].0 < w[b].0 { u64::MAX } else { 0 });
        w.circuit.populate_wire_witness(&mut w).unwrap();
    }
}

#[test]
fn test_icmp_eq() {
    // Build a circuit with only two inputs and check c = a == b.
    let builder = CircuitBuilder::new();
    let a = builder.add_inout();
    let b = builder.add_inout();
    let actual = builder.icmp_eq(a, b);
    let expected = builder.add_inout();
    builder.assert_false("eq", builder.bxor(actual, expected));
    let circuit = builder.build();

    // check that it actually works.
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..10000 {
        let mut w = circuit.new_witness_filler();
        w[a] = Word(rng.random::<u64>());
        w[b] = Word(rng.random::<u64>());
        w[expected] = Word(if w[a].0 == w[b].0 { u64::MAX } else { 0 });
        w.circuit.populate_wire_witness(&mut w).unwrap();
    }
}

#[test]
fn test_iadd_cin_cout_max_values() {
    let builder = CircuitBuilder::new();

    let a = builder.add_constant_64(0xFFFFFFFFFFFFFFFF);
    let b = builder.add_constant_64(0xFFFFFFFFFFFFFFFF);
    let cin_wire = builder.add_constant(Word::ZERO);
    let (sum_wire, cout_wire) = builder.iadd_cin_cout(a, b, cin_wire);

    let circuit = builder.build();
    let mut w = circuit.new_witness_filler();
    circuit.populate_wire_witness(&mut w).unwrap();

    assert_eq!(w[sum_wire], Word(0xFFFFFFFFFFFFFFFE));
    assert_eq!(w[cout_wire], Word(0xFFFFFFFFFFFFFFFF));
}

#[test]
fn test_iadd_cin_cout_zero() {
    let builder = CircuitBuilder::new();

    let a = builder.add_constant_64(0);
    let b = builder.add_constant_64(0);
    let cin_wire = builder.add_constant(Word::ZERO);
    let (sum_wire, cout_wire) = builder.iadd_cin_cout(a, b, cin_wire);

    let circuit = builder.build();
    let mut w = circuit.new_witness_filler();
    circuit.populate_wire_witness(&mut w).unwrap();

    assert_eq!(w[sum_wire], Word(0));
    assert_eq!(w[cout_wire], Word(0));
}

#[test]
fn test_isub_bin_bout_from_zero() {
    let builder = CircuitBuilder::new();

    let a = builder.add_constant_64(0);
    let b = builder.add_constant_64(u64::MAX);
    let bin_wire = builder.add_constant(Word::ONE << 63);
    let (diff_wire, bout_wire) = builder.isub_bin_bout(a, b, bin_wire);

    let circuit = builder.build();
    let mut w = circuit.new_witness_filler();
    circuit.populate_wire_witness(&mut w).unwrap();

    assert_eq!(w[diff_wire], Word(0));
    assert_eq!(w[bout_wire], Word(u64::MAX));
}

#[test]
fn test_biguint_divide_hint() {
    let builder = CircuitBuilder::new();

    // (2^128-1) % (2^64-5) = 24
    let d0 = builder.add_constant_64(u64::MAX);
    let d1 = builder.add_constant_64(u64::MAX);

    let m = builder.add_constant_64(u64::MAX - 4);

    let (q, r) = builder.biguint_divide_hint(&[d0, d1], &[m]);

    let circuit = builder.build();
    let mut w = circuit.new_witness_filler();
    circuit.populate_wire_witness(&mut w).unwrap();

    assert_eq!(r.len(), 1);
    assert_eq!(w[r[0]], Word(24));

    assert_eq!(q.len(), 2);
    assert_eq!(w[q[0]], Word(5));
    assert_eq!(w[q[1]], Word(1));
}

#[test]
fn test_biguint_divide_hint_div_by_zero() {
    let builder = CircuitBuilder::new();

    let d0 = builder.add_constant_64(u64::MAX);
    let d1 = builder.add_constant_64(u64::MAX);

    let m0 = builder.add_constant_64(0);
    let m1 = builder.add_constant_64(0);

    let (q, r) = builder.biguint_divide_hint(&[d0, d1], &[m0, m1]);

    let circuit = builder.build();
    let mut w = circuit.new_witness_filler();
    circuit.populate_wire_witness(&mut w).unwrap();

    assert_eq!(r.len(), 2);
    assert_eq!(w[r[0]], Word(0));
    assert_eq!(w[r[1]], Word(0));

    assert_eq!(q.len(), 2);
    assert_eq!(w[q[0]], Word(0));
    assert_eq!(w[q[1]], Word(0));
}

#[test]
fn test_mod_pow_hint() {
    let builder = CircuitBuilder::new();

    let c = builder.add_constant_64(0x123456789abcdef0);
    let modpow = builder.biguint_mod_pow_hint(&[c], &[c, c], &[c, c, c]);

    let circuit = builder.build();
    let mut w = circuit.new_witness_filler();
    circuit.populate_wire_witness(&mut w).unwrap();

    assert_eq!(modpow.len(), 3);
    assert_eq!(w[modpow[0]], Word(0x6f151e00d2c39f30));
    assert_eq!(w[modpow[1]], Word(0xfef75acc27ead52f));
    assert_eq!(w[modpow[2]], Word(0x00443adf222ea27));
}

#[test]
fn test_mod_inverse_hint() {
    let builder = CircuitBuilder::new();

    let b = builder.add_constant_64(0x123456789abcdef0);

    // M12 = 2^127-1
    let m0 = builder.add_constant_64(u64::MAX);
    let m1 = builder.add_constant_64((1u64 << 63) - 1);

    let (quotient, inverse) = builder.mod_inverse_hint(&[b], &[m0, m1]);

    let circuit = builder.build();
    let mut w = circuit.new_witness_filler();
    circuit.populate_wire_witness(&mut w).unwrap();

    assert_eq!(inverse.len(), 2);
    assert_eq!(w[inverse[0]], Word(0xe5a542e11f99750a));
    assert_eq!(w[inverse[1]], Word(0x1849faf75fbb9752));

    assert_eq!(quotient.len(), 2);
    assert_eq!(w[quotient[0]], Word(0x37455c1554b9aa1));
    assert_eq!(w[quotient[1]], Word::ZERO);
}

#[test]
fn test_mod_inverse_hint_non_coprime() {
    let builder = CircuitBuilder::new();

    let b = builder.add_constant_64((1 << 19) - 1);

    // M7 * M11 = (2^19-1)*(2^107-1)
    let m0 = builder.add_constant_64(0xfffffffffff80001);
    let m1 = builder.add_constant_64(0x3ffff7ffffffffff);

    let (quotient, inverse) = builder.mod_inverse_hint(&[b], &[m0, m1]);

    let circuit = builder.build();
    let mut w = circuit.new_witness_filler();
    circuit.populate_wire_witness(&mut w).unwrap();

    assert_eq!(inverse.len(), 2);
    assert_eq!(w[inverse[0]], Word::ZERO);
    assert_eq!(w[inverse[1]], Word::ZERO);

    assert_eq!(quotient.len(), 2);
    assert_eq!(w[quotient[0]], Word::ZERO);
    assert_eq!(w[quotient[1]], Word::ZERO);
}

fn prop_check_icmp_ult(a: u64, b: u64, expected_result: Word) {
    let builder = CircuitBuilder::new();
    let a_wire = builder.add_constant_64(a);
    let b_wire = builder.add_constant_64(b);
    let result_wire = builder.icmp_ult(a_wire, b_wire);

    let circuit = builder.build();
    let mut w = circuit.new_witness_filler();
    circuit.populate_wire_witness(&mut w).unwrap();

    assert_eq!(w[result_wire] >> 63, expected_result >> 63);

    let cs = circuit.constraint_system();
    verify_constraints(cs, &w.value_vec).unwrap();
}

fn prop_check_icmp_eq(a: u64, b: u64, expected_result: Word) {
    let builder = CircuitBuilder::new();
    let a_wire = builder.add_constant_64(a);
    let b_wire = builder.add_constant_64(b);
    let result_wire = builder.icmp_eq(a_wire, b_wire);

    let circuit = builder.build();
    let mut w = circuit.new_witness_filler();
    circuit.populate_wire_witness(&mut w).unwrap();

    assert_eq!(w[result_wire] >> 63, expected_result >> 63);

    let cs = circuit.constraint_system();
    verify_constraints(cs, &w.value_vec).unwrap();
}

proptest! {
    #[test]
    fn prop_iadd_cin_cout_carry_chain(a1 in any::<u64>(), b1 in any::<u64>(), a2 in any::<u64>(), b2 in any::<u64>()) {
        let builder = CircuitBuilder::new();

        // First addition
        let a1_wire = builder.add_constant_64(a1);
        let b1_wire = builder.add_constant_64(b1);
        let cin_wire = builder.add_constant(Word::ZERO);
        let (sum1_wire, cout1_wire) = builder.iadd_cin_cout(a1_wire, b1_wire, cin_wire);

        // Second addition with carry from first
        let a2_wire = builder.add_constant_64(a2);
        let b2_wire = builder.add_constant_64(b2);
        let (sum2_wire, cout2_wire) = builder.iadd_cin_cout(a2_wire, b2_wire, cout1_wire);

        let circuit = builder.build();
        let mut w = circuit.new_witness_filler();
        circuit.populate_wire_witness(&mut w).unwrap();

        // Check first addition
        let expected_sum1 = a1.wrapping_add(b1);
        let expected_cout1 = (a1 & b1) | ((a1 ^ b1) & !expected_sum1);
        assert_eq!(w[sum1_wire], Word(expected_sum1));
        assert_eq!(w[cout1_wire], Word(expected_cout1));

        // Check second addition with carry
        // Extract MSB of cout1 as the carry-in bit
        let cin2 = expected_cout1 >> 63;
        let expected_sum2 = a2.wrapping_add(b2).wrapping_add(cin2);
        let expected_cout2 = (a2 & b2) | ((a2 ^ b2) & !expected_sum2);
        assert_eq!(w[sum2_wire], Word(expected_sum2));
        assert_eq!(w[cout2_wire], Word(expected_cout2));

        let cs = circuit.constraint_system();
        verify_constraints(cs, &w.value_vec).unwrap();
    }

    #[test]
    fn prop_icmp_ult_gte(a in any::<u64>(), b in any::<u64>()) {
        prop_assume!(a >= b);
        prop_check_icmp_ult(a, b, Word::ZERO);
    }

    #[test]
    fn prop_icmp_ult_lt(a in any::<u64>(), b in any::<u64>()) {
        prop_assume!(a < b);
        prop_check_icmp_ult(a, b, Word::ALL_ONE);
    }

    #[test]
    fn prop_check_assert_eq(x in any::<u64>(), y in any::<u64>()) {
        let builder = CircuitBuilder::new();
        let is_equal = x == y;
        let x_wire = builder.add_inout();
        let y_wire = builder.add_inout();
        builder.assert_eq("eq", x_wire, y_wire);

        let circuit = builder.build();
        let mut w = circuit.new_witness_filler();

        w[x_wire] = Word(x);
        w[y_wire] = Word(y);
        let result = circuit.populate_wire_witness(&mut w);

        if is_equal {
            // When values are equal, witness population should succeed
            assert!(result.is_ok());
            // And constraints should verify
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.value_vec).unwrap();
        } else {
            // When values are not equal, witness population should fail
            assert!(result.is_err());
        }
    }

    #[test]
    fn prop_icmp_eq_equal(a in any::<u64>()) {
        prop_check_icmp_eq(a, a, Word::ALL_ONE);
    }

    #[test]
    fn prop_icmp_eq_not_equal(a in any::<u64>(), b in any::<u64>()) {
        prop_assume!(a != b);
        prop_check_icmp_eq(a, b, Word::ZERO);
    }

    #[test]
    fn prop_secp256k1_endosplit(k in any::<[u64; 4]>()) {
        let modulus = num_bigint::BigUint::from_bytes_be(
            &hex_literal::hex!("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141")
        );
        let lambda =  num_bigint::BigUint::from_bytes_be(
            &hex_literal::hex!("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72")
        );
        let k_bignum = num_biguint_from_u64_limbs(k.iter());
        prop_assume!(k_bignum < modulus);
        prop_assume!(k_bignum > num_bigint::BigUint::ZERO);

        let builder = CircuitBuilder::new();
        let k = k.map(|limb| builder.add_constant_64(limb));
        let (k1_neg, k2_neg, k1_abs, k2_abs) =
            builder.secp256k1_endomorphism_split_hint(&k);

        let circuit = builder.build();
        let mut w = circuit.new_witness_filler();
        circuit.populate_wire_witness(&mut w).unwrap();

        let k1_abs_bignum = num_biguint_from_u64_limbs(k1_abs.iter().map(|&l| &w[l].0));
        let k2_abs_bignum = num_biguint_from_u64_limbs(k2_abs.iter().map(|&l| &w[l].0));

        assert!(k1_abs_bignum.bits() <= 128);
        assert!(k2_abs_bignum.bits() <= 128);

        let k1 = if w[k1_neg] != Word::ZERO {
            &modulus - k1_abs_bignum
        } else {
            k1_abs_bignum
        };

        let k2 = if w[k2_neg] != Word::ZERO {
            &modulus - k2_abs_bignum
        } else {
            k2_abs_bignum
        };

        assert_eq!((k1 + lambda * k2) % modulus, k_bignum);
    }
}

#[test]
fn test_bxor_linear_constraint() {
    // Test that bxor operation internally uses linear constraints
    // which are then expanded to AND constraints with all_one
    let builder = CircuitBuilder::new();

    let a = builder.add_inout();
    let b = builder.add_inout();

    // bxor internally creates a linear constraint
    let c = builder.bxor(a, b);

    let circuit = builder.build();

    // Verify the circuit builds successfully and bxor works correctly
    let mut w = circuit.new_witness_filler();
    w[a] = Word(0x123456789abcdef0);
    w[b] = Word(0xfedcba9876543210);

    circuit.populate_wire_witness(&mut w).unwrap();

    // Verify result is correct
    assert_eq!(w[c], Word(0x123456789abcdef0 ^ 0xfedcba9876543210));

    // Verify constraints are satisfied
    let cs = circuit.constraint_system();
    verify_constraints(cs, &w.value_vec).unwrap();
}

#[test]
fn test_shift_operations_with_linear_constraints() {
    // Test that shift operations (shl, shr, sar) work correctly
    // These operations internally use linear constraints
    let builder = CircuitBuilder::new();

    let a = builder.add_inout();
    let b = builder.add_inout();

    // Test shift left
    let shl_result = builder.shl(a, 8);
    // Test shift right
    let shr_result = builder.shr(b, 16);
    // Combine with XOR
    let combined = builder.bxor(shl_result, shr_result);

    let circuit = builder.build();

    // Test with specific values
    let mut w = circuit.new_witness_filler();
    w[a] = Word(0xff00ff00ff00ff00);
    w[b] = Word(0x0000abcd0000ef12);

    circuit.populate_wire_witness(&mut w).unwrap();

    // Verify results
    assert_eq!(w[shl_result], Word(0xff00ff00ff00ff00 << 8));
    assert_eq!(w[shr_result], Word(0x0000abcd0000ef12 >> 16));
    assert_eq!(
        w[combined],
        Word((0xff00ff00ff00ff00 << 8) ^ (0x0000abcd0000ef12 >> 16))
    );

    // Verify constraints are satisfied
    let cs = circuit.constraint_system();
    verify_constraints(cs, &w.value_vec).unwrap();
}

#[test]
fn test_32bit_half_shift_operations() {
    let builder = CircuitBuilder::new();

    let a = builder.add_inout();
    let sll32_result = builder.sll32(a, 4);
    let srl32_result = builder.srl32(a, 4);
    let sra32_result = builder.sra32(a, 4);
    let rotr32_result = builder.rotr32(a, 4);

    let circuit = builder.build();

    let input = 0x12345678_89abcdef_u64;
    let mut w = circuit.new_witness_filler();
    w[a] = Word(input);

    circuit.populate_wire_witness(&mut w).unwrap();

    let expected_sll32 = Word(input).sll32(4);
    let expected_srl32 = Word(input).srl32(4);
    let expected_sra32 = Word(input).sra32(4);
    let expected_rotr32 = Word(input).rotr32(4);

    assert_eq!(w[sll32_result], expected_sll32);
    assert_eq!(w[srl32_result], expected_srl32);
    assert_eq!(w[sra32_result], expected_sra32);
    assert_eq!(w[rotr32_result], expected_rotr32);

    // These are lane-local operations, so they should differ from the plain 64-bit shifts
    // for inputs where bits would otherwise cross the 32-bit boundary.
    assert_ne!(w[sll32_result], Word(input << 4));
    assert_ne!(w[srl32_result], Word(input >> 4));
    assert_ne!(w[sra32_result], Word(((input as i64) >> 4) as u64));
    assert_ne!(w[rotr32_result], Word(input.rotate_right(4)));

    let cs = circuit.constraint_system();
    verify_constraints(cs, &w.value_vec).unwrap();
}

#[test]
fn test_rotr_operation_expansion() {
    // Test that rotr operation correctly expands to (srl XOR sll)
    // This tests the expansion logic in constraint_builder.rs
    let builder = CircuitBuilder::new();

    let a = builder.add_inout();
    let b = builder.add_inout();

    // rotr internally expands to: (a >> 12) XOR (a << 52)
    let rotr_result = builder.rotr(a, 12);
    let combined = builder.bxor(rotr_result, b);

    let circuit = builder.build();

    // Test with specific values
    let mut w = circuit.new_witness_filler();
    w[a] = Word(0xabcdef1234567890);
    w[b] = Word(0x1111111111111111);

    circuit.populate_wire_witness(&mut w).unwrap();

    // Verify rotr works correctly: rotr(a, 12)
    let expected_rotr = 0xabcdef1234567890u64.rotate_right(12);
    assert_eq!(w[rotr_result], Word(expected_rotr));
    assert_eq!(w[combined], Word(expected_rotr ^ 0x1111111111111111));

    // Verify constraints are satisfied
    let cs = circuit.constraint_system();
    verify_constraints(cs, &w.value_vec).unwrap();
}

#[test]
fn test_multiple_xor_operations() {
    // Test multiple XOR operations that internally use linear constraints
    let builder = CircuitBuilder::new();

    let a = builder.add_inout();
    let b = builder.add_inout();
    let c = builder.add_inout();
    let d = builder.add_inout();

    // Multiple XOR operations, each creating linear constraints
    let result1 = builder.bxor(a, b);
    let result2 = builder.bxor(c, d);
    // Chain XOR operations
    let final_result = builder.bxor(result1, result2);

    let circuit = builder.build();

    // Test with specific values
    let mut w = circuit.new_witness_filler();
    w[a] = Word(0xaaaaaaaaaaaaaaaa);
    w[b] = Word(0x5555555555555555);
    w[c] = Word(0x0f0f0f0f0f0f0f0f);
    w[d] = Word(0xf0f0f0f0f0f0f0f0);

    circuit.populate_wire_witness(&mut w).unwrap();

    // Verify intermediate results
    assert_eq!(w[result1], Word(0xaaaaaaaaaaaaaaaa ^ 0x5555555555555555));
    assert_eq!(w[result2], Word(0x0f0f0f0f0f0f0f0f ^ 0xf0f0f0f0f0f0f0f0));
    assert_eq!(w[final_result], Word(w[result1].0 ^ w[result2].0));

    // Verify constraints are satisfied
    let cs = circuit.constraint_system();
    verify_constraints(cs, &w.value_vec).unwrap();
}

#[test]
fn test_linear_constraint_conversion_to_and() {
    // This test verifies that linear constraints (created by XOR/shift operations)
    // are properly converted to AND constraints during circuit building.
    // The conversion happens in constraint_builder.rs build() method.

    let builder = CircuitBuilder::new();

    // Create a circuit with various operations that generate linear constraints
    let a = builder.add_inout();
    let b = builder.add_inout();

    // These operations create linear constraints internally:
    let xor_result = builder.bxor(a, b);
    let shift_left = builder.shl(a, 5);
    let shift_right = builder.shr(b, 10);
    let sar_result = builder.sar(a, 3);
    let rotr_result = builder.rotr(b, 7);

    // Combine some results
    let combined1 = builder.bxor(shift_left, shift_right);
    let combined2 = builder.bxor(sar_result, rotr_result);
    let final_result = builder.bxor(combined1, combined2);

    let circuit = builder.build();

    // Get the constraint system which should have AND constraints
    // (linear constraints were converted to AND constraints)
    let cs = circuit.constraint_system();

    // The circuit should have AND constraints but no separate linear constraints
    // (they were all converted during build)
    assert!(
        !cs.and_constraints.is_empty(),
        "Should have AND constraints from converted linear constraints"
    );

    // Test with values to ensure correctness
    let mut w = circuit.new_witness_filler();
    w[a] = Word(0xdeadbeefcafe1234);
    w[b] = Word(0x1234567890abcdef);

    circuit.populate_wire_witness(&mut w).unwrap();

    // Verify all operations computed correctly
    assert_eq!(w[xor_result], Word(0xdeadbeefcafe1234 ^ 0x1234567890abcdef));
    assert_eq!(w[shift_left], Word(0xdeadbeefcafe1234 << 5));
    assert_eq!(w[shift_right], Word(0x1234567890abcdef >> 10));
    assert_eq!(
        w[sar_result],
        Word(((0xdeadbeefcafe1234u64 as i64) >> 3) as u64)
    );
    assert_eq!(w[rotr_result], Word(0x1234567890abcdef_u64.rotate_right(7)));

    // Verify final results
    assert_eq!(w[combined1], Word(w[shift_left].0 ^ w[shift_right].0));
    assert_eq!(w[combined2], Word(w[sar_result].0 ^ w[rotr_result].0));
    assert_eq!(w[final_result], Word(w[combined1].0 ^ w[combined2].0));

    // Verify all constraints are satisfied
    verify_constraints(cs, &w.value_vec).unwrap();
}

proptest! {
    #[test]
    fn prop_xor_operations_with_shifts(a: u64, b: u64, shift1: u32, shift2: u32) {
        // Limit shifts to 0-63
        let shift1 = shift1 % 64;
        let shift2 = shift2 % 64;

        // Test that XOR operations with shifts work correctly
        let builder = CircuitBuilder::new();

        let wire_a = builder.add_constant_64(a);
        let wire_b = builder.add_constant_64(b);

        // Create shifted values
        let shifted_a = builder.shl(wire_a, shift1);
        let shifted_b = builder.shr(wire_b, shift2);

        // XOR the shifted values
        let result = builder.bxor(shifted_a, shifted_b);

        let circuit = builder.build();
        let mut w = circuit.new_witness_filler();
        circuit.populate_wire_witness(&mut w).unwrap();

        // Verify the result is computed correctly
        let expected = (a << shift1) ^ (b >> shift2);
        assert_eq!(w[result], Word(expected));

        // Verify constraints are satisfied
        let cs = circuit.constraint_system();
        verify_constraints(cs, &w.value_vec).unwrap();
    }

    #[test]
    fn prop_rotr_operation(value: u64, shift: u32) {
        // Limit shift to 0-63
        let shift = shift % 64;

        // Test that rotr operation works correctly
        let builder = CircuitBuilder::new();

        let wire_value = builder.add_constant_64(value);
        let rotr_result = builder.rotr(wire_value, shift);

        let circuit = builder.build();
        let mut w = circuit.new_witness_filler();
        circuit.populate_wire_witness(&mut w).unwrap();

        // Verify rotr is computed correctly
        let expected = value.rotate_right(shift);
        assert_eq!(w[rotr_result], Word(expected));

        // Verify constraints are satisfied
        let cs = circuit.constraint_system();
        verify_constraints(cs, &w.value_vec).unwrap();
    }
}
