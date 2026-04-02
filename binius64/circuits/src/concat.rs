// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};

use crate::{
    fixed_byte_vec::ByteVec,
    slice::{create_byte_mask, extract_word},
};

/// Verifies that a joined string is the concatenation of a list of terms.
///
/// This circuit validates that `joined` contains exactly the concatenation
/// of all provided terms in order.
pub struct Concat {
    /// The actual length of the concatenated result in bytes.
    ///
    /// This wire will be constrained to equal the sum of all term lengths.
    pub len_joined_bytes: Wire,
    /// The concatenated data packed as 64-bit words.
    ///
    /// Each wire contains 8 bytes in little-endian order.
    /// The circuit will check / enforce whether the total length of the concatenated data
    /// fits in `joined.len()` wires, i.e. `joined.len() << 3` bytes.
    pub joined: Vec<Wire>,
    /// The list of terms to be concatenated.
    ///
    /// Terms are concatenated in order: terms\[0\] || terms\[1\] || ... || terms\[n-1\]
    pub terms: Vec<ByteVec>,
}

impl Concat {
    /// Creates a new concatenation verifier circuit.
    ///
    /// # Arguments
    /// * `b` - Circuit builder for constructing constraints
    /// * `len_joined` - Wire containing the actual joined size in bytes
    /// * `joined` - Joined array packed as 64-bit words (8 bytes per word)
    /// * `terms` - Vector of terms that should concatenate to form `joined`
    pub fn new(
        b: &CircuitBuilder,
        len_joined_bytes: Wire,
        joined: Vec<Wire>,
        terms: Vec<ByteVec>,
    ) -> Self {
        // Input validation
        //
        // Ensure all inputs meet the word-alignment requirements necessary for
        // efficient word-level processing.

        // Algorithm overview
        //
        // Process terms sequentially, maintaining a running offset to track position
        // in the joined array. For each term, verify its data appears at the correct
        // location.
        //
        // The algorithm:
        // 1. Start with offset = 0
        // 2. For each term: a. Verify its data matches joined[offset : offset + term.len] b. Update
        //    offset += term.len
        // 3. Verify final offset equals total length
        //
        // Circuit constraints:
        // - No dynamic array indexing (use multiplexers instead)
        // - All operations must be on fixed-size data (hence word-level processing)
        // - Conditional operations use masking (condition & value)

        let mut offset = b.add_constant(Word::ZERO);

        // 1. Sequential term processing
        //
        // Process each term in order, verifying its data appears at the correct
        // position in the joined array.
        for (i, term) in terms.iter().enumerate() {
            let b = b.subcircuit(format!("term[{i}]"));

            let too_long = b.icmp_ugt(
                term.len_bytes,
                b.add_constant(Word((term.data.len() << 3) as u64)),
            );
            b.assert_false("term_length_check", too_long);

            // 2. Word-level verification for current term
            //
            // Process the term's data word by word (8 bytes at a time) for efficiency.
            let word_offset = b.shr(offset, 3);
            for (word_idx, &term_word) in term.data.iter().enumerate() {
                let b = b.subcircuit(format!("word[{word_idx}]"));

                // Calculate this word's byte position within the term
                let word_byte_offset = word_idx << 3;
                let word_byte_offset_wire = b.add_constant(Word(word_byte_offset as u64));

                // 2a. Validity checks
                //
                // Determine if this word contains valid data based on the term's actual length.
                // A word is:
                // - Partially valid if it at least one "meaningful" byte of the term.
                // - not partially valid if it lives entirely after the point where the term ends.
                let word_partially_valid = b.icmp_ult(word_byte_offset_wire, term.len_bytes);

                // 2b. Global position calculation
                //
                // Calculate where this word should appear in the joined array.
                // This is the current offset plus the word's position within the term.
                let input_word_idx = b.iadd_32(word_offset, b.add_constant(Word(word_idx as u64)));
                let byte_offset = b.band(offset, b.add_constant(Word(7)));

                // 2c. Extract corresponding data from joined array
                //
                // Extract the word from the joined array at the calculated position.
                // This handles both aligned (byte position % 8 == 0) and unaligned cases.
                let joined_data = extract_word(&b, &joined, input_word_idx, byte_offset);

                let neg_start = b.add_constant(Word((-(word_byte_offset as i64)) as u64));
                let bytes_remaining = b.iadd_32(term.len_bytes, neg_start);

                // The mask will handle clamping to 8 bytes internally
                let mask = create_byte_mask(&b, bytes_remaining);

                b.assert_eq_cond(
                    format!("term[{word_idx}]"),
                    b.band(joined_data, mask),
                    b.band(term_word, mask),
                    word_partially_valid,
                );
            }

            // 4. Update offset for next term
            //
            // After processing all words of the current term, advance the offset
            // by the term's actual length to position for the next term.
            offset = b.iadd_32(offset, term.len_bytes);
        }

        // 5. Final length verification
        //
        // The sum of all term lengths must equal the total joined length.
        b.assert_eq("concat_length", offset, len_joined_bytes);
        let too_long = b.icmp_ugt(
            len_joined_bytes,
            b.add_constant(Word((joined.len() << 3) as u64)),
        );
        b.assert_false("concat_length_lt_joined_len", too_long);

        Concat {
            len_joined_bytes,
            joined,
            terms,
        }
    }

    /// Populate the len_joined wire with the actual joined size in bytes.
    pub fn populate_len_joined_bytes(&self, w: &mut WitnessFiller, len_joined_bytes: usize) {
        w[self.len_joined_bytes] = Word(len_joined_bytes as u64);
    }

    /// Populate the joined array from a byte slice.
    ///
    /// Packs the bytes into 64-bit words in little-endian order and ensures
    /// any unused words are zeroed out.
    ///
    /// # Panics
    /// Panics if `joined_bytes.len()` > `max_joined_bytes()` (the maximum size possible)
    pub fn populate_joined(&self, w: &mut WitnessFiller, joined_bytes: &[u8]) {
        assert!(
            joined_bytes.len() <= self.max_joined_bytes(),
            "joined length {} exceeds maximum {}",
            joined_bytes.len(),
            self.max_joined_bytes()
        );

        for (i, chunk) in joined_bytes.chunks(8).enumerate() {
            let mut word = 0u64;
            for (j, &byte) in chunk.iter().enumerate() {
                word |= (byte as u64) << (j << 3);
            }
            w[self.joined[i]] = Word(word);
        }

        for i in joined_bytes.len().div_ceil(8)..self.joined.len() {
            w[self.joined[i]] = Word::ZERO;
        }
    }

    pub fn max_joined_bytes(&self) -> usize {
        self.joined.len() << 3
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Result, anyhow};
    use binius_core::verify::verify_constraints;
    use rand::prelude::*;

    use super::*;

    // Test utilities

    /// Helper to create a concat circuit with given parameters.
    ///
    /// Creates a circuit with the specified maximum sizes for joined data and terms.
    /// All wires are created as input/output wires for testing.
    fn create_concat_circuit(
        max_n_joined: usize,
        term_max_lens: Vec<usize>,
    ) -> (CircuitBuilder, Concat) {
        let b = CircuitBuilder::new();

        let len_joined = b.add_inout();
        let joined: Vec<Wire> = (0..max_n_joined).map(|_| b.add_inout()).collect();

        let terms: Vec<ByteVec> = term_max_lens
            .into_iter()
            .map(|max_len| ByteVec {
                len_bytes: b.add_inout(),
                data: (0..max_len).map(|_| b.add_inout()).collect(),
            })
            .collect();

        let concat = Concat::new(&b, len_joined, joined, terms);

        (b, concat)
    }

    /// Helper to test a concatenation scenario.
    ///
    /// Sets up a circuit with the given parameters and verifies that the
    /// concatenation of `term_data` equals `expected_joined`.
    fn test_concat(
        max_n_joined: usize,
        term_max_lens: Vec<usize>,
        expected_joined: &[u8],
        term_data: &[&[u8]],
    ) -> Result<()> {
        let (b, concat) = create_concat_circuit(max_n_joined, term_max_lens);
        let circuit = b.build();
        let mut filler = circuit.new_witness_filler();

        // Set the expected joined length
        concat.populate_len_joined_bytes(&mut filler, expected_joined.len());
        concat.populate_joined(&mut filler, expected_joined);

        // Set up each term
        for (i, data_bytes) in term_data.iter().enumerate() {
            concat.terms[i].populate_len_bytes(&mut filler, data_bytes.len());
            concat.terms[i].populate_data(&mut filler, data_bytes);
        }

        circuit.populate_wire_witness(&mut filler)?;

        // Verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler.into_value_vec())
            .map_err(|msg| anyhow!("verify_constraints: {}", msg))?;

        Ok(())
    }

    // Basic concatenation tests

    #[test]
    fn test_two_terms_concat() {
        // Verify basic two-term concatenation works correctly
        test_concat(2, vec![1, 1], b"helloworld", &[b"hello", b"world"]).unwrap();
    }

    #[test]
    fn test_three_terms_concat() {
        // Verify three-term concatenation maintains correct order
        test_concat(3, vec![1, 1, 1], b"foobarbaz", &[b"foo", b"bar", b"baz"]).unwrap();
    }

    #[test]
    fn test_single_term() {
        // Edge case: single term should equal the joined result
        test_concat(1, vec![1], b"hello", &[b"hello"]).unwrap();
    }

    // Empty term handling tests

    #[test]
    fn test_empty_term() {
        // Verify empty terms are handled correctly in the middle
        test_concat(2, vec![1, 1, 1], b"helloworld", &[b"hello", b"", b"world"]).unwrap();
    }

    #[test]
    fn test_all_terms_empty() {
        // Edge case: all empty terms should produce empty result
        test_concat(1, vec![1, 1], b"", &[b"", b""]).unwrap();
    }

    // Word alignment tests

    #[test]
    fn test_unaligned_terms() {
        // Test terms that don't align to word boundaries
        // This exercises the unaligned word extraction logic
        test_concat(
            3,
            vec![1, 2],
            b"hello12world456",
            &[b"hello12", b"world456"],
        )
        .unwrap();
    }

    #[test]
    fn test_single_byte_terms() {
        // Test many small terms to verify offset tracking
        test_concat(
            1,
            vec![1, 1, 1, 1, 1],
            b"abcde",
            &[b"a", b"b", b"c", b"d", b"e"],
        )
        .unwrap();
    }

    // Real-world use case tests

    #[test]
    fn test_domain_concat() {
        // Realistic example: concatenating domain name parts
        test_concat(
            4,
            vec![1, 1, 1, 1, 1],
            b"api.example.com",
            &[b"api", b".", b"example", b".", b"com"],
        )
        .unwrap();
    }

    // Error detection tests

    #[test]
    fn test_length_mismatch() {
        // Verify the circuit rejects incorrect length claims
        // Test where claimed length doesn't match actual concatenation
        let (b, concat) = create_concat_circuit(2, vec![1, 1]);
        let circuit = b.build();
        let mut filler = circuit.new_witness_filler();

        // Claim joined is 8 bytes but terms sum to 10
        concat.populate_len_joined_bytes(&mut filler, 8);
        concat.populate_joined(&mut filler, b"helloworld");

        concat.terms[0].populate_len_bytes(&mut filler, 5);
        concat.terms[0].populate_data(&mut filler, b"hello");
        concat.terms[1].populate_len_bytes(&mut filler, 5);
        concat.terms[1].populate_data(&mut filler, b"world");

        let result = circuit.populate_wire_witness(&mut filler);
        assert!(result.is_err());
    }

    #[test]
    fn test_full_last_word_rejects_wrong_data() {
        // Verify the circuit correctly rejects wrong data when the last word has 8 bytes

        // Setup: term with 16 bytes (2 full words)
        let correct_data = b"0123456789ABCDEF";
        let wrong_data = b"0123456789ABCDXX"; // Last 2 bytes are wrong
        assert_eq!(correct_data.len(), 16);
        assert_eq!(wrong_data.len(), 16);

        let (b, concat) = create_concat_circuit(2, vec![2]);
        let circuit = b.build();
        let mut filler = circuit.new_witness_filler();

        // Populate with WRONG data in joined array
        concat.populate_len_joined_bytes(&mut filler, 16);
        concat.populate_joined(&mut filler, wrong_data);

        // But claim it matches the CORRECT data in the term
        concat.terms[0].populate_len_bytes(&mut filler, 16);
        concat.terms[0].populate_data(&mut filler, correct_data);

        // This should fail since the data doesn't match
        let result = circuit.populate_wire_witness(&mut filler);
        assert!(result.is_err(), "Circuit should reject wrong data");
    }

    #[test]
    fn test_multiple_full_words_rejects_wrong_data() {
        // Test with 32 bytes - verify rejection works for multiple full words
        let correct_data = b"0123456789ABCDEF0123456789ABCDEF";
        let wrong_data = b"0123456789ABCDEF0123456789ABCDXX"; // Last word wrong

        let (b, concat) = create_concat_circuit(4, vec![4]);
        let circuit = b.build();
        let mut filler = circuit.new_witness_filler();

        concat.populate_len_joined_bytes(&mut filler, 32);
        concat.populate_joined(&mut filler, wrong_data);
        concat.terms[0].populate_len_bytes(&mut filler, 32);
        concat.terms[0].populate_data(&mut filler, correct_data);

        let result = circuit.populate_wire_witness(&mut filler);

        // Should reject wrong data
        assert!(result.is_err(), "Circuit should reject wrong data");
    }

    // Variable term size tests

    #[test]
    fn test_different_term_max_lens() {
        // Terms can have different maximum sizes
        // This allows efficient circuits when term sizes vary significantly
        test_concat(
            4,
            vec![1, 3],
            b"shorta very long string",
            &[b"short", b"a very long string"],
        )
        .unwrap();
    }

    #[test]
    fn test_mixed_term_sizes() {
        // Complex example with varied term sizes matching real-world usage
        test_concat(
            6,
            vec![1, 1, 4, 1, 2],
            b"hi.this is a much longer term.bye",
            &[b"hi", b".", b"this is a much longer term", b".", b"bye"],
        )
        .unwrap();
    }

    /// Helper to run a concat test with given data.
    ///
    /// - `term_specs`: Vector of (data, max_len) pairs for each term
    /// - `joined_override`: If Some, use this as joined data instead of concatenating terms
    /// - `should_succeed`: Whether we expect the circuit to accept or reject
    fn run_concat_test(
        term_specs: Vec<(Vec<u8>, usize)>,
        joined_override: Option<Vec<u8>>,
        should_succeed: bool,
    ) {
        let expected_joined_bytes: Vec<u8> = if joined_override.is_none() {
            term_specs
                .iter()
                .flat_map(|(data_bytes, _)| data_bytes.clone())
                .collect()
        } else {
            joined_override.clone().unwrap()
        };

        let max_n_joined = expected_joined_bytes.len().div_ceil(8);
        let term_max_lens: Vec<usize> = term_specs.iter().map(|(_, max_len)| *max_len).collect();

        let (b, concat) = create_concat_circuit(max_n_joined, term_max_lens);
        let circuit = b.build();
        let mut filler = circuit.new_witness_filler();

        concat.populate_len_joined_bytes(&mut filler, expected_joined_bytes.len());
        concat.populate_joined(&mut filler, &expected_joined_bytes);

        for (i, (data_bytes, _)) in term_specs.iter().enumerate() {
            concat.terms[i].populate_len_bytes(&mut filler, data_bytes.len());
            concat.terms[i].populate_data(&mut filler, data_bytes);
        }

        let result = circuit.populate_wire_witness(&mut filler);
        if should_succeed {
            assert!(result.is_ok(), "Expected success but got: {result:?}");
        } else {
            assert!(result.is_err(), "Expected failure but succeeded");
        }
    }

    fn random_byte_string(len: usize) -> Vec<u8> {
        let mut rng = StdRng::seed_from_u64(len as u64);
        let mut data = vec![0u8; len];
        rng.fill_bytes(&mut data);
        data
    }

    #[test]
    fn test_extra_data_rejected() {
        let term_specs = vec![(random_byte_string(5), 2), (random_byte_string(5), 2)];

        let mut joined_with_extra: Vec<u8> = term_specs
            .iter()
            .flat_map(|(data_bytes, _)| data_bytes.clone())
            .collect();
        joined_with_extra.push(42); // Add extra byte

        run_concat_test(term_specs, Some(joined_with_extra), false);
    }

    // Property-based tests
    //
    // These tests use proptest to verify the circuit behaves correctly
    // across a wide range of randomly generated inputs.

    #[cfg(test)]
    mod proptest_tests {
        use proptest::prelude::*;
        use rstest::rstest;

        use super::*;

        /// Strategy for generating random byte arrays for term data.
        fn term_data_strategy() -> impl Strategy<Value = Vec<u8>> {
            (0..=24usize).prop_map(random_byte_string)
        }

        /// Strategy for generating term specifications with proper word alignment.
        ///
        /// Each term gets a max_len that is:
        /// - At least as large as the actual data
        /// - Rounded up to the nearest multiple of 8
        fn term_specs_strategy() -> impl Strategy<Value = Vec<(Vec<u8>, usize)>> {
            prop::collection::vec(
                term_data_strategy().prop_map(|data| {
                    let max_len = (data.len().div_ceil(8) * 8).max(8);
                    (data, max_len)
                }),
                1..=3,
            )
        }

        #[rstest]
        #[case(0, 1)]
        #[case(2, 1)]
        #[case(2, 2)]
        #[case(10, 2)]
        #[case(10, 3)]
        #[case(18, 3)]
        fn test_single_term_concatenation(#[case] len: usize, #[case] max_words: usize) {
            // Special case: single term should equal joined
            let data_bytes = random_byte_string(len);
            let term_specs = vec![(data_bytes, max_words)];
            run_concat_test(term_specs, None, true);
        }

        proptest! {
            #[test]
            fn test_correct_concatenation(term_specs in term_specs_strategy()) {
                // Verify correct concatenations are accepted
                run_concat_test(term_specs, None, true);
            }

            #[test]
            fn test_empty_terms_allowed(n_terms in 1usize..=5) {
                // Verify empty terms are handled correctly
                let mut term_specs = vec![];
                for i in 0..n_terms {
                    if i % 2 == 0 {
                        term_specs.push((vec![], 8));
                    } else {
                        term_specs.push((vec![b'x'; i], (i.div_ceil(8) * 8).max(8)));
                    }
                }
                run_concat_test(term_specs, None, true);
            }

            #[test]
            fn test_wrong_joined_data(term_specs in term_specs_strategy()) {
                // Verify incorrect joined data is rejected
                prop_assume!(!term_specs.is_empty());

                let correct_joined: Vec<u8> = term_specs.iter()
                    .flat_map(|(data, _)| data.clone())
                    .collect();

                prop_assume!(!correct_joined.is_empty());

                let mut wrong_joined = correct_joined.clone();
                wrong_joined[0] ^= 1; // Flip one bit

                run_concat_test(term_specs, Some(wrong_joined), false);
            }

            #[test]
            fn test_wrong_last_byte(term_specs in term_specs_strategy()) {
                // Test modification of the LAST byte (would catch the bug)
                prop_assume!(!term_specs.is_empty());

                let correct_joined: Vec<u8> = term_specs.iter()
                    .flat_map(|(data_bytes, _)| data_bytes.clone())
                    .collect();

                prop_assume!(!correct_joined.is_empty());

                let mut wrong_joined = correct_joined.clone();
                let last_idx = wrong_joined.len() - 1;
                wrong_joined[last_idx] ^= 1; // Flip one bit in LAST byte

                run_concat_test(term_specs, Some(wrong_joined), false);
            }


            #[test]
            fn test_wrong_length_rejected(term_specs in term_specs_strategy()) {
                // Test that mismatched lengths are rejected
                prop_assume!(term_specs.len() >= 2);

                let correct_joined: Vec<u8> = term_specs.iter()
                    .flat_map(|(data_bytes, _)| data_bytes.clone())
                    .collect();

                prop_assume!(!correct_joined.is_empty());

                // Create joined data that's too short
                let short_joined = correct_joined[..correct_joined.len() - 1].to_vec();

                run_concat_test(term_specs, Some(short_joined), false);
            }

            #[test]
            fn test_swapped_terms_rejected(a in term_data_strategy(), b in term_data_strategy()) {
                // Test that swapping terms is detected
                prop_assume!(a != b && !a.is_empty() && !b.is_empty());

                let max_len_a = a.len().div_ceil(8);
                let max_len_b = b.len().div_ceil(8);

                let term_specs = vec![(a.clone(), max_len_a), (b.clone(), max_len_b)];
                let mut swapped_joined = b.clone();
                swapped_joined.extend(&a);

                run_concat_test(term_specs, Some(swapped_joined), false);
            }

            #[test]
            fn test_large_terms(n_terms in 1usize..=3, base_size in 50usize..=200) {
                // Test with larger data sizes
                let mut term_specs = vec![];
                for i in 0..n_terms {
                    let size = base_size + i * 10;
                    let data = vec![i as u8; size];
                    let max_len = size.div_ceil(8);
                    term_specs.push((data, max_len));
                }
                run_concat_test(term_specs, None, true);
            }

            #[test]
            fn test_word_boundary_terms(offset in 0usize..8) {
                // Test terms that specifically align/misalign with word boundaries
                let term1 = vec![1u8; offset];
                let term2 = vec![2u8; 8 - offset];
                let term3 = vec![3u8; 16];

                let term_specs = vec![
                    (term1, 1),
                    (term2, 1),
                    (term3, 2),
                ];

                run_concat_test(term_specs, None, true);
            }

            #[test]
            fn test_partial_term_data_rejected(term_specs in term_specs_strategy()) {
                // Test that providing partial term data is rejected
                prop_assume!(term_specs.len() >= 2);
                prop_assume!(term_specs[0].0.len() > 1);

                // Build correct joined
                let correct_joined: Vec<u8> = term_specs.iter()
                    .flat_map(|(data_bytes, _)| data_bytes.clone())
                    .collect();

                // But claim first term is shorter than it actually is
                let mut modified_specs = term_specs.clone();
                let shortened_len_bytes = modified_specs[0].0.len() - 1;
                modified_specs[0].0.truncate(shortened_len_bytes);

                // This should fail because total length won't match
                run_concat_test(modified_specs, Some(correct_joined), false);
            }
        }

        #[test]
        fn test_full_word_terms() {
            // Test terms with lengths that are multiples of 8
            let lengths = vec![1, 2, 3, 4, 5, 6];

            for len in lengths {
                let data_bytes = vec![0x55u8; len << 3]; // Repeated pattern
                let mut wrong_data = data_bytes.clone();
                wrong_data[(len << 3) - 1] = 0xAA; // Change last byte

                let term_specs = vec![(data_bytes.clone(), len)];

                // Should reject wrong data
                run_concat_test(term_specs.clone(), Some(wrong_data.clone()), false);
            }
        }

        // Additional deterministic edge case tests
        #[test]
        fn test_maximum_terms() {
            // Test with many terms to ensure no stack overflow or performance issues
            let term_specs: Vec<(Vec<u8>, usize)> =
                (0..50).map(|i| (vec![i as u8; 2], 1)).collect();
            run_concat_test(term_specs, None, true);
        }

        #[test]
        fn test_all_empty_terms() {
            // Test edge case of all empty terms
            let term_specs = vec![(vec![], 1), (vec![], 1), (vec![], 1)];
            run_concat_test(term_specs, None, true);
        }

        #[test]
        fn test_zero_length_joined_mismatch() {
            // Test when joined is empty but terms aren't
            let term_specs = vec![(vec![1, 2, 3], 1)];
            run_concat_test(term_specs, Some(vec![]), false);
        }
    }
}
