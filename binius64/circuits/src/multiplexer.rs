// Copyright 2025 Irreducible Inc.
use binius_core::{consts::WORD_SIZE_BITS, word::Word};
use binius_frontend::{CircuitBuilder, Wire};

/// Creates a multiplexer circuit that selects a group of wires from multiple groups based on a
/// selector value.
///
/// This circuit validates that the output contains the group at position `sel` from the input
/// groups. Each group must have the same number of wires.
///
/// # Arguments
/// * `b` - Circuit builder
/// * `inputs` - Slice of wire groups, where each group has the same number of wires
/// * `sel` - Selector value (only ceil(log2(N)) LSB bits are used, where N is the number of groups)
///
/// # Returns
/// A vector of wires representing the selected group
///
/// # Implementation Details
/// For each wire position across all groups, builds a separate multiplexer tree using select gates.
/// If inputs is `[[a1,a2], [b1,b2], [c1,c2]]` and sel=1, output is `[b1,b2]`.
///
/// # Panics
/// * If inputs is empty
/// * If any group is empty
/// * If groups have different lengths
pub fn multi_wire_multiplex(b: &CircuitBuilder, inputs: &[&[Wire]], sel: Wire) -> Vec<Wire> {
    assert!(!inputs.is_empty(), "Input groups must not be empty");

    let group_size = inputs[0].len();
    assert!(group_size > 0, "Groups must not be empty");

    // Assert all groups have the same length
    for (i, group) in inputs.iter().enumerate() {
        assert_eq!(
            group.len(),
            group_size,
            "All groups must have the same length. Group {} has length {}, expected {}",
            i,
            group.len(),
            group_size
        );
    }

    // For each position in the groups, build a multiplexer
    (0..group_size)
        .map(|position| {
            // Collect all wires at this position across groups
            let wires_at_position: Vec<Wire> = inputs.iter().map(|group| group[position]).collect();
            // Build multiplexer for this position
            single_wire_multiplex(b, &wires_at_position, sel)
        })
        .collect()
}

/// Creates a single-wire multiplexer circuit that selects an element from a vector based on a
/// selector value.
///
/// This circuit validates that the output contains the element at position `sel` from the input
/// vector `inputs`. The selection is done using a binary tree of 2-to-1 select gates.
///
/// # Arguments
/// * `b` - Circuit builder
/// * `inputs` - Input vector of N elements (N can be any positive number)
/// * `sel` - Selector value (only ceil(log2(N)) LSB bits are used)
///
/// # Returns
/// The output wire containing the selected element
///
/// # Implementation Details
/// - Builds a binary tree of 2-to-1 select gates, level by level
/// - Binary tree has ceil(log2(N)) levels for N inputs
/// - For non-power-of-two inputs, unpaired wires are carried forward to the next level
/// - Each level uses a different bit from the selector
/// - The final output is the single wire remaining after all levels
///
/// # Panics
/// * If inputs.len() is 0
pub fn single_wire_multiplex(b: &CircuitBuilder, inputs: &[Wire], sel: Wire) -> Wire {
    let n = inputs.len();
    if n == 0 {
        return b.add_constant(Word::ZERO);
    }

    // Calculate number of selector bits needed
    let num_sel_bits = log2_ceil_usize(n);

    // Build MUX tree from bottom to top using level-by-level approach
    // This creates an optimal tree with exactly N-1 MUX gates
    let mut current_level = inputs.to_vec();

    // Process level by level until we have a single output
    for bit_level in 0..num_sel_bits {
        let sel_bit = b.shl(sel, (WORD_SIZE_BITS - 1 - bit_level) as u32);

        // Process pairs of wires at the current level
        let next_level = current_level
            .chunks(2)
            .map(|pair| {
                if let Ok([lhs, rhs]) = TryInto::<[Wire; 2]>::try_into(pair) {
                    // We have a pair - create a MUX gate
                    // Use the current bit level for selection
                    b.select(sel_bit, rhs, lhs)
                } else {
                    // Odd wire out - carry it forward to the next level
                    pair[0]
                }
            })
            .collect();

        current_level = next_level;
    }

    // The final wire is our output
    current_level[0]
}

#[inline]
fn log2_ceil_usize(n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        (usize::BITS as usize) - ((n - 1).leading_zeros() as usize)
    }
}

#[cfg(test)]
mod tests {
    use binius_core::{verify::verify_constraints, word::Word};

    use super::*;

    /// Helper function to verify single-wire multiplexer behavior
    /// Takes input values and test cases as (selector, expected_output) pairs
    fn verify_single_wire_multiplex(values: &[u64], test_cases: &[(u64, u64)]) {
        let n = values.len();
        let builder = CircuitBuilder::new();

        // Create input wires
        let inputs: Vec<Wire> = (0..n).map(|_| builder.add_inout()).collect();
        let sel = builder.add_inout();

        // Create multiplexer circuit
        let output = single_wire_multiplex(&builder, &inputs, sel);
        let expected = builder.add_inout();
        builder.assert_eq("single_wire_multiplex_output", output, expected);

        let built = builder.build();

        // Test each case
        for &(selector, expected_val) in test_cases {
            let mut w = built.new_witness_filler();

            // Set input values
            for (i, &val) in values.iter().enumerate() {
                w[inputs[i]] = Word(val);
            }
            w[sel] = Word(selector);
            w[expected] = Word(expected_val);

            // Populate witness
            built.populate_wire_witness(&mut w).unwrap();

            // Verify constraints
            let cs = built.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_power_of_two_size() {
        // Test with 4 elements (common power-of-two case)
        verify_single_wire_multiplex(
            &[13, 7, 25, 100],
            &[
                (0, 13),  // Select index 0
                (1, 7),   // Select index 1
                (2, 25),  // Select index 2
                (3, 100), // Select index 3
            ],
        );

        // Test with 8 elements (larger power-of-two)
        let values: Vec<u64> = (10..18).collect();
        let test_cases: Vec<_> = (0..8).map(|i| (i, values[i as usize])).collect();
        verify_single_wire_multiplex(&values, &test_cases);
    }

    #[test]
    fn test_non_power_of_two() {
        // Test with 3 elements (creates asymmetric tree)
        verify_single_wire_multiplex(
            &[10, 20, 30],
            &[
                (0, 10), // Select index 0
                (1, 20), // Select index 1
                (2, 30), // Select index 2
                (3, 30), // Index 3 wraps in a specific way due to tree structure
            ],
        );

        // Test with 5 elements
        verify_single_wire_multiplex(
            &[100, 200, 300, 400, 500],
            &[
                (0, 100), // Select index 0
                (2, 300), // Select index 2
                (4, 500), // Select index 4
            ],
        );

        // Test with 7 elements
        let values = [11, 22, 33, 44, 55, 66, 77];
        verify_single_wire_multiplex(
            &values,
            &[
                (0, 11), // Select index 0
                (3, 44), // Select index 3
                (6, 77), // Select index 6
                (7, 77), // Index 7 wraps to 6 in the tree structure
            ],
        );
    }

    #[test]
    fn test_single_element() {
        // Edge case: single input always returns that input regardless of selector
        verify_single_wire_multiplex(
            &[42],
            &[
                (0, 42),   // Selector 0
                (1, 42),   // Selector 1 (ignored)
                (100, 42), // Large selector (ignored)
            ],
        );
    }

    #[test]
    fn test_out_of_bounds_selector() {
        // Test selector wrapping behavior with power-of-two size
        verify_single_wire_multiplex(
            &[10, 20, 30, 40],
            &[
                (4, 10),   // 4 & 3 = 0
                (5, 20),   // 5 & 3 = 1
                (6, 30),   // 6 & 3 = 2
                (7, 40),   // 7 & 3 = 3
                (15, 40),  // 15 & 3 = 3
                (100, 10), // 100 & 3 = 0
            ],
        );

        // Test with non-power-of-two (behavior depends on tree structure)
        verify_single_wire_multiplex(
            &[1, 2, 3],
            &[
                (3, 3), // Out of bounds wraps based on tree structure
                (4, 1), // Wraps around
                (5, 2), // Wraps around
            ],
        );
    }

    /// Helper function to verify multi-wire multiplexer behavior
    /// Takes groups of values and test cases as (selector, expected_group_index) pairs
    fn verify_multi_wire_multiplex(groups: &[Vec<u64>], test_cases: &[(u64, usize)]) {
        let num_groups = groups.len();
        let group_size = groups[0].len();
        let builder = CircuitBuilder::new();

        // Create input wire groups
        let input_groups: Vec<Vec<Wire>> = (0..num_groups)
            .map(|_| (0..group_size).map(|_| builder.add_inout()).collect())
            .collect();
        let sel = builder.add_inout();

        // Convert to the format needed by multi_wire_multiplex
        let input_refs: Vec<&[Wire]> = input_groups.iter().map(|g| g.as_slice()).collect();

        // Create multiplexer circuit
        let outputs = multi_wire_multiplex(&builder, &input_refs, sel);

        // Create expected output wires
        let expected: Vec<Wire> = (0..group_size).map(|_| builder.add_inout()).collect();
        for (i, &output) in outputs.iter().enumerate() {
            builder.assert_eq(format!("multi_wire_output_{i}"), output, expected[i]);
        }

        let built = builder.build();

        // Test each case
        for &(selector, expected_group_idx) in test_cases {
            let mut w = built.new_witness_filler();

            // Set input values
            for (group_idx, group) in groups.iter().enumerate() {
                for (wire_idx, &val) in group.iter().enumerate() {
                    w[input_groups[group_idx][wire_idx]] = Word(val);
                }
            }
            w[sel] = Word(selector);

            // Set expected values
            for (i, &val) in groups[expected_group_idx].iter().enumerate() {
                w[expected[i]] = Word(val);
            }

            // Populate witness
            built.populate_wire_witness(&mut w).unwrap();

            // Verify constraints
            let cs = built.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_multi_wire_two_wire_groups() {
        // Test with 2-wire groups
        let groups = vec![
            vec![10, 11], // Group 0
            vec![20, 21], // Group 1
            vec![30, 31], // Group 2
            vec![40, 41], // Group 3
        ];

        verify_multi_wire_multiplex(
            &groups,
            &[
                (0, 0), // Select group 0
                (1, 1), // Select group 1
                (2, 2), // Select group 2
                (3, 3), // Select group 3
                (4, 0), // Wraps to group 0
                (7, 3), // Wraps to group 3
            ],
        );
    }

    #[test]
    fn test_multi_wire_three_wire_groups() {
        // Test with 3-wire groups
        let groups = vec![
            vec![100, 101, 102], // Group 0
            vec![200, 201, 202], // Group 1
            vec![300, 301, 302], // Group 2
        ];

        verify_multi_wire_multiplex(
            &groups,
            &[
                (0, 0), // Select group 0
                (1, 1), // Select group 1
                (2, 2), // Select group 2
                (3, 2), // Wraps based on tree structure
            ],
        );
    }

    #[test]
    fn test_multi_wire_single_group() {
        // Edge case: single group with multiple wires
        let groups = vec![
            vec![50, 51, 52, 53], // Only group
        ];

        verify_multi_wire_multiplex(
            &groups,
            &[
                (0, 0),   // Select the only group
                (5, 0),   // Any selector returns the only group
                (100, 0), // Any selector returns the only group
            ],
        );
    }

    #[test]
    fn test_multi_wire_single_wire_per_group() {
        // Edge case: multiple groups but each has only one wire
        // This should behave identically to single_wire_multiplex
        let groups = vec![
            vec![10], // Group 0
            vec![20], // Group 1
            vec![30], // Group 2
            vec![40], // Group 3
        ];

        verify_multi_wire_multiplex(
            &groups,
            &[
                (0, 0), // Select group 0
                (1, 1), // Select group 1
                (2, 2), // Select group 2
                (3, 3), // Select group 3
            ],
        );
    }

    #[test]
    #[should_panic(expected = "All groups must have the same length")]
    fn test_multi_wire_mismatched_group_sizes() {
        let builder = CircuitBuilder::new();

        // Create mismatched groups
        let group1: Vec<Wire> = (0..2).map(|_| builder.add_inout()).collect();
        let group2: Vec<Wire> = (0..3).map(|_| builder.add_inout()).collect();
        let sel = builder.add_inout();

        let inputs = vec![group1.as_slice(), group2.as_slice()];

        // This should panic
        multi_wire_multiplex(&builder, &inputs, sel);
    }

    #[test]
    #[should_panic(expected = "Input groups must not be empty")]
    fn test_multi_wire_empty_inputs() {
        let builder = CircuitBuilder::new();
        let sel = builder.add_inout();

        let inputs: Vec<&[Wire]> = vec![];

        // This should panic
        multi_wire_multiplex(&builder, &inputs, sel);
    }

    #[test]
    #[should_panic(expected = "Groups must not be empty")]
    fn test_multi_wire_empty_group() {
        let builder = CircuitBuilder::new();
        let sel = builder.add_inout();

        let empty_group: &[Wire] = &[];
        let inputs = vec![empty_group];

        // This should panic
        multi_wire_multiplex(&builder, &inputs, sel);
    }
}
