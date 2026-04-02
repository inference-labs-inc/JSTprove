// Copyright 2025 Irreducible Inc.
use binius_core::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};

/// "Subset Sum" is the following "knowledge" problem:
///
/// - Given a list of non-negative integers and a target value, do you know a sublist which sums
///   exactly to the target value?
///
/// The circuit has the full list and the target value as public inputs, and
/// the selected sublist as a private input. It checks that that the sublist
/// sums to the target value.
pub struct SubsetSum {
    /// The number of input integers.
    ///
    /// Should be equal to `values.len()` and `selection.len()`.
    len: usize,
    /// The full list of input integers.
    values: Vec<Wire>,
    /// The target value that the sublist should sum to.
    target: Wire,
    /// The selection of the sublist. Each entry is interpreted as a boolean.
    selection: Vec<Wire>,
}

impl SubsetSum {
    /// This constructs the circuit in `builder` and stores some of the wires in
    /// a `SubsetSum` struct, which is then returned.
    /// The wires are stored so that they can be populated with values later.
    pub fn construct_circuit(builder: &mut CircuitBuilder, len: usize) -> Self {
        // the list of integers (public)
        let mut values = Vec::new();
        for _ in 0..len {
            values.push(builder.add_inout());
        }

        // the target value (public)
        let target = builder.add_inout();

        // bools which represent the selection of the subset (private)
        let mut selection = Vec::new();
        for _ in 0..len {
            selection.push(builder.add_witness());
        }

        // mask `values` using `selection`
        let mut values_masked = Vec::new();
        for i in 0..len {
            // the most significant bit is used to interpret `selection[i]` as a boolean,
            // and we create a bitmask out of that which is either all 0s or all 1s
            let bit_mask = builder.sar(selection[i], 63);
            let value_masked = builder.band(values[i], bit_mask);
            values_masked.push(value_masked);
        }

        // compute sum of `values_masked`
        let mut sum = builder.add_constant(Word::ZERO);
        let mut carry = builder.add_constant(Word::ZERO);
        for i in 0..len {
            (sum, carry) = builder.iadd_cin_cout(sum, values_masked[i], carry);
            // check that no overflow occurred
            builder.assert_false("no overflow", carry);
        }

        // check that the sum matches the target
        builder.assert_eq("sum matches target", sum, target);

        Self {
            len,
            values,
            target,
            selection,
        }
    }

    /// This populates the public wires which define the subset sum problem.
    ///
    /// - `values` is the list of integers available
    /// - `target` is the target value that a sublist should sum to
    pub fn populate_problem(&self, filler: &mut WitnessFiller<'_>, values: Vec<u64>, target: u64) {
        assert_eq!(values.len(), self.len);

        for i in 0..self.len {
            filler[self.values[i]] = Word(values[i]);
        }

        filler[self.target] = Word(target);
    }

    /// This populates the private wires which select a sublist which is claimed
    /// to sum to the target value.
    ///
    /// - `selection` should have the same length as the original list, and should contain `true`
    ///   for every number that should be included in the sublist
    pub fn populate_solution(&self, filler: &mut WitnessFiller<'_>, selection: Vec<bool>) {
        assert_eq!(selection.len(), self.len);

        for i in 0..self.len {
            let word = if selection[i] {
                Word::ALL_ONE
            } else {
                Word::ZERO
            };
            filler[self.selection[i]] = word;
        }
    }
}

#[cfg(test)]
mod tests {
    use binius_core::verify::verify_constraints;

    use super::*;

    /// Checks that it works with a valid solution.
    #[test]
    fn test_valid() {
        // build circuit
        let mut builder = CircuitBuilder::new();
        let subset_sum = SubsetSum::construct_circuit(&mut builder, 5);
        let circuit = builder.build();

        // populate witness
        let mut filler = circuit.new_witness_filler();
        subset_sum.populate_problem(&mut filler, vec![2, 5, 5, 3, 7], 17);
        subset_sum.populate_solution(&mut filler, vec![false, true, true, false, true]);
        circuit.populate_wire_witness(&mut filler).unwrap();

        // check
        let constraint_system = circuit.constraint_system();
        verify_constraints(constraint_system, &filler.into_value_vec()).unwrap();
    }

    /// Checks that it fails with an invalid solution.
    #[test]
    fn test_invalid() {
        // build circuit
        let mut builder = CircuitBuilder::new();
        let subset_sum = SubsetSum::construct_circuit(&mut builder, 5);
        let circuit = builder.build();

        // populate witness
        let mut filler = circuit.new_witness_filler();
        subset_sum.populate_problem(&mut filler, vec![2, 5, 5, 3, 7], 17);
        subset_sum.populate_solution(&mut filler, vec![false, true, true, false, false]);
        circuit.populate_wire_witness(&mut filler).unwrap_err();
    }

    /// Checks that it fails if the prover tries to use weird selections to select
    /// only part of a single value.
    #[test]
    fn test_weird_booleans() {
        // build circuit
        let mut builder = CircuitBuilder::new();
        let subset_sum = SubsetSum::construct_circuit(&mut builder, 1);
        let circuit = builder.build();

        // populate witness
        let mut filler = circuit.new_witness_filler();
        subset_sum.populate_problem(&mut filler, vec![3], 1);
        // This is trying to be a malicious prover: If the constraints are not assembled
        // carefully, this could pass because `values ^ selection` sums to target.
        // In other words, we carefully selected the bits of `selection` so that they
        // only extract certain bits of `values` if the constraints use a naive AND.
        filler[subset_sum.selection[0]] = Word(1);
        circuit.populate_wire_witness(&mut filler).unwrap_err();
    }

    /// Checks that it fails even if it's a valid solution modulo 2^64.
    #[test]
    fn test_overflow() {
        // build circuit
        let mut builder = CircuitBuilder::new();
        let subset_sum = SubsetSum::construct_circuit(&mut builder, 2);
        let circuit = builder.build();

        // populate witness
        let mut filler = circuit.new_witness_filler();
        subset_sum.populate_problem(&mut filler, vec![2 << 62, 3 << 62], 1 << 62);
        // This is a valid solution modulo 2^64, so if the constraints are not assembled
        // carefully, this could pass.
        subset_sum.populate_solution(&mut filler, vec![true, true]);
        circuit.populate_wire_witness(&mut filler).unwrap_err();
    }
}
