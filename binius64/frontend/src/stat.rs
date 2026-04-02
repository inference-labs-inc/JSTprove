// Copyright 2025 Irreducible Inc.

//! Circuit statistics module for analyzing constraint counts and circuit complexity.

use std::fmt;

use binius_core::{ConstraintSystem, Operand, ShiftedValueIndex};

use crate::compiler::circuit::Circuit;

/// Various stats of a circuit that affect the prover performance.
pub struct CircuitStat {
    /// Number of gates in the circuit.
    pub n_gates: usize,
    /// Number of instructions in the evaluation form of circuit.
    ///
    /// Directly proportional to performance of witness filling.
    pub n_eval_insn: usize,
    /// Number of AND constraints in the circuit.
    ///
    /// Affects performance of AND reduction.
    pub n_and_constraints: usize,
    /// Number of MUL constraints in the circuit.
    ///
    /// Affects performance of intmul reduction phase.
    pub n_mul_constraints: usize,
    /// Number of distinct value indices with non-zero shift in the circuit.
    ///
    /// Every use of a value with a distinct type and amount is counted here.
    ///
    /// Affects performance of shift reduction phase.
    pub distinct_shifted_value_indices: usize,
    /// Number of distinct value indices with zero shift in the circuit.
    ///
    /// Affects performance of shift reduction phase.
    pub distinct_unshifted_value_indices: usize,
    /// Length of the value vector.
    ///
    /// Affects performance of committing.
    pub value_vec_len: usize,
    /// Number of constant values used by the circuit.
    pub n_const: usize,
    /// Number of public input values in the circuit.
    pub n_inout: usize,
    /// Number of private input values in the circuit.
    pub n_witness: usize,
    /// Number of internal values in the circuit.
    ///
    /// Internal values are values produced by gates.
    pub n_internal: usize,
    /// Number of scratch values in the circuit.
    ///
    /// Those values are not committed, those only exist during witness generation.
    pub n_scratch: usize,
    /// Allocated size for AND constraints (power of 2)
    pub and_allocated: usize,
    /// Allocated size for MUL constraints (power of 2)
    pub mul_allocated: usize,
    /// Allocated size for public section (power of 2)
    pub public_allocated: usize,
    /// Allocated size for private section.
    ///
    /// This is the space available for witness and internal values. Note that unlike
    /// `public_allocated` and the total committed length, this is NOT necessarily a
    /// power of two. It's simply the difference between the total committed length
    /// (power of 2) and the public section size (power of 2). For example, if total
    /// is 8192 and public is 128, private is 8064.
    pub private_allocated: usize,
}

impl CircuitStat {
    /// Creates a new `CircuitStat` instance by collecting statistics from the given circuit.
    pub fn collect(circuit: &Circuit) -> Self {
        // Clone the constraint system so we can prepare it
        let mut cs = circuit.constraint_system().clone();

        // Store original counts before padding
        let n_and_constraints = cs.n_and_constraints();
        let n_mul_constraints = cs.n_mul_constraints();
        let (distinct_shifted_value_indices, distinct_unshifted_value_indices) =
            traverse_constraint_system(&cs);

        // validate_and_prepare will pad constraints to power of 2
        cs.validate_and_prepare()
            .expect("constraint system should be valid");

        // Now we have the actual allocated sizes after padding
        let and_allocated = cs.n_and_constraints();
        let mul_allocated = cs.n_mul_constraints();

        // The public section size is already determined by the layout
        let n_const = cs.value_vec_layout.n_const;
        let n_inout = cs.value_vec_layout.n_inout;
        let public_allocated = cs.value_vec_layout.offset_witness;
        let total_allocated = cs.value_vec_layout.committed_total_len;
        let private_allocated = total_allocated - public_allocated;

        Self {
            n_gates: circuit.n_gates(),
            n_eval_insn: circuit.n_eval_insn(),
            n_and_constraints,
            n_mul_constraints,
            value_vec_len: total_allocated,
            distinct_shifted_value_indices,
            distinct_unshifted_value_indices,
            n_const,
            n_inout,
            n_witness: cs.value_vec_layout.n_witness,
            n_internal: cs.value_vec_layout.n_internal,
            n_scratch: cs.value_vec_layout.n_scratch,
            and_allocated,
            mul_allocated,
            public_allocated,
            private_allocated,
        }
    }
}

impl fmt::Display for CircuitStat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Helper to format numbers with commas
        fn fmt_num(n: usize) -> String {
            let s = n.to_string();
            let mut result = String::new();
            for (i, c) in s.chars().rev().enumerate() {
                if i > 0 && i % 3 == 0 {
                    result.push(',');
                }
                result.push(c);
            }
            result.chars().rev().collect()
        }

        // Helper to create a simple progress bar
        fn progress_bar(used: usize, total: usize) -> String {
            let percent = (used as f64 / total as f64 * 100.0) as usize;
            let filled = percent / 10;
            let mut bar = String::from("[");
            for i in 0..10 {
                if i < filled {
                    bar.push('▓');
                } else {
                    bar.push('░');
                }
            }
            bar.push(']');
            bar
        }

        // Helper to get log2 of a power of 2
        fn log2(n: usize) -> u32 {
            n.trailing_zeros()
        }

        // Use pre-calculated values
        let public_used = self.n_const + self.n_inout;
        let private_used = self.n_witness + self.n_internal;
        let total_used = public_used + private_used;

        // Gates & Instructions
        writeln!(f, "Gates & Instructions")?;
        writeln!(f, "├─ Number of gates: {}", fmt_num(self.n_gates))?;
        writeln!(
            f,
            "└─ Number of evaluation instructions: {}",
            fmt_num(self.n_eval_insn)
        )?;
        writeln!(f)?;

        // Constraints
        writeln!(f, "Constraints")?;
        let and_percent = self.n_and_constraints as f64 / self.and_allocated as f64 * 100.0;
        let and_spare = self.and_allocated - self.n_and_constraints;
        writeln!(
            f,
            "├─ AND constraints: {} used ({:.1}% of 2^{})",
            fmt_num(self.n_and_constraints),
            and_percent,
            log2(self.and_allocated)
        )?;
        writeln!(
            f,
            "│  {} spare: {}",
            progress_bar(self.n_and_constraints, self.and_allocated),
            fmt_num(and_spare)
        )?;

        let mul_percent = if self.mul_allocated > 0 {
            self.n_mul_constraints as f64 / self.mul_allocated as f64 * 100.0
        } else {
            0.0
        };
        let mul_spare = self.mul_allocated - self.n_mul_constraints;
        writeln!(
            f,
            "├─ MUL constraints: {} used ({:.1}% of 2^{})",
            fmt_num(self.n_mul_constraints),
            mul_percent,
            log2(self.mul_allocated)
        )?;
        writeln!(
            f,
            "│  {} spare: {}",
            progress_bar(self.n_mul_constraints, self.mul_allocated),
            fmt_num(mul_spare)
        )?;
        writeln!(
            f,
            "└─ Distinct value indices: {}",
            fmt_num(self.distinct_shifted_value_indices + self.distinct_unshifted_value_indices)
        )?;
        writeln!(
            f,
            "   ├─ Distinct shifted value indices: {}",
            fmt_num(self.distinct_shifted_value_indices)
        )?;
        writeln!(
            f,
            "   └─ Distinct unshifted value indices: {}",
            fmt_num(self.distinct_unshifted_value_indices)
        )?;
        writeln!(f)?;

        // Value Vector
        writeln!(f, "Value Vector")?;

        // Public Section
        let public_percent = public_used as f64 / self.public_allocated as f64 * 100.0;
        let public_spare = self.public_allocated - public_used;
        writeln!(
            f,
            "├─ Public Section: {} used ({:.1}% of 2^{})",
            fmt_num(public_used),
            public_percent,
            log2(self.public_allocated)
        )?;
        writeln!(
            f,
            "│  {} spare: {}",
            progress_bar(public_used, self.public_allocated),
            fmt_num(public_spare)
        )?;
        writeln!(f, "│  ├─ Constants: {}", fmt_num(self.n_const))?;
        writeln!(f, "│  └─ Inout: {}", fmt_num(self.n_inout))?;

        // Private Section (no allocated size shown since it's not a power of 2)
        let private_percent = private_used as f64 / self.private_allocated as f64 * 100.0;
        let private_spare = self.private_allocated - private_used;
        writeln!(
            f,
            "├─ Private Section: {} used ({:.1}%)",
            fmt_num(private_used),
            private_percent
        )?;
        writeln!(
            f,
            "│  {} spare: {}",
            progress_bar(private_used, self.private_allocated),
            fmt_num(private_spare)
        )?;
        writeln!(f, "│  ├─ Witness: {}", fmt_num(self.n_witness))?;
        writeln!(f, "│  └─ Internal: {}", fmt_num(self.n_internal))?;

        // Total Committed
        let total_percent = total_used as f64 / self.value_vec_len as f64 * 100.0;
        let total_spare = self.value_vec_len - total_used;
        writeln!(
            f,
            "├─ Total Committed: {} used ({:.1}% of 2^{})",
            fmt_num(total_used),
            total_percent,
            log2(self.value_vec_len)
        )?;
        writeln!(
            f,
            "│  {} spare: {}",
            progress_bar(total_used, self.value_vec_len),
            fmt_num(total_spare)
        )?;

        // Scratch
        writeln!(f, "└─ Scratch (uncommitted): {}", fmt_num(self.n_scratch))?;
        writeln!(f)?;

        Ok(())
    }
}

/// Traverses the constraint system and returns the number of distinct value indices that
/// are shifted and unshifted, respectively.
fn traverse_constraint_system(cs: &ConstraintSystem) -> (usize, usize) {
    use std::collections::HashSet;
    let mut cx = Cx {
        shifted_terms: HashSet::new(),
        unshifted_terms: HashSet::new(),
    };
    for and in &cs.and_constraints {
        visit_operand(&and.a, &mut cx);
        visit_operand(&and.b, &mut cx);
        visit_operand(&and.c, &mut cx);
    }
    for mul in &cs.mul_constraints {
        visit_operand(&mul.a, &mut cx);
        visit_operand(&mul.b, &mut cx);
        visit_operand(&mul.lo, &mut cx);
        visit_operand(&mul.hi, &mut cx);
    }
    return (cx.shifted_terms.len(), cx.unshifted_terms.len());

    struct Cx {
        shifted_terms: HashSet<ShiftedValueIndex>,
        unshifted_terms: HashSet<ShiftedValueIndex>,
    }

    fn visit_operand(operand: &Operand, cx: &mut Cx) {
        for term in operand {
            if term.amount == 0 {
                cx.unshifted_terms.insert(*term);
            } else {
                cx.shifted_terms.insert(*term);
            }
        }
    }
}
