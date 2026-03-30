#[cfg(test)]
mod tests {
    #[test]
    fn data_parallel_gkr_cost_model() {
        // Data-parallel GKR (Libra/Thaler 2013) decomposes a circuit with B copies of
        // a subcircuit S into batch×local variables, factoring the eq polynomial as:
        //   eq(rz, o_id) = eq(rz_batch, batch_idx) * eq(rz_local, local_o_id)
        //
        // This reduces prepare_x_vals from O(B*|S|) to O(B + |S|) by precomputing
        // weighted sums across batch copies.
        //
        // Requirements for the optimization:
        //   1. B copies must have identical gate topology
        //   2. Allocation offsets must be power-of-2-aligned so batch_idx occupies
        //      the top log2(B) bits of the output ID
        //   3. The wiring predicate must decompose into batch × local components
        //
        // In JSTprove's architecture:
        //   - RecursiveCircuit segments DO contain multiple allocations (copies)
        //   - But flatten() destroys this structure, producing a flat gate list
        //   - Allocation offsets are arbitrary (set by layout solver), not power-of-2-aligned
        //   - The GKR prover uses explicit gate lists (CMT-style), not multilinear
        //     extension evaluation of wiring predicates (Libra-style)
        //
        // Converting to Libra-style would require:
        //   - Rewriting the layout solver to enforce power-of-2 alignment (increases padding waste)
        //   - Replacing gate iteration with MLE evaluation of wiring predicates
        //   - This is a complete GKR prover rewrite, not an optimization
        //
        // Profiling results (LeNet BN254, M3 Max):
        //   - prepare_x_vals + prepare_y_vals: 106ms (38.5% of GKR)
        //   - Sumcheck rounds (rx + ry): 170ms (61.5% of GKR)
        //   - Within prepare_x_vals: eq_eval 20%, gate_scatter 80%
        //
        // Even if alignment were free, the gate scatter (the dominant cost) still
        // requires writing to all B*|S| distinct hg_vals entries. The theoretical
        // O(B + |S|) saving applies to eq evaluations, not scatter writes.
        //
        // Conclusion: Not applicable without a fundamental GKR prover architecture change.
        // The explicit gate list approach is already optimal for sparse circuits where
        // gate_count << 2^input_var_num.

        let batch_copies = 64u64;
        let subcircuit_gates = 1000u64;
        let total_gates = batch_copies * subcircuit_gates;

        let explicit_cost = total_gates;
        let libra_eq_cost = batch_copies + subcircuit_gates;
        let libra_scatter_cost = total_gates;

        assert!(libra_eq_cost < explicit_cost);
        assert_eq!(libra_scatter_cost, explicit_cost);

        let eq_fraction = 0.20f64;
        let scatter_fraction = 0.80f64;
        let theoretical_speedup =
            1.0 / (eq_fraction * (libra_eq_cost as f64 / explicit_cost as f64) + scatter_fraction);
        assert!(theoretical_speedup < 1.3);
    }
}
