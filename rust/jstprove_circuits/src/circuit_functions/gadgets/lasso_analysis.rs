//! Investigation: Lasso Lookup Arguments for Range Checks
//!
//! # Objective
//!
//! Evaluate whether Lasso lookup arguments (Setty, Thaler, Wahby 2023)
//! can replace or complement existing LogUp-based range checks to reduce
//! constraint counts and improve proving time in the JSTprove pipeline.
//!
//! # Current Architecture
//!
//! JSTprove compiles ONNX models into layered arithmetic circuits via
//! the Expander Compiler Collection (ECC). The Expander prover uses the
//! GKR protocol to prove satisfiability of these layered circuits. Range
//! checks are implemented as LogUp arguments within this circuit model:
//!
//! 1. A shared lookup table of 2^chunk_bits entries (default: 2^12 = 4096)
//!    is materialized as circuit constants.
//! 2. Each value to be range-checked is decomposed into base-2^chunk_bits
//!    digits via a prover hint (`rangeproof_hint`).
//! 3. Digit decomposition correctness is enforced by a weighted-sum
//!    equality constraint.
//! 4. A LogUp grand-sum identity (rational polynomial check) ensures
//!    every digit appears in the lookup table.
//! 5. The LogUp check uses `get_random_value()` for Schwartz-Zippel
//!    randomness and `new_hint()` for query-count computation.
//!
//! For a typical BN254 circuit with kappa=18, n_bits=64, chunk_bits=12:
//! - Each range check produces ceil(64/12) = 6 digit queries
//! - Table size: 4096 constant variables
//! - LogUp finalization: O(m + T) rational additions where m = total
//!   queries, T = table size
//!
//! # Lasso Overview
//!
//! Lasso replaces traditional sorted-table lookups with:
//!
//! 1. **Sparse polynomial commitments (Spark)**: Commit to a sparse
//!    multilinear polynomial representing the lookup index matrix M,
//!    where M[i][j] = 1 iff query i accesses table row j.
//!
//! 2. **Offline memory checking**: Verify lookup correctness via
//!    timestamp-based read/write consistency (Blum et al.), reducing
//!    the problem to multilinear sum-check instances.
//!
//! 3. **Tensor-structured table decomposition**: For tables with
//!    structure (e.g., range [0, 2^n)), decompose into c subtables of
//!    size N^{1/c}, so prover commits to 3*c*m + c*N^{1/c} field
//!    elements for m lookups into a table of size N.
//!
//! Key claimed benefits:
//! - Prover work proportional to lookups performed, not table size
//! - No sorted permutation arguments
//! - Committed field elements are "small" (in {0, ..., m})
//! - 10-40x speedup over halo2 lookup arguments (different backend)
//!
//! # Architectural Incompatibility Analysis
//!
//! ## The Core Problem
//!
//! Lasso is a standalone lookup argument protocol with its own
//! polynomial commitment infrastructure. It is NOT a circuit-level
//! gadget that can be expressed through the Expander `RootAPI<C>`
//! interface.
//!
//! The Expander compiler API provides:
//! - `add`, `mul`, `sub` (arithmetic gate generation)
//! - `constant` (compile-time constants)
//! - `assert_is_equal`, `assert_is_bool`, `assert_is_zero`
//! - `get_random_value` (Schwartz-Zippel challenges)
//! - `new_hint` (unconstrained prover computation)
//!
//! These primitives generate layered arithmetic circuits for GKR
//! proving. The LogUp implementation works within this model by
//! expressing the LogUp grand-sum identity as arithmetic constraints.
//!
//! Lasso requires:
//! - A separate sparse polynomial commitment scheme (Spark/Hyrax)
//! - Independent sum-check protocol instances
//! - Offline memory checking with read/write timestamps
//! - Its own polynomial commitment infrastructure (not GKR gates)
//!
//! There is no way to express Lasso's sparse polynomial commitment
//! or offline memory checking through `RootAPI<C>` arithmetic gates.
//! The commitment scheme operates at the protocol level, below the
//! circuit abstraction.
//!
//! ## Why LogUp-GKR Already Captures Lasso's Key Insight
//!
//! The LogUp-GKR approach (Haböck 2023, "Improving logarithmic
//! derivative lookups using GKR") was directly inspired by Lasso.
//! It takes Lasso's core idea — using sum-check to verify lookups
//! without sorted permutations — and adapts it for GKR-based provers.
//!
//! LogUp-GKR achieves:
//! - Sum-check-based verification (same as Lasso)
//! - No sorted permutation arguments (same as Lasso)
//! - Native integration with GKR layered circuits (Lasso cannot)
//! - Prover cost proportional to queries + table size
//!
//! The Expander prover's `LogUpRangeProofTable` already uses this
//! approach. The rational polynomial identity check in `final_check()`
//! is the GKR-native equivalent of Lasso's sum-check verification.
//!
//! ## Concrete Cost Comparison
//!
//! For m range checks of n-bit values with chunk_bits = b:
//!
//! **Current LogUp (in-circuit)**:
//! - Digits per value: ceil(n/b)
//! - Total digit queries: m * ceil(n/b)
//! - Table constants: 2^b
//! - Rational additions in finalize: O(m * ceil(n/b) + 2^b)
//! - All expressed as GKR arithmetic gates
//!
//! **Hypothetical Lasso (standalone protocol)**:
//! - Committed field elements: 3*c*m*ceil(n/b) + c*(2^b)^{1/c}
//! - With c=2, b=12: 6*m*ceil(n/b) + 2*64 = 6*m*6 + 128
//! - Requires separate polynomial commitment scheme
//! - Cannot share GKR proving infrastructure
//!
//! For small tables (2^12 = 4096), Lasso's table-size-independent
//! benefit is negligible — the table is already small. The overhead
//! of running a separate commitment scheme would dominate any savings.
//!
//! Lasso's advantages materialize for very large tables (2^20+) where
//! LogUp's table-proportional cost dominates. For range checks with
//! chunk_bits=12, the table is only 4096 entries — well within
//! LogUp's efficient regime.
//!
//! ## Integration Paths Considered
//!
//! 1. **Pure circuit-level Lasso emulation**: Impossible. Lasso's
//!    sparse polynomial commitment cannot be expressed as arithmetic
//!    constraints in the GKR circuit model.
//!
//! 2. **Hybrid approach (Lasso for lookups, GKR for arithmetic)**:
//!    Would require Expander to support heterogeneous proving
//!    backends — a fundamental architectural change to the prover,
//!    not a gadget-level modification. This would need upstream
//!    changes to PolyhedraZK/Expander.
//!
//! 3. **Lasso-inspired optimizations within LogUp**: The relevant
//!    ideas (sum-check verification, no sorted permutations) are
//!    already captured by the LogUp-GKR approach that Expander uses.
//!
//! # Conclusion
//!
//! Lasso lookup arguments cannot be integrated into JSTprove's
//! proving pipeline without replacing the Expander GKR prover
//! with a Lasso-native proving system. The architectural mismatch
//! is fundamental: Lasso is a proving protocol, not a circuit gadget.
//!
//! The existing LogUp implementation already incorporates the key
//! insight from Lasso (sum-check-based lookup verification without
//! sorted permutations) through the LogUp-GKR approach. For the
//! table sizes used in range checks (2^12), Lasso would provide
//! no meaningful advantage even if integration were possible.
//!
//! Potential future directions that WOULD yield improvements:
//! - Upstream Expander support for native Lasso lookup gates
//! - Larger chunk widths to reduce digit count (diminishing returns
//!   past 12 bits due to GKR layer quantization, already investigated)
//! - LogUp* (eprint 2025/946) small-table optimizations if Expander
//!   adopts them

#[cfg(test)]
mod tests {
    use super::super::range_check::DEFAULT_LOGUP_CHUNK_BITS;

    fn logup_queries_per_value(n_bits: usize, chunk_bits: usize) -> usize {
        n_bits.div_ceil(chunk_bits)
    }

    fn logup_table_size(chunk_bits: usize) -> usize {
        1 << chunk_bits
    }

    fn logup_total_circuit_vars(n_values: usize, n_bits: usize, chunk_bits: usize) -> usize {
        let queries = n_values * logup_queries_per_value(n_bits, chunk_bits);
        let table = logup_table_size(chunk_bits);
        queries + table
    }

    fn lasso_committed_elements(
        n_values: usize,
        n_bits: usize,
        chunk_bits: usize,
        c: usize,
    ) -> usize {
        let queries = n_values * logup_queries_per_value(n_bits, chunk_bits);
        let table_size = logup_table_size(chunk_bits);
        let subtable_size = (table_size as f64).powf(1.0 / c as f64).ceil() as usize;
        3 * c * queries + c * subtable_size
    }

    #[test]
    fn lasso_offers_no_advantage_for_small_tables() {
        let n_values = 6500;
        let n_bits = 64;
        let chunk_bits = DEFAULT_LOGUP_CHUNK_BITS;

        let logup_vars = logup_total_circuit_vars(n_values, n_bits, chunk_bits);
        let lasso_c2 = lasso_committed_elements(n_values, n_bits, chunk_bits, 2);

        assert!(
            lasso_c2 > logup_vars,
            "For small tables (2^{chunk_bits}), Lasso c=2 commits {lasso_c2} elements \
             vs LogUp's {logup_vars} circuit variables — Lasso has higher overhead"
        );
    }

    #[test]
    fn lasso_advantage_requires_large_tables() {
        let n_values = 6500;
        let n_bits = 64;

        let large_chunk = 20;
        let logup_vars = logup_total_circuit_vars(n_values, n_bits, large_chunk);
        let lasso_c2 = lasso_committed_elements(n_values, n_bits, large_chunk, 2);

        assert!(
            lasso_c2 < logup_vars,
            "For large tables (2^{large_chunk}), Lasso c=2 commits {lasso_c2} elements \
             vs LogUp's {logup_vars} circuit variables — Lasso saves on table cost"
        );
    }

    #[test]
    fn current_chunk_bits_is_in_logup_sweet_spot() {
        let chunk_bits = DEFAULT_LOGUP_CHUNK_BITS;
        let table_size = logup_table_size(chunk_bits);

        assert!(
            table_size <= 8192,
            "Table size {table_size} is small enough that LogUp's table-proportional \
             cost is negligible compared to query-proportional cost"
        );

        let queries_per_value = logup_queries_per_value(64, chunk_bits);
        assert!(
            queries_per_value <= 6,
            "At chunk_bits={chunk_bits}, each 64-bit value produces only \
             {queries_per_value} queries — already near-optimal for GKR"
        );
    }
}
