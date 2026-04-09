//! Sparse multilinear extension commitment over WHIR.
//!
//! This module implements a SPARK-style sparse polynomial commitment
//! for the wiring polynomials of a layered arithmetic circuit, using
//! WHIR as the underlying dense PCS for the constituent polynomials.
//!
//! The construction follows Setty (2019), "Spartan: Efficient and
//! General-Purpose zkSNARKs Without Trusted Setup", §7.2 ("PCSPARK"),
//! adapted to the 3-arity wiring of GKR layers — a sparse multilinear
//! polynomial M(z, x, y) over n_z + n_x + n_y variables, encoded as
//! four dense vectors (row, col_x, col_y, val) of length nnz, with one
//! offline-memory-checking track per address space.
//!
//! In the holographic-GKR setting the commitment is created by a
//! trusted setup phase (the model owner) and distributed as part of
//! the verifying key, so reads of the audit timestamps are trusted —
//! Spartan §7.2.3 optimization (3) lets us drop the write-timestamp
//! commitments entirely (write_ts[i] = read_ts[i] + 1) and optimization
//! (4) lets the verifier compute mẽm directly via ẽq, avoiding any
//! commitment to the memory polynomial itself.
//!
//! Two sparse polynomials per layer (the `add` and `mul` selectors)
//! are batched into a single WHIR proximity test via the construction
//! of Arnon, Chiesa, Fenzi, and Yogev (2024), "WHIR: Reed-Solomon
//! Proximity Testing with Super-Fast Verification", §5.2.

mod commit;
mod eval_sumcheck;
mod memcheck;
mod open;
mod product_argument;
mod types;

#[cfg(test)]
mod tests;

pub use commit::{sparse_commit, SparseConstituentSlot, SparseLayout};
pub use eval_sumcheck::{
    product_arity, prove_eval_sumcheck, round_poly_degree, verify_eval_sumcheck, EvalSumcheckClaim,
    EvalSumcheckProof, EvalSumcheckRound,
};
pub use memcheck::{
    build_memcheck_sets, multiset_hash, AddrTimestamps, MemcheckSets, MemoryHashParams,
    MEMCHECK_HASH_VARS,
};
pub use open::{compute_eq_table_from_addresses, sparse_open_evalclaim, SparseEvalClaimOpening};
pub use product_argument::{
    prove_product_circuit, verify_product_circuit, ProductCircuitClaim, ProductCircuitProof,
    ProductLayerProof, ProductRound,
};
pub use types::{
    SparseArity, SparseMle3, SparseMle3Commitment, SparseMle3Opening, SparseMleError,
    SparseMleScratchPad,
};
