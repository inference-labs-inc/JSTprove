//! Offline memory-checking primitives for sparse-MLE openings.
//!
//! Implements the `MemoryInTheHead` derivation from Spartan §7.2.1
//! and the multiset hash functions
//!
//! ```text
//!     h_γ(a, v, t) = a · γ²  +  v · γ  +  t
//!     H_γ(M)       = Π_{e ∈ M} (e − γ)
//! ```
//!
//! used to compress (address, value, timestamp) triples into single
//! field elements and to compress whole multisets into single field
//! elements (Spartan §7.2.1, equations 2 and 3).
//!
//! Specialized to the computation-commitment setting per Spartan
//! §7.2.3 optimization (3): the read timestamps are trusted, so
//! `write_ts[i] = read_ts[i] + 1` and we never store the write
//! timestamp vector. Per-cell read counters are used as timestamps
//! directly — no global clock — which is sound because trusted reads
//! cannot fork between cells.

use arith::Field;

/// Number of variables in the binary representation of any read /
/// audit timestamp value. We bound timestamps at `2^32` per cell,
/// which is far above the entry counts of any honest gate-set we
/// expect (largest production layer is on the order of `2^25` gates).
/// The bound exists so that the hash variable `t` lives in a fixed
/// finite range and lemma 7.2's collision bound applies cleanly.
pub const MEMCHECK_HASH_VARS: usize = 32;

/// Per-axis timestamp tracks computed by `MemoryInTheHead`.
///
/// `read_ts[i]` is the number of prior reads to address `addrs[i]`
/// (0 on the first occurrence). `audit_ts[c]` is the total number of
/// reads to memory cell `c` over the whole address sequence. These
/// are exactly the two committed tracks in Spartan §7.2.3 opt. (3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddrTimestamps<F: Field> {
    pub read_ts: Vec<F>,
    pub audit_ts: Vec<F>,
}

impl<F: Field> AddrTimestamps<F> {
    /// Run `MemoryInTheHead(m, n, addrs)` per Spartan §7.2.1 with
    /// the §7.2.3 optimization (3) timestamp simplification.
    ///
    /// * `m` is the size of the address space (number of memory
    ///   cells). The output `audit_ts` has length `m`.
    /// * `addrs` is the operation address sequence. The output
    ///   `read_ts` has length `addrs.len()`.
    ///
    /// # Panics
    /// Panics if any address in `addrs` is `>= m`, or if any per-cell
    /// read counter would exceed `2^32 − 1`. The latter bound matches
    /// [`MEMCHECK_HASH_VARS`] and exists because the field-element
    /// lift used by the multiset hash is `Field: From<u32>`.
    #[must_use]
    pub fn memory_in_the_head(m: usize, addrs: &[usize]) -> Self {
        let mut audit_counter = vec![0u32; m];
        let mut read_ts = Vec::with_capacity(addrs.len());
        for (i, &addr) in addrs.iter().enumerate() {
            assert!(
                addr < m,
                "MemoryInTheHead: addrs[{i}]={addr} out of range, m={m}"
            );
            let prior = audit_counter[addr];
            read_ts.push(F::from(prior));
            audit_counter[addr] = prior
                .checked_add(1)
                .expect("MemoryInTheHead: per-cell read counter exceeded u32::MAX");
        }
        let audit_ts = audit_counter.into_iter().map(F::from).collect();
        Self { read_ts, audit_ts }
    }
}

/// Multiset hash parameters used by an opening proof.
///
/// `gamma_1` parameterizes `h_γ` (compresses an `(addr, val, ts)`
/// triple into a single field element). `gamma_2` parameterizes `H_γ`
/// (compresses a multiset of those triples into a single product).
/// Both are sampled from the transcript at evaluation time so the
/// prover cannot influence them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryHashParams<F: Field> {
    pub gamma_1: F,
    pub gamma_2: F,
}

impl<F: Field> MemoryHashParams<F> {
    /// `h_γ(a, v, t) = a · γ²  +  v · γ  +  t`
    #[inline]
    #[must_use]
    pub fn hash_triple(&self, addr: F, val: F, ts: F) -> F {
        addr * self.gamma_1.square() + val * self.gamma_1 + ts
    }
}

/// `H_γ(M) = Π_{e ∈ M} (e − γ)`
///
/// Computed in linear time over the input slice. The caller is
/// responsible for first compressing each multiset element to a
/// single field element via [`MemoryHashParams::hash_triple`] (or any
/// other collision-resistant compression).
#[inline]
#[must_use]
pub fn multiset_hash<F: Field>(elements: &[F], gamma: F) -> F {
    elements
        .iter()
        .map(|e| *e - gamma)
        .fold(F::ONE, |acc, v| acc * v)
}

/// Build the four "memory-checking sets" for one address axis as
/// described in Spartan §7.2.1, specialized to the
/// computation-commitment setting:
///
/// ```text
///     Init   = { h_γ(c,  mem[c], 0)             | c ∈ [m] }
///     RS     = { h_γ(addrs[i], val[i], read_ts[i])     | i ∈ [n] }
///     WS     = { h_γ(addrs[i], val[i], read_ts[i] + 1) | i ∈ [n] }
///     Audit  = { h_γ(c,  mem[c], audit_ts[c])   | c ∈ [m] }
/// ```
///
/// where `mem[c]` is the value stored at memory cell `c` (the eq
/// evaluations `ẽq(c, r_axis)` in our setting; supplied by the caller
/// since the caller knows the per-axis evaluation point).
///
/// Soundness check: `H_γ(Init) · H_γ(WS)  =  H_γ(RS) · H_γ(Audit)`.
/// This is checked by the verifier at evaluation time and is why we
/// build the four hashed sets here as a single primitive — the same
/// helper is invoked once per axis.
///
/// # Panics
/// Panics if `addrs`, `val_per_op`, `read_ts` differ in length, or if
/// `mem` and `audit_ts` differ in length.
#[must_use]
pub fn build_memcheck_sets<F: Field>(
    params: &MemoryHashParams<F>,
    addrs: &[usize],
    val_per_op: &[F],
    read_ts: &[F],
    mem: &[F],
    audit_ts: &[F],
) -> MemcheckSets<F> {
    assert_eq!(addrs.len(), val_per_op.len());
    assert_eq!(addrs.len(), read_ts.len());
    assert_eq!(mem.len(), audit_ts.len());
    assert!(
        mem.len() <= u32::MAX as usize,
        "build_memcheck_sets: mem.len() {} exceeds u32::MAX",
        mem.len()
    );
    let one = F::ONE;
    let init: Vec<F> = (0..mem.len())
        .map(|c| params.hash_triple(F::from(c as u32), mem[c], F::ZERO))
        .collect();
    let audit: Vec<F> = (0..mem.len())
        .map(|c| params.hash_triple(F::from(c as u32), mem[c], audit_ts[c]))
        .collect();
    let rs: Vec<F> = (0..addrs.len())
        .map(|i| {
            let addr_u32 =
                u32::try_from(addrs[i]).expect("build_memcheck_sets: address exceeds u32::MAX");
            params.hash_triple(F::from(addr_u32), val_per_op[i], read_ts[i])
        })
        .collect();
    let ws: Vec<F> = (0..addrs.len())
        .map(|i| {
            let addr_u32 =
                u32::try_from(addrs[i]).expect("build_memcheck_sets: address exceeds u32::MAX");
            params.hash_triple(F::from(addr_u32), val_per_op[i], read_ts[i] + one)
        })
        .collect();
    MemcheckSets {
        init,
        rs,
        ws,
        audit,
    }
}

/// The four hashed multisets produced by [`build_memcheck_sets`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemcheckSets<F: Field> {
    pub init: Vec<F>,
    pub rs: Vec<F>,
    pub ws: Vec<F>,
    pub audit: Vec<F>,
}

impl<F: Field> MemcheckSets<F> {
    /// Apply `H_γ` to all four sets and return whether the subset
    /// equation `H(Init)·H(WS) = H(RS)·H(Audit)` holds.
    ///
    /// In a real protocol the verifier would not run this directly —
    /// it would receive the four product hashes from the prover and
    /// verify them via a grand-product argument. Exposed here so
    /// tests can sanity-check the construction end-to-end.
    #[must_use]
    pub fn check_subset_equation(&self, gamma: F) -> bool {
        let h_init = multiset_hash(&self.init, gamma);
        let h_ws = multiset_hash(&self.ws, gamma);
        let h_rs = multiset_hash(&self.rs, gamma);
        let h_audit = multiset_hash(&self.audit, gamma);
        h_init * h_ws == h_rs * h_audit
    }
}
