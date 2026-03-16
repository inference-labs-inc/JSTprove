# Blendy Sumcheck Time-Space Tradeoff: Feasibility Assessment

Investigation of Chiesa & Fenzi (2024) space-efficient sumcheck algorithm for reducing
ProverScratchPad memory in JSTprove's GKR prover.

## Current Memory Profile

The `ProverScratchPad` allocates 8 arrays of size `1 << max_num_input_var`:

| Array | Element Type | Size Per Element |
|---|---|---|
| `v_evals` | `Field` (BN254) | 32 bytes |
| `hg_evals_5` | `ChallengeField` | 32 bytes |
| `hg_evals_1` | `ChallengeField` | 32 bytes |
| `hg_evals` | `Field` | 32 bytes |
| `eq_evals_at_rx` | `ChallengeField` | 32 bytes |
| `eq_evals_at_rz0` | `ChallengeField` | 32 bytes |
| `gate_exists_5` | `bool` | 1 byte |
| `gate_exists_1` | `bool` | 1 byte |

For 20 input variables: `1 << 20 = 1,048,576` elements.
Total: ~6 * 32MB + 2 * 1MB = ~194MB per layer.

## Blendy Algorithm Summary

Standard sumcheck maintains full bookkeeping tables (size N = 2^n), folding in-place each
round. Blendy with k stages divides the n variables into k groups of n/k. Instead of storing
all N evaluations, it stores O(N^{1/k}) entries and recomputes the rest by streaming through
the original polynomial evaluations k times.

With k=2: O(sqrt(N)) space, 2x time overhead.

Reference implementation: https://github.com/compsec-epfl/space-efficient-sumcheck

## Architectural Compatibility Analysis

### Problem 1: GKR's hg construction is gate-based, not stream-based

Blendy's core assumption is that the polynomial being sumchecked can be evaluated via a
**streaming pass** over a multilinear evaluation table. In the reference implementation, this
is modeled as a `Stream<F>` trait that yields evaluations sequentially.

JSTprove's GKR sumcheck does not work this way. The `prepare_x_vals` method constructs
`hg_evals` by iterating over the **gate list** (sparse), not by streaming over a dense
polynomial. Specifically:

```
for g in mul.iter() {
    hg_vals[g.i_ids[0]] += eq_evals_at_rz0[g.o_id] * g.coef * vals[g.i_ids[1]];
}
for g in add.iter() {
    hg_vals[g.i_ids[0]] += Field::from(eq_evals_at_rz0[g.o_id] * g.coef);
}
```

The gates have **random-access patterns** into `hg_vals` — gate output IDs and input IDs are
arbitrary indices. This is fundamentally incompatible with Blendy's sequential streaming model.

To apply Blendy, we would need to reformulate the hg construction to work in a streaming
fashion. This would require sorting gates by input ID and restructuring the accumulation loop,
which changes the core GKR architecture.

### Problem 2: Two-phase sumcheck structure

The GKR prover runs sumcheck in two phases:
1. **Phase 1 (rx)**: Sum over x-variables with `hg(x) = sum_z eq(z,rz) * [mul(z,x,y)*V(y) + add(z,x)*coef]`
2. **Phase 2 (ry)**: Sum over y-variables with `hg(y) = sum_z eq(z,rz) * eq(x,rx) * mul(z,x,y)`

Each phase has its own `prepare_*_vals` step that builds hg from scratch using gate iteration.
Blendy would need to be applied independently to each phase, and each phase has the same
random-access gate construction problem.

### Problem 3: The gate_exists optimization

The `gate_exists` array enables skipping zero entries during the evaluation loop
(`SumcheckProductGateHelper::evaluate`). This sparse-skip optimization is critical for
performance — many circuit positions have no gates connecting to them. Blendy's streaming
model has no analog for this sparse structure.

### Problem 4: Metal GPU path requires full arrays

The Metal acceleration path (`try_metal_xy_rounds`, `try_metal_ry_rounds`) copies full
`v_evals`, `hg_evals`, and `gate_exists` arrays to GPU buffers. Blendy's reduced-memory
representation cannot be used with the Metal path. The Metal path already gates on
`input_var_num >= 12`, which are exactly the layers where memory savings matter most.

### Problem 5: In-place folding during receive_challenge

After each round, `receive_challenge` folds the bookkeeping tables in-place:
```
bk_f[i] = bk_f[2*i] + (bk_f[2*i+1] - bk_f[2*i]).scale(&r);
```

This halves the active portion each round. Blendy replaces this with recomputation from the
original stream using stored verifier challenges. In JSTprove, the "original stream" is not a
simple multilinear polynomial — it is the gate-derived hg table, which itself requires the
full eq_evals_at_rz0 table to construct.

## What Would Be Required

To integrate Blendy into JSTprove's GKR prover:

1. **Reformulate hg construction as a streamable computation**: Sort gates by input ID,
   precompute eq_evals_at_rz0 in chunks, and accumulate hg values in streaming order.
   This requires changes to `circuit::CircuitLayer` gate storage format.

2. **Implement a Blendy-compatible bookkeeping structure**: Replace the flat `hg_evals`
   and `v_evals` arrays with a staged table that stores O(sqrt(N)) entries and can
   recompute others on demand.

3. **Disable Metal acceleration for Blendy layers**: The GPU path is incompatible.
   This means the layers that benefit most from memory savings (large layers) lose GPU
   acceleration, introducing a time cost beyond the theoretical 2x.

4. **Handle the eq polynomial tables**: `eq_evals_at_rz0` and `eq_evals_at_rx` are also
   O(N) arrays. These would need their own space-efficient computation, or they become the
   new memory bottleneck.

5. **Maintain two code paths**: Standard (current) for small layers and Metal-eligible layers;
   Blendy for large CPU-only layers that approach memory limits.

## Estimated Effort

| Task | Effort |
|---|---|
| Gate sorting and streamable hg construction | 1 week |
| Blendy bookkeeping tables (k=2) | 3-4 days |
| Eq polynomial space-efficient evaluation | 2-3 days |
| Integration with two-phase sumcheck | 2-3 days |
| Testing and correctness verification | 3-4 days |
| Benchmarking and tuning threshold | 1-2 days |
| **Total** | **3-4 weeks** |

## Expected Impact

For a layer with 20 input variables (current bottleneck):

| Metric | Standard | Blendy k=2 |
|---|---|---|
| Scratch pad memory | ~194 MB | ~6 MB (sqrt(1M) * 194 bytes) |
| Time per layer (CPU) | T | ~2T-3T (recomputation + no Metal) |
| Time per layer (Metal) | 0.3T | N/A (incompatible) |

The memory savings are substantial. The time overhead is 2-3x on CPU, but the loss of Metal
acceleration for large layers means the effective overhead vs current Metal-accelerated path
is ~7-10x.

## Recommendation

Blendy is **not directly applicable** to JSTprove's GKR sumcheck without significant
architectural changes. The fundamental mismatch is between Blendy's streaming polynomial
evaluation model and GKR's sparse gate-based hg construction.

**Alternative approaches to investigate for memory reduction:**

1. **Layer-at-a-time scratch pad reuse**: Already done (single scratch pad reused across
   layers). The current bottleneck is the max layer size.

2. **Chunked hg construction**: Build hg_evals in chunks rather than all at once. Process
   gates in sorted order, accumulating into a chunk-sized buffer. This gives O(chunk_size)
   memory without requiring the full Blendy machinery. Estimated effort: 1 week.

3. **Lazy eq evaluation**: Instead of materializing eq_evals_at_rz0 as a full array, evaluate
   eq(z, rz0) on-the-fly during gate iteration. This eliminates one O(N) array at the cost
   of per-gate eq evaluation (log(N) multiplications per gate). For sparse circuits, this is
   efficient. Estimated effort: 3-4 days.

4. **Memory-mapped scratch pad**: Use mmap for scratch pad arrays, allowing the OS to page
   out unused portions. Zero implementation effort but unpredictable performance. Worth
   benchmarking.

5. **Circuit partitioning**: Split large layers into sub-layers that fit in memory. This
   requires changes to the circuit compiler but preserves all existing prover optimizations
   including Metal. This is the approach most likely to ship.
