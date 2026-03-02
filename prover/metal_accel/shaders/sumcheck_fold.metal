#include <metal_stdlib>
using namespace metal;

#include "field_goldilocks.metal"

// Domain halving after receiving a challenge r.
// bk_f[i] = bk_f[2i] + (bk_f[2i+1] - bk_f[2i]) * r
kernel void fold_f(
    device GoldilocksField* bk_f [[buffer(0)]],
    device const GoldilocksField* r_val [[buffer(1)]],
    constant uint32_t& eval_size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= eval_size) return;

    GoldilocksField r = r_val[0];
    GoldilocksField f0 = bk_f[tid * 2];
    GoldilocksField f1 = bk_f[tid * 2 + 1];
    GoldilocksField diff = gl_sub(f1, f0);
    bk_f[tid] = gl_add(f0, gl_mul(diff, r));
}

// Domain halving for hg bookkeeping table, respecting gate_exists.
// bk_hg[i] = gate_exists[2i]||gate_exists[2i+1]
//            ? bk_hg[2i] + (bk_hg[2i+1] - bk_hg[2i]) * r
//            : 0
kernel void fold_hg(
    device GoldilocksField* bk_hg [[buffer(0)]],
    device uint32_t* gate_exists [[buffer(1)]],
    device const GoldilocksField* r_val [[buffer(2)]],
    constant uint32_t& eval_size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= eval_size) return;

    uint32_t ge0 = gate_exists[tid * 2];
    uint32_t ge1 = gate_exists[tid * 2 + 1];

    if (!ge0 && !ge1) {
        gate_exists[tid] = 0;
        bk_hg[tid] = gl_zero();
        return;
    }

    gate_exists[tid] = 1;
    GoldilocksField r = r_val[0];
    GoldilocksField h0 = bk_hg[tid * 2];
    GoldilocksField h1 = bk_hg[tid * 2 + 1];
    GoldilocksField diff = gl_sub(h1, h0);
    bk_hg[tid] = gl_add(h0, gl_mul(diff, r));
}
