#include <metal_stdlib>
using namespace metal;

#include "field_goldilocks.metal"

// Build EQ polynomial half: eq_evals[0..2^r_len] for a given r vector.
// This kernel handles one "doubling step" of the build_eq_x_r_with_buf algorithm.
// Each step k doubles the number of live entries from 2^k to 2^(k+1).
// For step k, thread i (where i < 2^k) computes:
//   eq_evals[i + 2^k] = eq_evals[i] * r[k]
//   eq_evals[i]      -= eq_evals[i + 2^k]
//
// Must dispatch one step at a time (barrier between steps).
kernel void build_eq_step(
    device GoldilocksField* eq_evals [[buffer(0)]],
    device const GoldilocksField* r_val [[buffer(1)]],
    constant uint32_t& cur_eval_num [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= cur_eval_num) return;

    GoldilocksField r = r_val[0];
    GoldilocksField orig = eq_evals[tid];
    GoldilocksField prod = gl_mul(orig, r);
    eq_evals[tid + cur_eval_num] = prod;
    eq_evals[tid] = gl_sub(orig, prod);
}

// Cross product of two half-evaluations into the full eq_evals array.
// eq_evals[i] = first_half[i & mask] * second_half[i >> first_half_bits]
kernel void cross_prod_eq(
    device GoldilocksField* eq_evals [[buffer(0)]],
    device const GoldilocksField* first_half [[buffer(1)]],
    device const GoldilocksField* second_half [[buffer(2)]],
    constant uint32_t& first_half_bits [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint32_t mask = (1u << first_half_bits) - 1u;
    uint32_t fh_idx = tid & mask;
    uint32_t sh_idx = tid >> first_half_bits;
    eq_evals[tid] = gl_mul(first_half[fh_idx], second_half[sh_idx]);
}
