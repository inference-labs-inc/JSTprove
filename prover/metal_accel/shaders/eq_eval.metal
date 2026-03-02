// Build EQ polynomial half: eq_evals[0..2^r_len] for a given r vector.
kernel void build_eq_step(
    device BN254Fr* eq_evals [[buffer(0)]],
    device const BN254Fr* r_val [[buffer(1)]],
    constant uint32_t& cur_eval_num [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= cur_eval_num) return;

    BN254Fr r = r_val[0];
    BN254Fr orig = eq_evals[tid];
    BN254Fr prod = bn254_mul(orig, r);
    eq_evals[tid + cur_eval_num] = prod;
    eq_evals[tid] = bn254_sub(orig, prod);
}

// Cross product of two half-evaluations into the full eq_evals array.
kernel void cross_prod_eq(
    device BN254Fr* eq_evals [[buffer(0)]],
    device const BN254Fr* first_half [[buffer(1)]],
    device const BN254Fr* second_half [[buffer(2)]],
    constant uint32_t& first_half_bits [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint32_t mask = (1u << first_half_bits) - 1u;
    uint32_t fh_idx = tid & mask;
    uint32_t sh_idx = tid >> first_half_bits;
    eq_evals[tid] = bn254_mul(first_half[fh_idx], second_half[sh_idx]);
}
