kernel void gl_build_eq_step(
    device GoldilocksFr* eq_evals [[buffer(0)]],
    device const GoldilocksFr* r_val [[buffer(1)]],
    constant uint32_t& cur_eval_num [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= cur_eval_num) return;

    GoldilocksFr r = r_val[0];
    GoldilocksFr orig = eq_evals[tid];
    GoldilocksFr prod = gl_mul(orig, r);
    eq_evals[tid + cur_eval_num] = prod;
    eq_evals[tid] = gl_sub(orig, prod);
}

kernel void gl_vec_add(
    device GoldilocksFr* dst [[buffer(0)]],
    device const GoldilocksFr* src [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    dst[tid] = gl_add(dst[tid], src[tid]);
}

kernel void gl_cross_prod_eq(
    device GoldilocksFr* eq_evals [[buffer(0)]],
    device const GoldilocksFr* first_half [[buffer(1)]],
    device const GoldilocksFr* second_half [[buffer(2)]],
    constant uint32_t& first_half_bits [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint32_t mask = (1u << first_half_bits) - 1u;
    uint32_t fh_idx = tid & mask;
    uint32_t sh_idx = tid >> first_half_bits;
    eq_evals[tid] = gl_mul(first_half[fh_idx], second_half[sh_idx]);
}
