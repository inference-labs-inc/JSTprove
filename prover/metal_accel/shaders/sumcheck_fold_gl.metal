kernel void gl_fold_f(
    device const GoldilocksFr* bk_f_in [[buffer(0)]],
    device GoldilocksFr* bk_f_out [[buffer(1)]],
    device const GoldilocksFr* r_val [[buffer(2)]],
    constant uint32_t& eval_size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= eval_size) return;

    GoldilocksFr r = r_val[0];
    GoldilocksFr f0 = bk_f_in[tid * 2];
    GoldilocksFr f1 = bk_f_in[tid * 2 + 1];
    GoldilocksFr diff = gl_sub(f1, f0);
    bk_f_out[tid] = gl_add(f0, gl_mul(diff, r));
}

kernel void gl_fold_hg(
    device const GoldilocksFr* bk_hg_in [[buffer(0)]],
    device GoldilocksFr* bk_hg_out [[buffer(1)]],
    device const uint32_t* gate_exists_in [[buffer(2)]],
    device uint32_t* gate_exists_out [[buffer(3)]],
    device const GoldilocksFr* r_val [[buffer(4)]],
    constant uint32_t& eval_size [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= eval_size) return;

    uint32_t ge0 = gate_exists_in[tid * 2];
    uint32_t ge1 = gate_exists_in[tid * 2 + 1];

    if (!ge0 && !ge1) {
        gate_exists_out[tid] = 0;
        bk_hg_out[tid] = gl_zero();
        return;
    }

    gate_exists_out[tid] = 1;
    GoldilocksFr r = r_val[0];
    GoldilocksFr h0 = bk_hg_in[tid * 2];
    GoldilocksFr h1 = bk_hg_in[tid * 2 + 1];
    GoldilocksFr diff = gl_sub(h1, h0);
    bk_hg_out[tid] = gl_add(h0, gl_mul(diff, r));
}
