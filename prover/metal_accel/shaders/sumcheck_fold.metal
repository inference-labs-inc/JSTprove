kernel void fold_f(
    device const BN254Fr* bk_f_in [[buffer(0)]],
    device BN254Fr* bk_f_out [[buffer(1)]],
    device const BN254Fr* r_val [[buffer(2)]],
    constant uint32_t& eval_size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= eval_size) return;

    BN254Fr r = r_val[0];
    BN254Fr f0 = bk_f_in[tid * 2];
    BN254Fr f1 = bk_f_in[tid * 2 + 1];
    BN254Fr diff = bn254_sub(f1, f0);
    bk_f_out[tid] = bn254_add(f0, bn254_mul(diff, r));
}

kernel void fold_hg(
    device const BN254Fr* bk_hg_in [[buffer(0)]],
    device BN254Fr* bk_hg_out [[buffer(1)]],
    device const uint32_t* gate_exists_in [[buffer(2)]],
    device uint32_t* gate_exists_out [[buffer(3)]],
    device const BN254Fr* r_val [[buffer(4)]],
    constant uint32_t& eval_size [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= eval_size) return;

    uint32_t ge0 = gate_exists_in[tid * 2];
    uint32_t ge1 = gate_exists_in[tid * 2 + 1];

    if (!ge0 && !ge1) {
        gate_exists_out[tid] = 0;
        bk_hg_out[tid] = bn254_zero();
        return;
    }

    gate_exists_out[tid] = 1;
    BN254Fr r = r_val[0];
    BN254Fr h0 = bk_hg_in[tid * 2];
    BN254Fr h1 = bk_hg_in[tid * 2 + 1];
    BN254Fr diff = bn254_sub(h1, h0);
    bk_hg_out[tid] = bn254_add(h0, bn254_mul(diff, r));
}
