// Per-threadgroup partial sums for the 3-point polynomial evaluation.
// Computes: p0 = sum hg[2i]*f[2i], p1 = sum hg[2i+1]*f[2i+1],
//           p2 = sum (hg[2i]+hg[2i+1])*(f[2i]+f[2i+1])

constant uint THREADGROUP_SIZE = 256;

kernel void poly_eval_kernel(
    device const BN254Fr* bk_f [[buffer(0)]],
    device const BN254Fr* bk_hg [[buffer(1)]],
    device const uint32_t* gate_exists [[buffer(2)]],
    device BN254Fr* block_results [[buffer(3)]],
    constant uint32_t& eval_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint grid_size [[threads_per_grid]]
) {
    threadgroup BN254Fr shared_p0[THREADGROUP_SIZE];
    threadgroup BN254Fr shared_p1[THREADGROUP_SIZE];
    threadgroup BN254Fr shared_p2[THREADGROUP_SIZE];

    BN254Fr local_p0 = bn254_zero();
    BN254Fr local_p1 = bn254_zero();
    BN254Fr local_p2 = bn254_zero();

    for (uint i = tid; i < eval_size; i += grid_size) {
        if (!gate_exists[i * 2] && !gate_exists[i * 2 + 1]) continue;

        BN254Fr f0 = bk_f[i * 2];
        BN254Fr f1 = bk_f[i * 2 + 1];
        BN254Fr h0 = bk_hg[i * 2];
        BN254Fr h1 = bk_hg[i * 2 + 1];

        local_p0 = bn254_add(local_p0, bn254_mul(h0, f0));
        local_p1 = bn254_add(local_p1, bn254_mul(h1, f1));
        local_p2 = bn254_add(local_p2, bn254_mul(bn254_add(h0, h1), bn254_add(f0, f1)));
    }

    shared_p0[lid] = local_p0;
    shared_p1[lid] = local_p1;
    shared_p2[lid] = local_p2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADGROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_p0[lid] = bn254_add(shared_p0[lid], shared_p0[lid + stride]);
            shared_p1[lid] = bn254_add(shared_p1[lid], shared_p1[lid + stride]);
            shared_p2[lid] = bn254_add(shared_p2[lid], shared_p2[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        block_results[gid * 3 + 0] = shared_p0[0];
        block_results[gid * 3 + 1] = shared_p1[0];
        block_results[gid * 3 + 2] = shared_p2[0];
    }
}

kernel void reduce_blocks(
    device BN254Fr* block_results [[buffer(0)]],
    device BN254Fr* output [[buffer(1)]],
    constant uint32_t& num_blocks [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup BN254Fr shared_p0[THREADGROUP_SIZE];
    threadgroup BN254Fr shared_p1[THREADGROUP_SIZE];
    threadgroup BN254Fr shared_p2[THREADGROUP_SIZE];

    BN254Fr acc0 = bn254_zero();
    BN254Fr acc1 = bn254_zero();
    BN254Fr acc2 = bn254_zero();

    for (uint i = lid; i < num_blocks; i += THREADGROUP_SIZE) {
        acc0 = bn254_add(acc0, block_results[i * 3 + 0]);
        acc1 = bn254_add(acc1, block_results[i * 3 + 1]);
        acc2 = bn254_add(acc2, block_results[i * 3 + 2]);
    }

    shared_p0[lid] = acc0;
    shared_p1[lid] = acc1;
    shared_p2[lid] = acc2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADGROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_p0[lid] = bn254_add(shared_p0[lid], shared_p0[lid + stride]);
            shared_p1[lid] = bn254_add(shared_p1[lid], shared_p1[lid + stride]);
            shared_p2[lid] = bn254_add(shared_p2[lid], shared_p2[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        output[0] = shared_p0[0];
        output[1] = shared_p1[0];
        output[2] = shared_p2[0];
    }
}
