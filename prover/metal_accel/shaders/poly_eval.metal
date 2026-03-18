// Per-threadgroup partial sums for the 3-point polynomial evaluation.
// Computes: p0 = sum hg[2i]*f[2i], p1 = sum hg[2i+1]*f[2i+1],
//           p2 = sum (hg[2i]+hg[2i+1])*(f[2i]+f[2i+1])

constant uint THREADGROUP_SIZE = 256;
constant uint SIMDGROUP_SIZE = 32;
constant uint SIMDGROUPS_PER_TG = THREADGROUP_SIZE / SIMDGROUP_SIZE;

kernel void poly_eval_kernel(
    device const BN254Fr* bk_f [[buffer(0)]],
    device const BN254Fr* bk_hg [[buffer(1)]],
    device const uint32_t* gate_exists [[buffer(2)]],
    device BN254Fr* block_results [[buffer(3)]],
    constant uint32_t& eval_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint grid_size [[threads_per_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup BN254Fr shared_p0[SIMDGROUPS_PER_TG];
    threadgroup BN254Fr shared_p1[SIMDGROUPS_PER_TG];
    threadgroup BN254Fr shared_p2[SIMDGROUPS_PER_TG];

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

    local_p0 = bn254_simd_sum(local_p0);
    local_p1 = bn254_simd_sum(local_p1);
    local_p2 = bn254_simd_sum(local_p2);

    if (simd_lane == 0) {
        shared_p0[simd_id] = local_p0;
        shared_p1[simd_id] = local_p1;
        shared_p2[simd_id] = local_p2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0 && simd_lane < SIMDGROUPS_PER_TG) {
        local_p0 = shared_p0[simd_lane];
        local_p1 = shared_p1[simd_lane];
        local_p2 = shared_p2[simd_lane];

        local_p0 = bn254_simd_sum(local_p0);
        local_p1 = bn254_simd_sum(local_p1);
        local_p2 = bn254_simd_sum(local_p2);

        if (simd_lane == 0) {
            block_results[gid * 3 + 0] = local_p0;
            block_results[gid * 3 + 1] = local_p1;
            block_results[gid * 3 + 2] = local_p2;
        }
    }
}

kernel void reduce_blocks(
    device BN254Fr* block_results [[buffer(0)]],
    device BN254Fr* output [[buffer(1)]],
    constant uint32_t& num_blocks [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup BN254Fr shared_p0[SIMDGROUPS_PER_TG];
    threadgroup BN254Fr shared_p1[SIMDGROUPS_PER_TG];
    threadgroup BN254Fr shared_p2[SIMDGROUPS_PER_TG];

    BN254Fr acc0 = bn254_zero();
    BN254Fr acc1 = bn254_zero();
    BN254Fr acc2 = bn254_zero();

    for (uint i = lid; i < num_blocks; i += THREADGROUP_SIZE) {
        acc0 = bn254_add(acc0, block_results[i * 3 + 0]);
        acc1 = bn254_add(acc1, block_results[i * 3 + 1]);
        acc2 = bn254_add(acc2, block_results[i * 3 + 2]);
    }

    acc0 = bn254_simd_sum(acc0);
    acc1 = bn254_simd_sum(acc1);
    acc2 = bn254_simd_sum(acc2);

    if (simd_lane == 0) {
        shared_p0[simd_id] = acc0;
        shared_p1[simd_id] = acc1;
        shared_p2[simd_id] = acc2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0 && simd_lane < SIMDGROUPS_PER_TG) {
        acc0 = shared_p0[simd_lane];
        acc1 = shared_p1[simd_lane];
        acc2 = shared_p2[simd_lane];

        acc0 = bn254_simd_sum(acc0);
        acc1 = bn254_simd_sum(acc1);
        acc2 = bn254_simd_sum(acc2);

        if (simd_lane == 0) {
            output[0] = acc0;
            output[1] = acc1;
            output[2] = acc2;
        }
    }
}
