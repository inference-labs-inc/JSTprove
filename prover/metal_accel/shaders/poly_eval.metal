// Per-threadgroup partial sums for the 3-point polynomial evaluation.
// Computes: p0 = Σ hg[2i]*f[2i], p1 = Σ hg[2i+1]*f[2i+1],
//           p2 = Σ (hg[2i]+hg[2i+1])*(f[2i]+f[2i+1])
// Each threadgroup reduces its portion, writes partial result to block_results.

constant uint THREADGROUP_SIZE = 256;

kernel void poly_eval_kernel(
    device const GoldilocksField* bk_f [[buffer(0)]],
    device const GoldilocksField* bk_hg [[buffer(1)]],
    device const uint32_t* gate_exists [[buffer(2)]],
    device GoldilocksField* block_results [[buffer(3)]],
    constant uint32_t& eval_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint grid_size [[threads_per_grid]]
) {
    threadgroup GoldilocksField shared_p0[THREADGROUP_SIZE];
    threadgroup GoldilocksField shared_p1[THREADGROUP_SIZE];
    threadgroup GoldilocksField shared_p2[THREADGROUP_SIZE];

    GoldilocksField local_p0 = gl_zero();
    GoldilocksField local_p1 = gl_zero();
    GoldilocksField local_p2 = gl_zero();

    for (uint i = tid; i < eval_size; i += grid_size) {
        if (!gate_exists[i * 2] && !gate_exists[i * 2 + 1]) continue;

        GoldilocksField f0 = bk_f[i * 2];
        GoldilocksField f1 = bk_f[i * 2 + 1];
        GoldilocksField h0 = bk_hg[i * 2];
        GoldilocksField h1 = bk_hg[i * 2 + 1];

        local_p0 = gl_add(local_p0, gl_mul(h0, f0));
        local_p1 = gl_add(local_p1, gl_mul(h1, f1));
        local_p2 = gl_add(local_p2, gl_mul(gl_add(h0, h1), gl_add(f0, f1)));
    }

    shared_p0[lid] = local_p0;
    shared_p1[lid] = local_p1;
    shared_p2[lid] = local_p2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADGROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_p0[lid] = gl_add(shared_p0[lid], shared_p0[lid + stride]);
            shared_p1[lid] = gl_add(shared_p1[lid], shared_p1[lid + stride]);
            shared_p2[lid] = gl_add(shared_p2[lid], shared_p2[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        block_results[gid * 3 + 0] = shared_p0[0];
        block_results[gid * 3 + 1] = shared_p1[0];
        block_results[gid * 3 + 2] = shared_p2[0];
    }
}

// Final reduction across thread blocks.
// Sums block_results[i*3+{0,1,2}] for i in 0..num_blocks.
kernel void reduce_blocks(
    device GoldilocksField* block_results [[buffer(0)]],
    device GoldilocksField* output [[buffer(1)]],
    constant uint32_t& num_blocks [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup GoldilocksField shared_p0[THREADGROUP_SIZE];
    threadgroup GoldilocksField shared_p1[THREADGROUP_SIZE];
    threadgroup GoldilocksField shared_p2[THREADGROUP_SIZE];

    GoldilocksField acc0 = gl_zero();
    GoldilocksField acc1 = gl_zero();
    GoldilocksField acc2 = gl_zero();

    for (uint i = lid; i < num_blocks; i += THREADGROUP_SIZE) {
        acc0 = gl_add(acc0, block_results[i * 3 + 0]);
        acc1 = gl_add(acc1, block_results[i * 3 + 1]);
        acc2 = gl_add(acc2, block_results[i * 3 + 2]);
    }

    shared_p0[lid] = acc0;
    shared_p1[lid] = acc1;
    shared_p2[lid] = acc2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADGROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_p0[lid] = gl_add(shared_p0[lid], shared_p0[lid + stride]);
            shared_p1[lid] = gl_add(shared_p1[lid], shared_p1[lid + stride]);
            shared_p2[lid] = gl_add(shared_p2[lid], shared_p2[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        output[0] = shared_p0[0];
        output[1] = shared_p1[0];
        output[2] = shared_p2[0];
    }
}
