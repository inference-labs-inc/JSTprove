constant uint GL_THREADGROUP_SIZE = 256;

inline GoldilocksFr gl_simd_reduce_add(GoldilocksFr val) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        GoldilocksFr other;
        uint32_t lo = static_cast<uint32_t>(val.v);
        uint32_t hi = static_cast<uint32_t>(val.v >> 32);
        lo = simd_shuffle_down(lo, offset);
        hi = simd_shuffle_down(hi, offset);
        other.v = static_cast<uint64_t>(hi) << 32 | static_cast<uint64_t>(lo);
        val = gl_add(val, other);
    }
    return val;
}

kernel void gl_poly_eval_kernel(
    device const GoldilocksFr* bk_f [[buffer(0)]],
    device const GoldilocksFr* bk_hg [[buffer(1)]],
    device const uint32_t* gate_exists [[buffer(2)]],
    device GoldilocksFr* block_results [[buffer(3)]],
    constant uint32_t& eval_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint grid_size [[threads_per_grid]]
) {
    threadgroup GoldilocksFr shared_p0[GL_THREADGROUP_SIZE];
    threadgroup GoldilocksFr shared_p1[GL_THREADGROUP_SIZE];
    threadgroup GoldilocksFr shared_p2[GL_THREADGROUP_SIZE];

    GoldilocksFr local_p0 = gl_zero();
    GoldilocksFr local_p1 = gl_zero();
    GoldilocksFr local_p2 = gl_zero();

    for (uint i = tid; i < eval_size; i += grid_size) {
        if (!gate_exists[i * 2] && !gate_exists[i * 2 + 1]) continue;

        GoldilocksFr f0 = bk_f[i * 2];
        GoldilocksFr f1 = bk_f[i * 2 + 1];
        GoldilocksFr h0 = bk_hg[i * 2];
        GoldilocksFr h1 = bk_hg[i * 2 + 1];

        local_p0 = gl_add(local_p0, gl_mul(h0, f0));
        local_p1 = gl_add(local_p1, gl_mul(h1, f1));
        local_p2 = gl_add(local_p2, gl_mul(gl_add(h0, h1), gl_add(f0, f1)));
    }

    shared_p0[lid] = local_p0;
    shared_p1[lid] = local_p1;
    shared_p2[lid] = local_p2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = GL_THREADGROUP_SIZE / 2; stride >= 32; stride >>= 1) {
        if (lid < stride) {
            shared_p0[lid] = gl_add(shared_p0[lid], shared_p0[lid + stride]);
            shared_p1[lid] = gl_add(shared_p1[lid], shared_p1[lid + stride]);
            shared_p2[lid] = gl_add(shared_p2[lid], shared_p2[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid < 32) {
        GoldilocksFr p0 = shared_p0[lid];
        GoldilocksFr p1 = shared_p1[lid];
        GoldilocksFr p2 = shared_p2[lid];

        p0 = gl_simd_reduce_add(p0);
        p1 = gl_simd_reduce_add(p1);
        p2 = gl_simd_reduce_add(p2);

        if (lid == 0) {
            block_results[gid * 3 + 0] = p0;
            block_results[gid * 3 + 1] = p1;
            block_results[gid * 3 + 2] = p2;
        }
    }
}

kernel void gl_reduce_blocks(
    device GoldilocksFr* block_results [[buffer(0)]],
    device GoldilocksFr* output [[buffer(1)]],
    constant uint32_t& num_blocks [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup GoldilocksFr shared_p0[GL_THREADGROUP_SIZE];
    threadgroup GoldilocksFr shared_p1[GL_THREADGROUP_SIZE];
    threadgroup GoldilocksFr shared_p2[GL_THREADGROUP_SIZE];

    GoldilocksFr acc0 = gl_zero();
    GoldilocksFr acc1 = gl_zero();
    GoldilocksFr acc2 = gl_zero();

    for (uint i = lid; i < num_blocks; i += GL_THREADGROUP_SIZE) {
        acc0 = gl_add(acc0, block_results[i * 3 + 0]);
        acc1 = gl_add(acc1, block_results[i * 3 + 1]);
        acc2 = gl_add(acc2, block_results[i * 3 + 2]);
    }

    shared_p0[lid] = acc0;
    shared_p1[lid] = acc1;
    shared_p2[lid] = acc2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = GL_THREADGROUP_SIZE / 2; stride >= 32; stride >>= 1) {
        if (lid < stride) {
            shared_p0[lid] = gl_add(shared_p0[lid], shared_p0[lid + stride]);
            shared_p1[lid] = gl_add(shared_p1[lid], shared_p1[lid + stride]);
            shared_p2[lid] = gl_add(shared_p2[lid], shared_p2[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid < 32) {
        GoldilocksFr p0 = shared_p0[lid];
        GoldilocksFr p1 = shared_p1[lid];
        GoldilocksFr p2 = shared_p2[lid];

        p0 = gl_simd_reduce_add(p0);
        p1 = gl_simd_reduce_add(p1);
        p2 = gl_simd_reduce_add(p2);

        if (lid == 0) {
            output[0] = p0;
            output[1] = p1;
            output[2] = p2;
        }
    }
}
