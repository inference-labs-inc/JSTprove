struct MulGate {
    uint32_t o_id;
    uint32_t i_id_0;
    uint32_t i_id_1;
    uint32_t _pad;
    GoldilocksField coef;
};

struct AddGate {
    uint32_t o_id;
    uint32_t i_id_0;
    uint32_t _pad0;
    uint32_t _pad1;
    GoldilocksField coef;
};

inline void gl_locked_add(
    device uint64_t* target,
    device atomic_uint* locks,
    uint idx,
    GoldilocksField addend
) {
    while (atomic_exchange_explicit(&locks[idx], 1u, memory_order_relaxed) != 0u) {}
    GoldilocksField old_val{target[idx]};
    GoldilocksField new_val = gl_add(old_val, addend);
    target[idx] = new_val.val;
    atomic_store_explicit(&locks[idx], 0u, memory_order_relaxed);
}

kernel void accumulate_mul_gates(
    device uint64_t* hg_vals [[buffer(0)]],
    device const GoldilocksField* eq_evals [[buffer(1)]],
    device const GoldilocksField* input_vals [[buffer(2)]],
    device const MulGate* gates [[buffer(3)]],
    device atomic_uint* gate_exists [[buffer(4)]],
    device atomic_uint* hg_locks [[buffer(5)]],
    constant uint32_t& num_gates [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_gates) return;

    MulGate g = gates[tid];
    GoldilocksField r = gl_mul(eq_evals[g.o_id], g.coef);
    GoldilocksField contribution = gl_mul(r, input_vals[g.i_id_1]);

    gl_locked_add(hg_vals, hg_locks, g.i_id_0, contribution);
    atomic_store_explicit(&gate_exists[g.i_id_0], 1u, memory_order_relaxed);
}

kernel void accumulate_add_gates(
    device uint64_t* hg_vals [[buffer(0)]],
    device const GoldilocksField* eq_evals [[buffer(1)]],
    device const AddGate* gates [[buffer(2)]],
    device atomic_uint* gate_exists [[buffer(3)]],
    device atomic_uint* hg_locks [[buffer(4)]],
    constant uint32_t& num_gates [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_gates) return;

    AddGate g = gates[tid];
    GoldilocksField contribution = gl_mul(eq_evals[g.o_id], g.coef);

    gl_locked_add(hg_vals, hg_locks, g.i_id_0, contribution);
    atomic_store_explicit(&gate_exists[g.i_id_0], 1u, memory_order_relaxed);
}
