struct MulGate {
    uint32_t o_id;
    uint32_t i_id_0;
    uint32_t i_id_1;
    uint32_t _pad;
    BN254Fr coef;
};

struct AddGate {
    uint32_t o_id;
    uint32_t i_id_0;
    uint32_t _pad0;
    uint32_t _pad1;
    BN254Fr coef;
};

inline void bn254_locked_add(
    device uint64_t* target,
    device atomic_uint* locks,
    uint idx,
    BN254Fr addend
) {
    while (atomic_exchange_explicit(&locks[idx], 1u, memory_order_acquire) != 0u) {}
    BN254Fr old_val;
    old_val.v[0] = target[idx * 4];
    old_val.v[1] = target[idx * 4 + 1];
    old_val.v[2] = target[idx * 4 + 2];
    old_val.v[3] = target[idx * 4 + 3];
    BN254Fr new_val = bn254_add(old_val, addend);
    target[idx * 4]     = new_val.v[0];
    target[idx * 4 + 1] = new_val.v[1];
    target[idx * 4 + 2] = new_val.v[2];
    target[idx * 4 + 3] = new_val.v[3];
    atomic_store_explicit(&locks[idx], 0u, memory_order_release);
}

kernel void accumulate_mul_gates(
    device uint64_t* hg_vals [[buffer(0)]],
    device const BN254Fr* eq_evals [[buffer(1)]],
    device const BN254Fr* input_vals [[buffer(2)]],
    device const MulGate* gates [[buffer(3)]],
    device atomic_uint* gate_exists [[buffer(4)]],
    device atomic_uint* hg_locks [[buffer(5)]],
    constant uint32_t& num_gates [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_gates) return;

    MulGate g = gates[tid];
    BN254Fr r = bn254_mul(eq_evals[g.o_id], g.coef);
    BN254Fr contribution = bn254_mul(r, input_vals[g.i_id_1]);

    bn254_locked_add(hg_vals, hg_locks, g.i_id_0, contribution);
    atomic_store_explicit(&gate_exists[g.i_id_0], 1u, memory_order_relaxed);
}

kernel void accumulate_add_gates(
    device uint64_t* hg_vals [[buffer(0)]],
    device const BN254Fr* eq_evals [[buffer(1)]],
    device const AddGate* gates [[buffer(2)]],
    device atomic_uint* gate_exists [[buffer(3)]],
    device atomic_uint* hg_locks [[buffer(4)]],
    constant uint32_t& num_gates [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_gates) return;

    AddGate g = gates[tid];
    BN254Fr contribution = bn254_mul(eq_evals[g.o_id], g.coef);

    bn254_locked_add(hg_vals, hg_locks, g.i_id_0, contribution);
    atomic_store_explicit(&gate_exists[g.i_id_0], 1u, memory_order_relaxed);
}
