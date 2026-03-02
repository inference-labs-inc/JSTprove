#include <metal_stdlib>
using namespace metal;

#include "field_goldilocks.metal"

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

// Atomic CAS loop for Goldilocks field addition.
// Metal has no native 64-bit atomic add, so we use compare-exchange.
inline void gl_atomic_add(device atomic_ulong* target, GoldilocksField addend) {
    uint64_t expected = atomic_load_explicit(target, memory_order_relaxed);
    GoldilocksField old_val{expected};
    GoldilocksField new_val = gl_add(old_val, addend);
    while (!atomic_compare_exchange_weak_explicit(
        target, &expected, new_val.val,
        memory_order_relaxed, memory_order_relaxed
    )) {
        old_val = GoldilocksField{expected};
        new_val = gl_add(old_val, addend);
    }
}

// One thread per multiplication gate.
// hg_vals[g.i_ids[0]] += eq_evals[g.o_id] * g.coef * input_vals[g.i_ids[1]]
kernel void accumulate_mul_gates(
    device atomic_ulong* hg_vals [[buffer(0)]],
    device const GoldilocksField* eq_evals [[buffer(1)]],
    device const GoldilocksField* input_vals [[buffer(2)]],
    device const MulGate* gates [[buffer(3)]],
    device atomic_uint* gate_exists [[buffer(4)]],
    constant uint32_t& num_gates [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_gates) return;

    MulGate g = gates[tid];
    GoldilocksField r = gl_mul(eq_evals[g.o_id], g.coef);
    GoldilocksField contribution = gl_mul(r, input_vals[g.i_id_1]);

    gl_atomic_add(&hg_vals[g.i_id_0], contribution);
    atomic_store_explicit(&gate_exists[g.i_id_0], 1u, memory_order_relaxed);
}

// One thread per addition gate.
// hg_vals[g.i_ids[0]] += eq_evals[g.o_id] * g.coef
kernel void accumulate_add_gates(
    device atomic_ulong* hg_vals [[buffer(0)]],
    device const GoldilocksField* eq_evals [[buffer(1)]],
    device const AddGate* gates [[buffer(2)]],
    device atomic_uint* gate_exists [[buffer(3)]],
    constant uint32_t& num_gates [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_gates) return;

    AddGate g = gates[tid];
    GoldilocksField contribution = gl_mul(eq_evals[g.o_id], g.coef);

    gl_atomic_add(&hg_vals[g.i_id_0], contribution);
    atomic_store_explicit(&gate_exists[g.i_id_0], 1u, memory_order_relaxed);
}
