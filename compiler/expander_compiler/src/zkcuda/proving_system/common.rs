use crate::{
    circuit::{
        config::{Config, SIMDField},
        layered::{Circuit, NormalInputType},
    },
    zkcuda::kernel::LayeredCircuitInputVec,
};

use arith::Field;

use super::super::kernel::Kernel;

pub fn check_inputs<C: Config>(
    kernel: &Kernel<C>,
    values: &[&[SIMDField<C>]],
    parallel_count: usize,
    is_broadcast: &[bool],
) {
    if kernel.layered_circuit_input().len() != values.len() {
        panic!("Input size mismatch");
    }
    if kernel.layered_circuit_input().len() != is_broadcast.len() {
        panic!("Input size mismatch");
    }
    for i in 0..kernel.layered_circuit_input().len() {
        if is_broadcast[i] {
            if kernel.layered_circuit_input()[i].len != values[i].len() {
                panic!("Input size mismatch");
            }
        } else if kernel.layered_circuit_input()[i].len * parallel_count != values[i].len() {
            panic!("Input size mismatch");
        }
    }
}

pub fn prepare_inputs<C: Config>(
    layered_circuit: &Circuit<C, NormalInputType>,
    partition_info: &[LayeredCircuitInputVec],
    values: &[&[SIMDField<C>]],
    is_broadcast: &[bool],
    parallel_index: usize,
) -> Vec<SIMDField<C>> {
    let mut lc_input = vec![SIMDField::<C>::zero(); layered_circuit.input_size()];
    for ((input, value), ib) in partition_info.iter().zip(values.iter()).zip(is_broadcast) {
        if *ib {
            for (i, x) in value.iter().enumerate() {
                lc_input[input.offset + i] = *x;
            }
        } else {
            for (i, x) in value
                .iter()
                .skip(parallel_index * input.len)
                .take(input.len)
                .enumerate()
            {
                lc_input[input.offset + i] = *x;
            }
        }
    }
    lc_input
}
