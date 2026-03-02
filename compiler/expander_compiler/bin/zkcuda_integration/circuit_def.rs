#![allow(clippy::ptr_arg)]
#![allow(clippy::needless_range_loop)]

use expander_compiler::frontend::{
    BasicAPI, CircuitField, Config, Error, SIMDField, Variable, API,
};
use expander_compiler::zkcuda::shape::Reshape;
use expander_compiler::zkcuda::{
    context::{call_kernel, ComputationGraph, Context, DeviceMemoryHandle},
    kernel::{compile_with_spec_and_shapes, kernel, IOVecSpec, KernelPrimitive},
};

#[kernel]
fn add_2_macro<C: Config>(api: &mut API<C>, a: &[InputVariable; 2], b: &mut OutputVariable) {
    *b = api.add(a[0], a[1]);
}

#[kernel]
fn add_16_macro<C: Config>(api: &mut API<C>, a: &[InputVariable; 16], b: &mut OutputVariable) {
    let mut sum = api.constant(0);
    for i in 0..16 {
        sum = api.add(sum, a[i]);
    }
    *b = sum;
}

#[allow(clippy::type_complexity)]
pub fn gen_computation_graph_and_witness<C: Config>(
    input: Option<Vec<Vec<CircuitField<C>>>>,
) -> (ComputationGraph<C>, Option<Vec<Vec<SIMDField<C>>>>) {
    let kernel_add_2: KernelPrimitive<C> = compile_add_2_macro().unwrap();
    let kernel_add_16: KernelPrimitive<C> = compile_add_16_macro().unwrap();

    let mut ctx: Context<C> = Context::default();
    let a = if let Some(input) = input.as_ref() {
        assert_eq!(input.len(), 16);
        assert!(input.iter().all(|v| v.len() == 2));
        input.clone()
    } else {
        let mut tmp = vec![vec![]; 16];
        for i in 0..16 {
            for j in 0..2 {
                tmp[i].push(CircuitField::<C>::from((i * 2 + j + 1) as u32));
            }
        }
        tmp
    };

    let expected_result = a.iter().flatten().sum::<CircuitField<C>>();

    let a = ctx.copy_to_device(&a);
    let mut b: DeviceMemoryHandle = None;
    call_kernel!(ctx, kernel_add_2, 16, a, mut b).unwrap();
    let b = b.reshape(&[1, 16]);
    let mut c: DeviceMemoryHandle = None;
    call_kernel!(ctx, kernel_add_16, 1, b, mut c).unwrap();
    let c = c.reshape(&[]);
    let result: CircuitField<C> = ctx.copy_to_host(c);
    assert_eq!(result, expected_result);

    let computation_graph = ctx.compile_computation_graph().unwrap();

    let extended_witness = if input.is_some() {
        ctx.solve_witness().unwrap();
        Some(ctx.export_device_memories())
    } else {
        None
    };

    (computation_graph, extended_witness)
}
