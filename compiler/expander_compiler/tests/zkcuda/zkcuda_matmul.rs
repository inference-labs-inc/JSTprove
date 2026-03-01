use expander_compiler::frontend::*;
use expander_compiler::zkcuda::proving_system::Expander;
use expander_compiler::zkcuda::proving_system::ProvingSystem;
use expander_compiler::zkcuda::shape::Reshape;
use expander_compiler::zkcuda::{context::*, kernel::*};

#[kernel]
fn mul_line<C: Config>(
    api: &mut API<C>,
    a: &[InputVariable; 32],
    b: &[[InputVariable; 64]; 32],
    c: &mut [OutputVariable; 64],
) {
    for j in 0..64 {
        c[j] = api.constant(0);
    }
    for i in 0..32 {
        for j in 0..64 {
            let t = api.mul(a[i], b[i][j]);
            c[j] = api.add(c[j], t);
        }
    }
}

#[kernel]
fn sum_8_elements<C: Config>(api: &mut API<C>, a: &[InputVariable; 8], b: &mut OutputVariable) {
    let mut sum = api.constant(0);
    for i in 0..8 {
        sum = api.add(sum, a[i]);
    }
    *b = sum;
}

#[test]
fn zkcuda_matmul_sum() {
    let kernel_mul_line: KernelPrimitive<M31Config> = compile_mul_line().unwrap();
    let kernel_sum_8_elements: KernelPrimitive<M31Config> = compile_sum_8_elements().unwrap();

    let mut ctx: Context<M31Config> = Context::default();

    let mut mat_a: Vec<Vec<M31>> = vec![];
    for i in 0..64 {
        mat_a.push(vec![]);
        for j in 0..32 {
            mat_a[i].push(M31::from((i * 233 + j + 1) as u32));
        }
    }
    let mut mat_b: Vec<Vec<M31>> = vec![];
    for i in 0..32 {
        mat_b.push(vec![]);
        for j in 0..64 {
            mat_b[i].push(M31::from((i * 2333 + j + 1) as u32));
        }
    }
    let mut expected_result = M31::zero();
    for i in 0..64 {
        for j in 0..64 {
            for k in 0..32 {
                expected_result += mat_a[i][k] * mat_b[k][j];
            }
        }
    }

    let a = ctx.copy_to_device(&mat_a);
    let b = ctx.copy_to_device(&mat_b);
    let mut c = None;
    call_kernel!(ctx, kernel_mul_line, 64, a, b, mut c).unwrap();

    let c = c.reshape(&[512, 8]);
    let mut d = None;
    call_kernel!(ctx, kernel_sum_8_elements, 512, c, mut d).unwrap();

    let d = d.reshape(&[64, 8]);
    let mut e = None;
    call_kernel!(ctx, kernel_sum_8_elements, 64, d, mut e).unwrap();

    let e = e.reshape(&[8, 8]);
    let mut f = None;
    call_kernel!(ctx, kernel_sum_8_elements, 8, e, mut f).unwrap();

    let f = f.reshape(&[1, 8]);
    let mut g = None;
    call_kernel!(ctx, kernel_sum_8_elements, 1, f, mut g).unwrap();

    let g = g.reshape(&[]);
    let result: M31 = ctx.copy_to_host(g);
    assert_eq!(result, expected_result);

    type P = Expander<M31Config>;
    let computation_graph = ctx.compile_computation_graph().unwrap();
    ctx.solve_witness().unwrap();
    let (prover_setup, verifier_setup) = P::setup(&computation_graph);
    let proof = P::prove(
        &prover_setup,
        &computation_graph,
        ctx.export_device_memories(),
    );
    assert!(P::verify(&verifier_setup, &computation_graph, &proof));
}
