#![allow(clippy::pedantic)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::ptr_arg)]

use expander_compiler::frontend::{
    BN254Config, BasicAPI, CircuitField, Config, Error, FieldArith, Variable, API,
};
use expander_compiler::zkcuda::proving_system::expander_pcs_defered::BN254ConfigSha2UniKZG;
// use expander_compiler::zkcuda::proving_system::expander_pcs_defered::BN254ConfigSha2UniKZG;
use expander_compiler::zkcuda::proving_system::{ParallelizedExpander, ProvingSystem};
use expander_compiler::zkcuda::shape::Reshape;
use expander_compiler::zkcuda::{
    context::{call_kernel, Context},
    kernel::{compile_with_spec_and_shapes, kernel, IOVecSpec, KernelPrimitive},
};

const M: usize = 512;
const K: usize = 512;

#[kernel]
fn mul_line<C: Config>(
    api: &mut API<C>,
    a: &[InputVariable; M],
    b: &[[InputVariable; K]; M],
    c: &mut [OutputVariable; K],
) {
    for j in 0..K {
        c[j] = api.constant(0);
    }

    for i in 0..M {
        for j in 0..K {
            let t = api.mul(a[i], b[i][j]);
            c[j] = api.add(c[j], t);
        }
    }
}

pub fn zkcuda_matmul<C: Config, P: ProvingSystem<C>, const N: usize>() {
    let kernel_mul_line: KernelPrimitive<C> = compile_mul_line().unwrap();

    let mut ctx: Context<C> = Context::default();

    let mut mat_a: Vec<Vec<CircuitField<C>>> = vec![];
    for i in 0..N {
        mat_a.push(vec![]);
        for j in 0..M {
            mat_a[i].push(CircuitField::<C>::from((i * 233 + j + 1) as u32));
        }
    }
    let mut mat_b: Vec<Vec<CircuitField<C>>> = vec![];
    for i in 0..M {
        mat_b.push(vec![]);
        for j in 0..K {
            mat_b[i].push(CircuitField::<C>::from((i * 2333 + j + 1) as u32));
        }
    }
    let mut expected_result = vec![vec![CircuitField::<C>::zero(); K]; N];
    for i in 0..N {
        for j in 0..K {
            for k in 0..M {
                expected_result[i][j] += mat_a[i][k] * mat_b[k][j];
            }
        }
    }

    let a = ctx.copy_to_device(&mat_a);
    let b = ctx.copy_to_device(&mat_b);
    let mut c = None;
    call_kernel!(ctx, kernel_mul_line, N, a, b, mut c).unwrap();

    let c = c.reshape(&[N, K]);
    let result: Vec<Vec<CircuitField<C>>> = ctx.copy_to_host(c);
    assert_eq!(result, expected_result);

    let computation_graph = ctx.compile_computation_graph().unwrap();
    ctx.solve_witness().unwrap();

    let (prover_setup, verifier_setup) = P::setup(&computation_graph);

    let timer = std::time::Instant::now();
    let proof = P::prove(
        &prover_setup,
        &computation_graph,
        ctx.export_device_memories(),
    );
    let elapsed = timer.elapsed();
    println!("Parallel Count {N}, Proving time: {elapsed:?}");

    let timer = std::time::Instant::now();
    assert!(P::verify(&verifier_setup, &computation_graph, &proof));
    let elapsed = timer.elapsed();
    println!("Parallel Count {N}, Verification time: {elapsed:?}");
    P::post_process();
}

fn main() {
    // zkcuda_matmul::<BN254Config, Expander<BN254ConfigSha2Hyrax>, 4>();
    // zkcuda_matmul::<BN254Config, Expander<BN254ConfigSha2Hyrax>, 8>();
    // zkcuda_matmul::<BN254Config, Expander<BN254ConfigSha2Hyrax>, 16>();
    zkcuda_matmul::<BN254Config, ParallelizedExpander<BN254ConfigSha2UniKZG>, 4>();
    zkcuda_matmul::<BN254Config, ParallelizedExpander<BN254ConfigSha2UniKZG>, 8>();
    zkcuda_matmul::<BN254Config, ParallelizedExpander<BN254ConfigSha2UniKZG>, 16>();
}
