use super::{context::*, kernel::*, proving_system::*, shape::*};
use crate::frontend::*;

#[kernel]
fn macro_kernel<C: Config>(
    api: &mut API<C>,
    a: &[[InputVariable; 4]; 2],
    b: &mut [[OutputVariable; 1]; 4],
    c: &mut [[[InputOutputVariable; 2]; 1]; 4],
) {
    for i in 0..4 {
        b[i][0] = api.add(a[0][i], a[1][i]);
        c[i][0][0] = api.add(c[i][0][0], c[i][0][1]);
    }
}

#[kernel]
fn macro_kernel_2<C: Config>(
    api: &mut API<C>,
    a: &InputVariable,
    b: &mut OutputVariable,
    c: &mut InputOutputVariable,
) {
    *b = api.add(*a, *c);
    *c = api.add(*c, *b);
}

#[kernel]
fn macro_kernel_3<C: Config>(
    api: &mut API<C>,
    a: &mut [[[InputOutputVariable; 4]; 8]; 16],
    b: &mut [[[InputOutputVariable; 16]; 8]; 4],
    c: &InputVariable,
) {
    for i in 0..16 {
        for j in 0..8 {
            for k in 0..4 {
                let x = api.add(a[i][j][k], c);
                a[i][j][k] = b[k][j][i];
                b[k][j][i] = x;
            }
        }
    }
}

#[test]
fn compile_macro_kernels() {
    let _ = compile_macro_kernel::<M31Config>();
    let _ = compile_macro_kernel_2::<M31Config>();
    let _ = compile_macro_kernel_3::<M31Config>();
}

#[kernel]
fn identity_1<C: Config>(_api: &mut API<C>, _a: &mut InputOutputVariable) {}
#[kernel]
fn identity_3<C: Config>(_api: &mut API<C>, _a: &mut [InputOutputVariable; 3]) {}
#[kernel]
fn identity_5<C: Config>(_api: &mut API<C>, _a: &mut [InputOutputVariable; 5]) {}

fn context_shape_test_1_impl<P: ProvingSystem<M31Config>>() {
    type C = M31Config;
    type F = CircuitField<C>;
    let one = F::one();
    let identity_1 = compile_identity_1::<C>().unwrap();
    let identity_3 = compile_identity_3::<C>().unwrap();

    let mut ctx: Context<C> = Context::default();

    // Part 1
    // Since we only use the shape [15, 1], the representation of the vector is "xxxxxxxxxxxxxxx.".
    let mut a = ctx.copy_to_device(&vec![one; 15]);
    call_kernel!(ctx, identity_1, 15, mut a).unwrap();
    assert_eq!(ctx.copy_to_host::<Vec<F>>(a), vec![one; 15]);

    // Part 2
    // Since we use [15, 1] and [3, 5], the context will find a representation that is compatible with both.
    // The representation of the vector is "xxxxx...xxxxx...xxxxx...........".
    let mut a = ctx.copy_to_device(&vec![one; 15]);
    let mut b = a.reshape(&[5, 3]);
    call_kernel!(ctx, identity_1, 15, mut a).unwrap();
    call_kernel!(ctx, identity_3, 5, mut b).unwrap();
    let b = b.reshape(&[15]);
    assert_eq!(ctx.copy_to_host::<Vec<F>>(a), vec![one; 15]);
    assert_eq!(ctx.copy_to_host::<Vec<F>>(b), vec![one; 15]);

    let computation_graph = ctx.compile_computation_graph().unwrap();
    ctx.solve_witness().unwrap();

    // Debugging output and assertions
    let dm_len = ctx
        .export_device_memories()
        .iter()
        .map(|m| m.len())
        .collect::<Vec<_>>();
    for kernel in computation_graph.kernels().iter() {
        println!("Kernel input: {:?}", kernel.layered_circuit_input());
    }
    println!(
        "Commitments lens: {:?}",
        computation_graph.commitments_lens()
    );
    for proof_template in computation_graph.proof_templates() {
        println!("Proof template: {:?}", proof_template);
    }
    println!("Device memories length: {:?}", dm_len);
    assert_eq!(
        dm_len,
        vec![
            16, // Part 1 input
            16, // Part 1 output
            32, // Part 2 input
            32, // Part 2 output a
            32, // Part 2 output b
        ]
    );
    assert_eq!(computation_graph.commitments_lens(), dm_len);

    let (prover_setup, verifier_setup) = P::setup(&computation_graph);
    let proof = P::prove(
        &prover_setup,
        &computation_graph,
        ctx.export_device_memories(),
    );
    assert!(P::verify(&verifier_setup, &computation_graph, &proof));
    P::post_process();
}

#[test]
#[allow(deprecated)]
fn context_shape_test_1() {
    context_shape_test_1_impl::<DummyProvingSystem<M31Config>>();
    context_shape_test_1_impl::<Expander<M31Config>>();
}

/*
    In this test, we try to reshape a vector of length 15 into a shape of [3, 5] and then [5, 3].
    The [3, 5] shape forces the lowlevel representation to be "xxxxx...xxxxx...xxxxx...........".
    The [5, 3] shape forces the lowlevel representation to be "xxx.xxx.xxx.xxx.xxx.............".
    They are incompatible, so it will panic at the second kernel call.
*/
#[test]
#[should_panic(expected = "Detected illegal shape operation")]
fn context_shape_test_2() {
    type C = M31Config;
    type F = CircuitField<C>;
    let one = F::one();
    let identity_3 = compile_identity_3::<C>().unwrap();
    let identity_5 = compile_identity_5::<C>().unwrap();

    let mut ctx: Context<C> = Context::default();
    let a = ctx.copy_to_device(&vec![one; 15]);
    let mut b = a.reshape(&[5, 3]);
    let mut a = a.reshape(&[3, 5]);
    call_kernel!(ctx, identity_5, 3, mut a).unwrap();
    call_kernel!(ctx, identity_3, 5, mut b).unwrap();
    let _ = (a, b);
}

#[test]
fn context_shape_test_2_success() {
    type C = M31Config;
    type F = CircuitField<C>;
    let one = F::one();
    let identity_5 = compile_identity_5::<C>().unwrap();

    let mut ctx: Context<C> = Context::default();
    let a = ctx.copy_to_device(&vec![one; 15]);
    let b = a.reshape(&[5, 3]);
    let mut a = a.reshape(&[3, 5]);
    call_kernel!(ctx, identity_5, 3, mut a).unwrap();
    let _ = (a, b);
}
