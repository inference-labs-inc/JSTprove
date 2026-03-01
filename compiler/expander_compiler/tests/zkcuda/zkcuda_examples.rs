use expander_compiler::frontend::*;
use expander_compiler::zkcuda::proving_system::expander::config::{
    ZKCudaBN254KZG, ZKCudaBN254KZGBatchPCS,
};
use expander_compiler::zkcuda::proving_system::expander_pcs_defered::BN254ConfigSha2UniKZG;
use expander_compiler::zkcuda::proving_system::{
    Expander, ExpanderNoOverSubscribe, ParallelizedExpander, ProvingSystem,
};
use expander_compiler::zkcuda::shape::Reshape;
use expander_compiler::zkcuda::{context::*, kernel::*};

use gkr::BN254ConfigSha2Hyrax;
use serdes::ExpSerde;

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

fn zkcuda_test<C: Config, P: ProvingSystem<C>>() {
    let kernel_add_2: KernelPrimitive<C> = compile_add_2_macro().unwrap();
    let kernel_add_16: KernelPrimitive<C> = compile_add_16_macro().unwrap();
    println!("{:?}", kernel_add_2.io_shapes());
    println!("{:?}", kernel_add_16.io_shapes());

    let mut ctx: Context<C> = Context::default();
    let mut a: Vec<Vec<CircuitField<C>>> = vec![];
    for i in 0..16 {
        a.push(vec![]);
        for j in 0..2 {
            a[i].push(CircuitField::<C>::from((i * 2 + j + 1) as u32));
        }
    }
    let a = ctx.copy_to_device(&a);
    let mut b: DeviceMemoryHandle = None;
    call_kernel!(ctx, kernel_add_2, 16, a, mut b).unwrap();
    let b = b.reshape(&[1, 16]);
    let mut c: DeviceMemoryHandle = None;
    call_kernel!(ctx, kernel_add_16, 1, b, mut c).unwrap();
    let c = c.reshape(&[]);
    let result: CircuitField<C> = ctx.copy_to_host(c);
    assert_eq!(result, CircuitField::<C>::from(32 * 33 / 2));

    let computation_graph = ctx.compile_computation_graph().unwrap();
    ctx.solve_witness().unwrap();
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
fn zkcuda_test_single_core_dummy() {
    // DO NOT USE DUMMY PROVING SYSTEM IN PRODUCTION!!!
    zkcuda_test::<
        M31Config,
        expander_compiler::zkcuda::proving_system::DummyProvingSystem<M31Config>,
    >();
}

#[test]
fn zkcuda_test_single_core() {
    zkcuda_test::<GF2Config, Expander<GF2Config>>();
    zkcuda_test::<M31Config, Expander<M31Config>>();
    zkcuda_test::<GoldilocksConfig, Expander<GoldilocksConfig>>();
    zkcuda_test::<BabyBearConfig, Expander<BabyBearConfig>>();
    zkcuda_test::<BN254Config, Expander<BN254Config>>();
    zkcuda_test::<BN254Config, Expander<BN254ConfigSha2Hyrax>>();
    zkcuda_test::<BN254Config, Expander<BN254ConfigSha2UniKZG>>();
}

#[test]
fn zkcuda_test_multi_core() {
    zkcuda_test::<M31Config, ParallelizedExpander<M31Config>>();
    zkcuda_test::<GF2Config, ParallelizedExpander<GF2Config>>();
    zkcuda_test::<GoldilocksConfig, ParallelizedExpander<GoldilocksConfig>>();
    zkcuda_test::<BabyBearConfig, ParallelizedExpander<BabyBearConfig>>();
    zkcuda_test::<BN254Config, ParallelizedExpander<BN254Config>>();
    zkcuda_test::<BN254Config, ParallelizedExpander<BN254ConfigSha2Hyrax>>();
    zkcuda_test::<BN254Config, ParallelizedExpander<BN254ConfigSha2UniKZG>>();

    // zkcuda_test::<_, ExpanderNoOverSubscribe<ZKCudaBN254Hyrax>>();
    // zkcuda_test::<_, ExpanderNoOverSubscribe<ZKCudaBN254HyraxBatchPCS>>();
    zkcuda_test::<_, ExpanderNoOverSubscribe<ZKCudaBN254KZG>>();
    zkcuda_test::<_, ExpanderNoOverSubscribe<ZKCudaBN254KZGBatchPCS>>();
}

fn zkcuda_test_simd_prepare_ctx() -> Context<M31Config> {
    use arith::SimdField;

    let kernel_add_2_tmp: KernelPrimitive<M31Config> = compile_add_2_macro().unwrap();
    let kernel_add_16: KernelPrimitive<M31Config> = compile_add_16_macro().unwrap();

    let mut buf: Vec<u8> = Vec::new();
    kernel_add_2_tmp.serialize_into(&mut buf).unwrap();
    let kernel_add_2: KernelPrimitive<M31Config> =
        KernelPrimitive::deserialize_from(&mut buf.as_slice()).unwrap();

    let mut ctx: Context<M31Config> = Context::default();
    let mut a: Vec<Vec<mersenne31::M31x16>> = vec![];
    for i in 0..16 {
        a.push(vec![]);
        for j in 0..2 {
            let mut tmp = Vec::new();
            for k in 0..16 {
                tmp.push(M31::from((i * 2 + j + 1 + k) as u32));
            }
            a[i].push(mersenne31::M31x16::pack(&tmp));
        }
    }
    let a = ctx.copy_simd_to_device(&a);
    let mut b = None;
    call_kernel!(ctx, kernel_add_2, 16, a, mut b).unwrap();
    let b = b.reshape(&[1, 16]);
    let mut c = None;
    call_kernel!(ctx, kernel_add_16, 1, b, mut c).unwrap();
    let c = c.reshape(&[]);
    let result: mersenne31::M31x16 = ctx.copy_simd_to_host(c);
    let result = result.unpack();
    for k in 0..16 {
        assert_eq!(result[k], M31::from((32 * 33 / 2 + 32 * k) as u32));
    }
    ctx
}

#[test]
fn zkcuda_test_simd() {
    type P = Expander<M31Config>;

    let mut ctx = zkcuda_test_simd_prepare_ctx();

    let computation_graph = ctx.compile_computation_graph().unwrap();
    ctx.solve_witness().unwrap();
    let (prover_setup, verifier_setup) = P::setup(&computation_graph);
    let proof = P::prove(
        &prover_setup,
        &computation_graph,
        ctx.export_device_memories(),
    );
    assert!(P::verify(&verifier_setup, &computation_graph, &proof));

    // test proof serde and verification
    let mut buf_cg: Vec<u8> = Vec::new();
    computation_graph.serialize_into(&mut buf_cg).unwrap();
    let mut buf_proof: Vec<u8> = Vec::new();
    proof.serialize_into(&mut buf_proof).unwrap();

    let computation_graph2 =
        ComputationGraph::<M31Config>::deserialize_from(&mut buf_cg.as_slice()).unwrap();
    let proof2 =
        <P as ProvingSystem<M31Config>>::Proof::deserialize_from(&mut buf_proof.as_slice())
            .unwrap();
    let (_prover_setup2, verifier_setup2) = P::setup(&computation_graph2);
    assert!(P::verify(&verifier_setup2, &computation_graph2, &proof2));
    assert!(P::verify(&verifier_setup, &computation_graph, &proof));
    assert!(P::verify(&verifier_setup2, &computation_graph2, &proof));

    // test load computation graph
    let mut ctx3: Context<M31Config> = zkcuda_test_simd_prepare_ctx();
    let (prover_setup3, _verifier_setup3) = P::setup(&computation_graph2);
    ctx3.load_computation_graph(computation_graph2).unwrap();
    ctx3.solve_witness().unwrap();
    let proof3 = P::prove(
        &prover_setup3,
        &computation_graph,
        ctx3.export_device_memories(),
    );
    assert!(P::verify(&verifier_setup2, &computation_graph, &proof3));
}

#[test]
fn zkcuda_test_simd_autopack() {
    let kernel_add_2_tmp: KernelPrimitive<M31Config> = compile_add_2_macro().unwrap();
    let kernel_add_16: KernelPrimitive<M31Config> = compile_add_16_macro().unwrap();

    let mut buf: Vec<u8> = Vec::new();
    kernel_add_2_tmp.serialize_into(&mut buf).unwrap();
    let kernel_add_2: KernelPrimitive<M31Config> =
        KernelPrimitive::deserialize_from(&mut buf.as_slice()).unwrap();

    let mut ctx: Context<M31Config> = Context::default();
    let mut a: Vec<Vec<Vec<M31>>> = vec![];
    for k in 0..16 {
        a.push(vec![]);
        for i in 0..16 {
            a[k].push(vec![]);
            for j in 0..2 {
                a[k][i].push(M31::from((i * 2 + j + 1 + k) as u32));
            }
        }
    }
    let a = ctx.copy_to_device_and_pack_simd(&a);
    let mut b = None;
    call_kernel!(ctx, kernel_add_2, 16, a, mut b).unwrap();
    let b = b.reshape(&[1, 16]);
    let mut c = None;
    call_kernel!(ctx, kernel_add_16, 1, b, mut c).unwrap();
    let c = c.reshape(&[]);
    let result: Vec<M31> = ctx.copy_to_host_and_unpack_simd(c);
    for k in 0..16 {
        assert_eq!(result[k], M31::from((32 * 33 / 2 + 32 * k) as u32));
    }

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

fn to_binary<C: Config>(api: &mut API<C>, x: Variable, n_bits: usize) -> Vec<Variable> {
    api.new_hint("myhint.tobinary", &[x], n_bits)
}

fn to_binary_hint(x: &[M31], y: &mut [M31]) -> Result<(), Error> {
    let t = x[0].to_u256();
    for (i, k) in y.iter_mut().enumerate() {
        *k = M31::from_u256(t >> i as u32 & 1);
    }
    Ok(())
}

#[kernel]
fn convert_to_binary<C: Config>(api: &mut API<C>, x: &InputVariable, y: &mut [OutputVariable; 8]) {
    let bits = to_binary(api, *x, 8);
    for i in 0..8 {
        y[i] = bits[i];
    }
}

#[test]
fn zkcuda_to_binary() {
    let mut hint_registry = HintRegistry::<M31>::new();
    hint_registry.register("myhint.tobinary", to_binary_hint);

    let kernel: KernelPrimitive<M31Config> = compile_convert_to_binary().unwrap();
    let mut ctx: Context<M31Config, _> = Context::new(hint_registry);

    let a = M31::from(0x55);
    let a = ctx.copy_to_device(&a);
    let a = a.reshape(&[1]);
    let mut b: DeviceMemoryHandle = None;
    call_kernel!(ctx, kernel, 1, a, mut b).unwrap();
    let b = b.reshape(&[8]);
    let result: Vec<M31> = ctx.copy_to_host(b);
    assert_eq!(
        result,
        vec![
            M31::from(1),
            M31::from(0),
            M31::from(1),
            M31::from(0),
            M31::from(1),
            M31::from(0),
            M31::from(1),
            M31::from(0)
        ]
    );

    type P = Expander<M31Config>;
    let computation_graph = ctx.compile_computation_graph().unwrap();
    ctx.solve_witness().unwrap();
    println!("{:?}", computation_graph);
    println!("{:?}", ctx.export_device_memories());
    let (prover_setup, verifier_setup) = P::setup(&computation_graph);
    let proof = P::prove(
        &prover_setup,
        &computation_graph,
        ctx.export_device_memories(),
    );
    assert!(P::verify(&verifier_setup, &computation_graph, &proof));
}

#[kernel]
fn assertion<C: Config>(api: &mut API<C>, a: &InputVariable, b: &InputVariable) {
    api.assert_is_equal(*a, *b);
}

#[test]
fn zkcuda_assertion() {
    let kernel_tmp: KernelPrimitive<M31Config> = compile_assertion().unwrap();

    let mut ctx: Context<M31Config> = Context::default();
    let a = ctx.copy_to_device(&M31::from(10u32)).reshape(&[1]);
    let b = ctx.copy_to_device(&M31::from(10u32)).reshape(&[1]);
    call_kernel!(ctx, kernel_tmp, 1, a, b).unwrap();

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

#[test]
#[should_panic]
fn zkcuda_assertion_fail() {
    let kernel_tmp: KernelPrimitive<M31Config> = compile_assertion().unwrap();

    let mut ctx: Context<M31Config> = Context::default();
    let a = ctx.copy_to_device(&M31::from(10u32)).reshape(&[1]);
    let b = ctx.copy_to_device(&M31::from(9u32)).reshape(&[1]);
    call_kernel!(ctx, kernel_tmp, 1, a, b).unwrap();

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
