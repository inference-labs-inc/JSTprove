mod common;

use circuit_std_rs::{
    logup::{query_count_hint, rangeproof_hint, LogUpRangeProofTable},
    LogUpCircuit, LogUpParams,
};
use expander_compiler::{
    field::{BN254Fr, Goldilocks},
    frontend::*,
    zkcuda::{context::*, kernel::*, proving_system::*, shape::Reshape},
};

#[test]
fn logup_test() {
    let logup_params = LogUpParams {
        key_len: 7,
        value_len: 7,
        n_table_rows: 123,
        n_queries: 456,
    };

    common::circuit_test_helper::<BN254Config, LogUpCircuit>(&logup_params);
    common::circuit_test_helper::<M31Config, LogUpCircuit>(&logup_params);
    common::circuit_test_helper::<GF2Config, LogUpCircuit>(&logup_params);
}

declare_circuit!(LogUpRangeproofCircuit { test: Variable });
impl Define<M31Config> for LogUpRangeproofCircuit<Variable> {
    fn define<Builder: RootAPI<M31Config>>(&self, builder: &mut Builder) {
        let mut table = LogUpRangeProofTable::new(8);
        table.initial(builder);
        for i in 1..12 {
            for j in (1 << (i - 1))..(1 << i) {
                let key = builder.constant(j);
                if i > 8 {
                    table.rangeproof(builder, key, i);
                } else {
                    table.rangeproof_onechunk(builder, key, i);
                }
            }
        }
        table.final_check(builder);
    }
}

#[test]
fn rangeproof_logup_test() {
    let mut hint_registry = HintRegistry::<M31>::new();
    hint_registry.register("myhint.querycounthint", query_count_hint);
    hint_registry.register("myhint.rangeproofhint", rangeproof_hint);
    //compile and test
    let compile_result = compile(
        &LogUpRangeproofCircuit::default(),
        CompileOptions::default(),
    )
    .unwrap();
    let assignment = LogUpRangeproofCircuit { test: M31::from(0) };
    let witness = compile_result
        .witness_solver
        .solve_witness_with_hints(&assignment, &mut hint_registry)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![true]);
}

declare_circuit!(RangeproofCircuit {
    place_holder: Variable
});
impl Define<GoldilocksConfig> for RangeproofCircuit<Variable> {
    fn define<Builder: RootAPI<GoldilocksConfig>>(&self, builder: &mut Builder) {
        let mut table = LogUpRangeProofTable::new(16);
        table.initial(builder);

        // < 2^24 value
        let key = builder.constant(16777215);
        table.rangeproof(builder, key, 24);

        table.final_check(builder);
    }
}

#[test]
fn rangeproof_goldilocks_test() {
    let mut hint_registry = HintRegistry::<Goldilocks>::new();
    hint_registry.register("myhint.querycounthint", query_count_hint);
    hint_registry.register("myhint.rangeproofhint", rangeproof_hint);
    //compile and test
    let compile_result = compile(&RangeproofCircuit::default(), CompileOptions::default()).unwrap();
    let assignment = RangeproofCircuit {
        place_holder: Goldilocks::one(),
    };
    let witness = compile_result
        .witness_solver
        .solve_witness_with_hints(&assignment, &mut hint_registry)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![true]);
}

declare_circuit!(RangeproofLogupCircuit {
    _placeholder: Variable
});
impl Define<BN254Config> for RangeproofLogupCircuit<Variable> {
    fn define<Builder: RootAPI<BN254Config>>(&self, builder: &mut Builder) {
        let mut table = LogUpRangeProofTable::new(16);
        table.initial(builder);
        let key1 = builder.constant(18888888);
        table.rangeproof(builder, key1, 37);

        let key2 = builder.constant(58888888);
        table.rangeproof(builder, key2, 49);

        table.final_check(builder);
    }
}

#[test]
fn rangeproof_bn254_test() {
    let mut hint_registry = HintRegistry::<BN254Fr>::new();
    hint_registry.register("myhint.querycounthint", query_count_hint);
    hint_registry.register("myhint.rangeproofhint", rangeproof_hint);
    //compile and test
    let compile_result = compile(
        &RangeproofLogupCircuit::default(),
        CompileOptions::default(),
    )
    .unwrap();
    let assignment = RangeproofLogupCircuit {
        _placeholder: BN254Fr::zero(),
    };
    let witness = compile_result
        .witness_solver
        .solve_witness_with_hints(&assignment, &mut hint_registry)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![true]);
}

#[kernel]
fn rangeproof_test_kernel<C: Config>(builder: &mut API<C>, test: &InputVariable) {
    let mut table = LogUpRangeProofTable::new(8);
    table.initial(builder);
    table.rangeproof(builder, *test, 10);
    table.final_check(builder);
}

#[test]
fn rangeproof_zkcuda_test() {
    let mut hint_registry = HintRegistry::<M31>::new();
    hint_registry.register("myhint.querycounthint", query_count_hint);
    hint_registry.register("myhint.rangeproofhint", rangeproof_hint);
    //compile and test
    let kernel: KernelPrimitive<M31Config> = compile_rangeproof_test_kernel().unwrap();
    let mut ctx: Context<M31Config, _> = Context::new(hint_registry);

    let a = M31::from(1 << 9);
    let a = ctx.copy_to_device(&a);
    let a = a.reshape(&[1]);
    call_kernel!(ctx, kernel, 1, a).unwrap();

    type P = Expander<M31Config>;
    let computation_graph = ctx.compile_computation_graph().unwrap();
    ctx.solve_witness().unwrap();
    let (prover_setup, verifier_setup) = <P as ProvingSystem<M31Config>>::setup(&computation_graph);
    let proof = P::prove(
        &prover_setup,
        &computation_graph,
        ctx.export_device_memories(),
    );
    assert!(P::verify(&verifier_setup, &computation_graph, &proof));
}

#[test]
#[should_panic]
fn rangeproof_zkcuda_test_fail() {
    let mut hint_registry = HintRegistry::<M31>::new();
    hint_registry.register("myhint.querycounthint", query_count_hint);
    hint_registry.register("myhint.rangeproofhint", rangeproof_hint);
    //compile and test
    let kernel: KernelPrimitive<M31Config> = compile_rangeproof_test_kernel().unwrap();
    let mut ctx: Context<M31Config, _> = Context::new(hint_registry);

    let a = M31::from(1 << 11);
    let a = ctx.copy_to_device(&a);
    let a = a.reshape(&[1]);
    call_kernel!(ctx, kernel, 1, a).unwrap();

    type P = Expander<M31Config>;
    let computation_graph = ctx.compile_computation_graph().unwrap();
    ctx.solve_witness().unwrap();
    let (prover_setup, verifier_setup) = <P as ProvingSystem<M31Config>>::setup(&computation_graph);
    let proof = P::prove(
        &prover_setup,
        &computation_graph,
        ctx.export_device_memories(),
    );
    assert!(P::verify(&verifier_setup, &computation_graph, &proof));
}
