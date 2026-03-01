use arith::Field;
use arith::SimdField;
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::MPIConfig;
use rand::SeedableRng;

declare_circuit!(Circuit {
    s: [Variable; 100],
    sum: PublicVariable
});

impl<C: Config> Define<C> for Circuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let mut sum = api.constant(0);
        for x in self.s.iter() {
            sum = api.add(sum, x);
        }
        api.assert_is_equal(sum, self.sum);
    }
}

fn example<C: Config>() {
    let n_witnesses = SIMDField::<C>::PACK_SIZE;
    println!("n_witnesses: {}", n_witnesses);
    let compile_result: CompileResult<C> =
        compile(&Circuit::default(), CompileOptions::default()).unwrap();
    let mut s = [CircuitField::<C>::zero(); 100];
    let mut rng = rand::rngs::StdRng::seed_from_u64(1235);
    for i in 0..s.len() {
        s[i] = CircuitField::<C>::random_unsafe(&mut rng);
    }
    let assignment = Circuit::<CircuitField<C>> {
        s,
        sum: s.iter().sum(),
    };
    let assignments = vec![assignment; n_witnesses];
    let witness = compile_result
        .witness_solver
        .solve_witnesses(&assignments)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    for x in output.iter() {
        assert!(*x);
    }

    let mut expander_circuit = compile_result.layered_circuit.export_to_expander_flatten();

    let mpi_config = MPIConfig::prover_new();

    let (simd_input, simd_public_input) = witness.to_simd();
    println!("{} {}", simd_input.len(), simd_public_input.len());
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input = simd_public_input.clone();

    // prove
    expander_circuit.evaluate();
    let (claimed_v, proof) = executor::prove::<C>(&mut expander_circuit, mpi_config.clone());

    // verify
    assert!(executor::verify::<C>(
        &mut expander_circuit,
        mpi_config,
        &proof,
        &claimed_v
    ));
}

#[test]
fn example_gf2() {
    example::<GF2Config>();
}

#[test]
fn example_m31() {
    example::<M31Config>();
}

#[test]
fn example_goldilocks() {
    example::<GoldilocksConfig>();
}

#[test]
fn example_bn254() {
    example::<BN254Config>();
}

#[test]
fn example_babybear() {
    example::<BabyBearConfig>();
}
