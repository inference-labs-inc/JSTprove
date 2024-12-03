use arith::Field;
use expander_compiler::frontend::*;
use expander_config::{
    BN254ConfigKeccak, BN254ConfigSha2, GF2ExtConfigKeccak, GF2ExtConfigSha2, M31ExtConfigKeccak,
    M31ExtConfigSha2,
};

declare_circuit!(Circuit {
    s: [Variable; 100],
    sum: PublicVariable
});

impl<C: Config> Define<C> for Circuit<Variable> {
    fn define(&self, builder: &mut API<C>) {
        let mut sum = builder.constant(0);
        for x in self.s.iter() {
            sum = builder.add(sum, x);
        }
        builder.assert_is_equal(sum, self.sum);
    }
}

fn example<C: Config, GKRC>()
where
    GKRC: expander_config::GKRConfig<CircuitField = C::CircuitField>,
{
    let n_witnesses = <GKRC::SimdCircuitField as arith::SimdField>::pack_size();
    println!("n_witnesses: {}", n_witnesses);
    let compile_result: CompileResult<C> = compile(&Circuit::default()).unwrap();
    let mut s = [C::CircuitField::zero(); 100];
    for i in 0..s.len() {
        s[i] = C::CircuitField::random_unsafe(&mut rand::thread_rng());
    }
    let assignment = Circuit::<C::CircuitField> {
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
        assert_eq!(*x, true);
    }

    let mut expander_circuit = compile_result
        .layered_circuit
        .export_to_expander::<GKRC>()
        .flatten();
    let config = expander_config::Config::<GKRC>::new(
        expander_config::GKRScheme::Vanilla,
        expander_config::MPIConfig::new(),
    );

    let (simd_input, simd_public_input) = witness.to_simd::<GKRC::SimdCircuitField>();
    println!("{} {}", simd_input.len(), simd_public_input.len());
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input = simd_public_input.clone();

    // prove
    expander_circuit.evaluate();
    let mut prover = gkr::Prover::new(&config);
    prover.prepare_mem(&expander_circuit);
    let (claimed_v, proof) = prover.prove(&mut expander_circuit);

    // verify
    let verifier = gkr::Verifier::new(&config);
    assert!(verifier.verify(
        &mut expander_circuit,
        &simd_public_input,
        &claimed_v,
        &proof
    ));
}

#[test]
fn example_gf2() {
    example::<GF2Config, GF2ExtConfigSha2>();
    example::<GF2Config, GF2ExtConfigKeccak>();
}

#[test]
fn example_m31() {
    example::<M31Config, M31ExtConfigSha2>();
    example::<M31Config, M31ExtConfigKeccak>();
}

#[test]
fn example_bn254() {
    example::<BN254Config, BN254ConfigSha2>();
    example::<BN254Config, BN254ConfigKeccak>();
}
