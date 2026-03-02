use circuit_std_rs::{poseidon::poseidon_m31::*, poseidon::utils::*, utils::register_hint};
use expander_compiler::frontend::*;

declare_circuit!(PoseidonSpongeLen8Circuit {
    inputs: [Variable; 8],
    outputs: [Variable; 16]
});

impl Define<M31Config> for PoseidonSpongeLen8Circuit<Variable> {
    fn define<Builder: RootAPI<M31Config>>(&self, builder: &mut Builder) {
        let params = PoseidonM31Params::new(
            builder,
            POSEIDON_M31X16_RATE,
            16,
            POSEIDON_M31X16_FULL_ROUNDS,
            POSEIDON_M31X16_PARTIAL_ROUNDS,
        );
        let res = params.hash_to_state(builder, &self.inputs);
        (0..params.width).for_each(|i| builder.assert_is_equal(res[i], self.outputs[i]));
    }
}

#[test]
// NOTE(HS) Poseidon Mersenne-31 Width-16 Sponge tested over input length 8
fn test_poseidon_m31x16_hash_to_state_input_len8() {
    let compile_result = compile(
        &PoseidonSpongeLen8Circuit::default(),
        CompileOptions::default(),
    )
    .unwrap();

    let assignment = PoseidonSpongeLen8Circuit::<M31> {
        inputs: [M31::from(114514); 8],
        outputs: [
            M31 { v: 1021105124 },
            M31 { v: 1342990709 },
            M31 { v: 1593716396 },
            M31 { v: 2100280498 },
            M31 { v: 330652568 },
            M31 { v: 1371365483 },
            M31 { v: 586650367 },
            M31 { v: 345482939 },
            M31 { v: 849034538 },
            M31 { v: 175601510 },
            M31 { v: 1454280121 },
            M31 { v: 1362077584 },
            M31 { v: 528171622 },
            M31 { v: 187534772 },
            M31 { v: 436020341 },
            M31 { v: 1441052621 },
        ],
    };
    let witness = compile_result
        .witness_solver
        .solve_witness(&assignment)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![true]);
}

declare_circuit!(PoseidonSpongeLen16Circuit {
    inputs: [Variable; 16],
    outputs: [Variable; 16]
});

impl Define<M31Config> for PoseidonSpongeLen16Circuit<Variable> {
    fn define<Builder: RootAPI<M31Config>>(&self, builder: &mut Builder) {
        let params = PoseidonM31Params::new(
            builder,
            POSEIDON_M31X16_RATE,
            16,
            POSEIDON_M31X16_FULL_ROUNDS,
            POSEIDON_M31X16_PARTIAL_ROUNDS,
        );
        let res = params.hash_to_state_flatten(builder, &self.inputs);
        (0..params.width).for_each(|i| builder.assert_is_equal(res[i], self.outputs[i]));
    }
}

#[test]
// NOTE(HS) Poseidon Mersenne-31 Width-16 Sponge tested over input length 16
fn test_poseidon_m31x16_hash_to_state_input_len16() {
    let compile_result = compile(
        &PoseidonSpongeLen16Circuit::default(),
        CompileOptions::default(),
    )
    .unwrap();
    let mut hint_registry = HintRegistry::<M31>::new();
    register_hint(&mut hint_registry);
    let assignment = PoseidonSpongeLen16Circuit::<M31> {
        inputs: [M31::from(114514); 16],
        outputs: [
            M31 { v: 1510043913 },
            M31 { v: 1840611937 },
            M31 { v: 45881205 },
            M31 { v: 1134797377 },
            M31 { v: 803058407 },
            M31 { v: 1772167459 },
            M31 { v: 846553905 },
            M31 { v: 2143336151 },
            M31 { v: 300871060 },
            M31 { v: 545838827 },
            M31 { v: 1603101164 },
            M31 { v: 396293243 },
            M31 { v: 502075988 },
            M31 { v: 2067011878 },
            M31 { v: 402134378 },
            M31 { v: 535675968 },
        ],
    };
    let witness = compile_result
        .witness_solver
        .solve_witness_with_hints(&assignment, &mut hint_registry)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![true]);
}
