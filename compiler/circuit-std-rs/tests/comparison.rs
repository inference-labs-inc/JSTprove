use ark_std::test_rng;
use expander_compiler::{
    compile::CompileOptions,
    declare_circuit,
    frontend::{compile, BN254Config, CircuitField, Config, Define, M31Config, RootAPI, Variable},
};
use rand::RngCore;

declare_circuit!(ComparisonCircuit {
    a: Variable,
    b: Variable,
    c: Variable
});

impl<C: Config> Define<C> for ComparisonCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, builder: &mut Builder) {
        let r1 = builder.gt(self.a, self.b);
        let r2 = builder.geq(self.a, self.b);
        let r3 = builder.geq(self.a, self.c);

        builder.assert_is_bool(r1);
        builder.assert_is_non_zero(r1);

        builder.assert_is_bool(r2);
        builder.assert_is_non_zero(r2);

        builder.assert_is_bool(r3);
        builder.assert_is_non_zero(r3);
    }
}

#[test]
fn test_comp() {
    let mut rng = test_rng();
    let t = rng.next_u32() as i32;

    // positive tests
    test_comp_helper::<BN254Config>(true, 1, 0, 1);
    test_comp_helper::<M31Config>(true, 1, 0, 1);

    test_comp_helper::<BN254Config>(true, 1 << 20, 1 << 19, 1 << 20);
    test_comp_helper::<M31Config>(true, 1 << 20, 1 << 19, 1 << 20);

    test_comp_helper::<BN254Config>(true, 67890, 12345, 67890);
    test_comp_helper::<M31Config>(true, 67890, 12345, 67890);

    test_comp_helper::<BN254Config>(true, -1, -2, -1);
    test_comp_helper::<M31Config>(true, -1, -2, -1);

    test_comp_helper::<BN254Config>(true, t, t - 1, t);
    test_comp_helper::<M31Config>(true, t, t - 1, t);

    // negative tests
    test_comp_helper::<BN254Config>(false, 1, 1, 1);
    test_comp_helper::<M31Config>(false, 1, 1, 1);

    test_comp_helper::<BN254Config>(false, 1, 0, 2);
    test_comp_helper::<M31Config>(false, 1, 0, 2);

    test_comp_helper::<BN254Config>(false, t, t, t);
    test_comp_helper::<M31Config>(false, t, t - 1, t + 1);
}

fn test_comp_helper<C: Config>(sat: bool, a: i32, b: i32, c: i32) {
    let compile_result =
        compile::<C, _>(&ComparisonCircuit::default(), CompileOptions::default()).unwrap();

    let a = if a >= 0 {
        CircuitField::<C>::from(a as u32)
    } else {
        -CircuitField::<C>::from(-a as u32)
    };
    let b = if b >= 0 {
        CircuitField::<C>::from(b as u32)
    } else {
        -CircuitField::<C>::from(-b as u32)
    };
    let c = if c >= 0 {
        CircuitField::<C>::from(c as u32)
    } else {
        -CircuitField::<C>::from(-c as u32)
    };

    println!("a: {:?}, b: {:?}, c: {:?}", a, b, c);

    let assignment = ComparisonCircuit { a, b, c };

    let witness = compile_result
        .witness_solver
        .solve_witness(&assignment)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![sat]);
}
