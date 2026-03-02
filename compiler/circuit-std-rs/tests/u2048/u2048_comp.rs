use expander_compiler::frontend::*;
use halo2curves::bn256::Fr;

use circuit_std_rs::u2048::U2048Variable;
use circuit_std_rs::N_LIMBS;

declare_circuit!(CompareCircuit {
    x: [Variable; N_LIMBS],
    y: [Variable; N_LIMBS],
    result: Variable,
});

impl Define<BN254Config> for CompareCircuit<Variable> {
    fn define<Builder: RootAPI<BN254Config>>(&self, builder: &mut Builder) {
        let x = U2048Variable { limbs: self.x };
        let y = U2048Variable { limbs: self.y };

        let comparison_result = x.assert_is_less_than(&y, builder);
        builder.assert_is_equal(comparison_result, self.result);
    }
}

impl CompareCircuit<Fr> {
    fn create_circuit(x: Vec<u64>, y: Vec<u64>, expected_result: bool) -> CompareCircuit<Fr> {
        assert_eq!(x.len(), N_LIMBS);
        assert_eq!(y.len(), N_LIMBS);

        let mut x_limbs = [Fr::zero(); N_LIMBS];
        let mut y_limbs = [Fr::zero(); N_LIMBS];

        for i in 0..N_LIMBS {
            x_limbs[i] = Fr::from(x[i]);
            y_limbs[i] = Fr::from(y[i]);
        }

        Self {
            x: x_limbs,
            y: y_limbs,
            result: Fr::from(expected_result as u64),
        }
    }
}

#[test]
fn test_u2048_comparison() {
    let compile_result = compile(&CompareCircuit::default(), CompileOptions::default()).unwrap();

    {
        // Test case: Equal numbers
        let x = vec![5; N_LIMBS];
        let y = vec![5; N_LIMBS];

        let assignment = CompareCircuit::<Fr>::create_circuit(x, y, false); // x < y is false when equal
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case: Less than in most significant limb
        let mut x = vec![0; N_LIMBS];
        let mut y = vec![0; N_LIMBS];
        x[N_LIMBS - 1] = 5;
        y[N_LIMBS - 1] = 10;

        let assignment = CompareCircuit::<Fr>::create_circuit(x, y, true); // x < y is true
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case: Greater in most significant limb
        let mut x = vec![0; N_LIMBS];
        let mut y = vec![0; N_LIMBS];
        x[N_LIMBS - 1] = 10;
        y[N_LIMBS - 1] = 5;

        let assignment = CompareCircuit::<Fr>::create_circuit(x, y, false); // x < y is false
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case: Equal in all limbs except least significant
        let mut x = vec![1; N_LIMBS];
        let mut y = vec![1; N_LIMBS];
        x[0] = 5;
        y[0] = 6;

        let assignment = CompareCircuit::<Fr>::create_circuit(x, y, true); // x < y is true
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case: Equal in most significant limb, less than in next limb
        let mut x = vec![0; N_LIMBS];
        let mut y = vec![0; N_LIMBS];
        x[N_LIMBS - 1] = 5;
        y[N_LIMBS - 1] = 5;
        x[N_LIMBS - 2] = 5;
        y[N_LIMBS - 2] = 10;

        let assignment = CompareCircuit::<Fr>::create_circuit(x, y, true); // x < y is true
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    // Negative test cases
    {
        // Negative test: Claiming x < y when x > y
        let mut x = vec![0; N_LIMBS];
        let mut y = vec![0; N_LIMBS];
        x[N_LIMBS - 1] = 10;
        y[N_LIMBS - 1] = 5;

        let assignment = CompareCircuit::<Fr>::create_circuit(x, y, true); // incorrect result
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]); // should fail
    }

    {
        // Negative test: Claiming x < y when x = y
        let x = vec![5; N_LIMBS];
        let y = vec![5; N_LIMBS];

        let assignment = CompareCircuit::<Fr>::create_circuit(x, y, true); // incorrect result
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]); // should fail
    }

    {
        // Test case: Equal in most significant limb, comparison in lower limb
        let mut x = vec![0; N_LIMBS];
        let mut y = vec![0; N_LIMBS];
        x[N_LIMBS - 1] = 5;
        y[N_LIMBS - 1] = 5;
        x[N_LIMBS - 2] = 4;
        y[N_LIMBS - 2] = 5;

        let assignment = CompareCircuit::<Fr>::create_circuit(x, y, true); // x < y is true
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
}
