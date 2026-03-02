use expander_compiler::frontend::*;
use halo2curves::bn256::Fr;

use circuit_std_rs::u2048::U2048Variable;
use circuit_std_rs::{BN_TWO_TO_120, N_LIMBS};

declare_circuit!(AddModCircuit {
    x: [Variable; N_LIMBS],
    y: [Variable; N_LIMBS],
    result: [Variable; N_LIMBS],
    carry: Variable,
    modulus: [Variable; N_LIMBS],
});

impl Define<BN254Config> for AddModCircuit<Variable> {
    fn define<Builder: RootAPI<BN254Config>>(&self, builder: &mut Builder) {
        let x = U2048Variable::from_raw(self.x);
        let y = U2048Variable::from_raw(self.y);
        let result = U2048Variable::from_raw(self.result);
        let modulus = U2048Variable::from_raw(self.modulus);
        let two_to_120 = builder.constant(BN_TWO_TO_120);

        U2048Variable::assert_add(&x, &y, &result, &self.carry, &modulus, &two_to_120, builder);
    }
}

impl AddModCircuit<Fr> {
    fn create_circuit(
        x: [[u64; 2]; N_LIMBS],
        y: [[u64; 2]; N_LIMBS],
        result: [[u64; 2]; N_LIMBS],
        carry: u64,
        modulus: [[u64; 2]; N_LIMBS],
    ) -> AddModCircuit<Fr> {
        let mut x_limbs = [Fr::zero(); N_LIMBS];
        let mut y_limbs = [Fr::zero(); N_LIMBS];
        let mut result_limbs = [Fr::zero(); N_LIMBS];
        let mut modulus_limbs = [Fr::zero(); N_LIMBS];

        for i in 0..N_LIMBS {
            x_limbs[i] = Fr::from_raw([x[i][0], x[i][1], 0, 0]);
            y_limbs[i] = Fr::from_raw([y[i][0], y[i][1], 0, 0]);
            result_limbs[i] = Fr::from_raw([result[i][0], result[i][1], 0, 0]);
            modulus_limbs[i] = Fr::from_raw([modulus[i][0], modulus[i][1], 0, 0]);
        }

        Self {
            x: x_limbs,
            y: y_limbs,
            result: result_limbs,
            carry: Fr::from(carry),
            modulus: modulus_limbs,
        }
    }
}

#[test]
fn test_mod_add() {
    let compile_result = compile(&AddModCircuit::default(), CompileOptions::default()).unwrap();

    {
        // Test case: Simple addition without mod reduction
        let x = [[5, 0]; N_LIMBS];
        let y = [[3, 0]; N_LIMBS];
        let result = [[8, 0]; N_LIMBS];
        let modulus = [[10, 0]; N_LIMBS];

        let assignment = AddModCircuit::<Fr>::create_circuit(x, y, result, 0, modulus);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case: Addition with carry between limbs (like u64::MAX + u64::MAX)
        let mut x = [[0, 0]; N_LIMBS];
        let mut y = [[0, 0]; N_LIMBS];
        let mut result = [[0, 0]; N_LIMBS];
        let mut modulus = [[0, 0]; N_LIMBS];

        x[0] = [u64::MAX, 0];
        y[0] = [u64::MAX, 0];
        result[0] = [u64::MAX - 1, 1];
        modulus[1] = [0, 1]; // Large modulus to avoid reduction

        let assignment = AddModCircuit::<Fr>::create_circuit(x, y, result, 0, modulus);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case: Addition with modular reduction
        let mut x = [[0, 0]; N_LIMBS];
        let mut y = [[0, 0]; N_LIMBS];
        let mut result = [[0, 0]; N_LIMBS];
        let mut modulus = [[0, 0]; N_LIMBS];

        x[0] = [7, 0];
        y[0] = [5, 0];
        result[0] = [2, 0];
        modulus[0] = [10, 0];

        let assignment = AddModCircuit::<Fr>::create_circuit(x, y, result, 1, modulus);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case: Invalid carry value
        let x = [[5, 0]; N_LIMBS];
        let y = [[3, 0]; N_LIMBS];
        let result = [[8, 0]; N_LIMBS];
        let modulus = [[10, 0]; N_LIMBS];

        let assignment = AddModCircuit::<Fr>::create_circuit(x, y, result, 2, modulus); // carry > 1
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]);
    }

    {
        // Negative test: result >= modulus
        let mut x = [[0, 0]; N_LIMBS];
        let mut y = [[0, 0]; N_LIMBS];
        let mut result = [[0, 0]; N_LIMBS];
        let mut modulus = [[0, 0]; N_LIMBS];

        x[0] = [7, 0];
        y[0] = [5, 0];
        result[0] = [12, 0]; // result > modulus
        modulus[0] = [10, 0];

        let assignment = AddModCircuit::<Fr>::create_circuit(x, y, result, 0, modulus);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]);
    }
}
