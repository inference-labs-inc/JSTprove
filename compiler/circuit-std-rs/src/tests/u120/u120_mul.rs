use std::mem::transmute;

use expander_compiler::frontend::*;
use halo2curves::bn256::Fr;

use crate::{u120, BN_TWO_TO_120, MASK120};

declare_circuit!(MulCircuit {
    x: Variable,
    y: Variable,
    carry_in: Variable,
    carry_out: Variable,
    result: Variable,
});

impl Define<BN254Config> for MulCircuit<Variable> {
    fn define<Builder: RootAPI<BN254Config>>(&self, builder: &mut Builder) {
        let two_to_120 = builder.constant(BN_TWO_TO_120);

        let (result, carry_out) =
            u120::mul_u120(&self.x, &self.y, &self.carry_in, &two_to_120, builder);

        builder.assert_is_equal(result, self.result);
        builder.assert_is_equal(carry_out, self.carry_out);
    }
}

impl MulCircuit<Fr> {
    // Helper function to create circuit instance with given inputs
    fn create_circuit(
        x: [u64; 2],
        y: [u64; 2],
        carry_in: [u64; 2],
        carry_out: [u64; 2],
        result: [u64; 2],
    ) -> MulCircuit<Fr> {
        Self {
            x: Fr::from_raw([x[0], x[1], 0, 0]),
            y: Fr::from_raw([y[0], y[1], 0, 0]),
            carry_in: Fr::from_raw([carry_in[0], carry_in[1], 0, 0]),
            carry_out: Fr::from_raw([carry_out[0], carry_out[1], 0, 0]),
            result: Fr::from_raw([result[0], result[1], 0, 0]),
        }
    }
}

#[test]
fn test_rsa_circuit_120_multiplication() {
    let compile_result = compile(&MulCircuit::default(), CompileOptions::default()).unwrap();

    {
        // Test case: Simple multiplication without carries
        let x = [5, 0];
        let y = [4, 0];
        let carry_in = [0, 0];
        let result = [20, 0];
        let carry_out = [0, 0];

        let assignment = MulCircuit::<Fr>::create_circuit(x, y, carry_in, carry_out, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: Multiplication with carry
        let x = [1 << 63, 0]; // 2^63
        let y = [2, 0]; // 2
        let carry_in = [0, 0];
        let result = [0, 1]; // 2^64
        let carry_out = [0, 0];

        let assignment = MulCircuit::<Fr>::create_circuit(x, y, carry_in, carry_out, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: Multiplication with 120-bit overflow
        let x = unsafe { transmute(MASK120) }; // 2^120 - 1
        let y = [2, 0]; // 2
        let carry_in = [0, 0];
        let result = unsafe { transmute(MASK120 - 1) }; // 2^120 - 2
        let carry_out = [1, 0]; // 1

        let assignment = MulCircuit::<Fr>::create_circuit(x, y, carry_in, carry_out, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: Multiplication with carry_in
        let x = [10, 0];
        let y = [5, 0];
        let carry_in = [2, 0];
        let result = [52, 0]; // 10 * 5 + 2 = 52
        let carry_out = [0, 0];

        let assignment = MulCircuit::<Fr>::create_circuit(x, y, carry_in, carry_out, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: Large numbers multiplication
        let x = [u64::MAX >> 1, 0]; // Maximum value that won't overflow
        let y = [2, 0];
        let carry_in = [0, 0];
        let result = [u64::MAX - 1, 0];
        let carry_out = [0, 0];

        let assignment = MulCircuit::<Fr>::create_circuit(x, y, carry_in, carry_out, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: Large numbers multiplication
        let x = unsafe { transmute(MASK120) }; // 2^120 - 1
        let y = unsafe { transmute(MASK120) }; // 2^120 - 1
        let carry_in = unsafe { transmute(MASK120) }; // 2^120 - 1
        let result = [0, 0];
        let carry_out = unsafe { transmute(1329227995784915872903807060280344575u128) };

        let assignment = MulCircuit::<Fr>::create_circuit(x, y, carry_in, carry_out, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: Negative case (incorrect result)
        let x = [5, 0];
        let y = [4, 0];
        let carry_in = [0, 0];
        let result = [21, 0]; // Incorrect result (should be 20)
        let carry_out = [0, 0];

        let assignment = MulCircuit::<Fr>::create_circuit(x, y, carry_in, carry_out, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]);
    }
}
