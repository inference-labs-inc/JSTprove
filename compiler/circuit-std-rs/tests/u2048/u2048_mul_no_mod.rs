use expander_compiler::frontend::*;
use halo2curves::bn256::Fr;

use circuit_std_rs::u2048::U2048Variable;
use circuit_std_rs::{BN_TWO_TO_120, N_LIMBS};

declare_circuit!(MulNoModCircuit {
    x: [Variable; N_LIMBS],
    y: [Variable; N_LIMBS],
    result: [Variable; 2 * N_LIMBS],
});

impl Define<BN254Config> for MulNoModCircuit<Variable> {
    fn define<Builder: RootAPI<BN254Config>>(&self, builder: &mut Builder) {
        let x = U2048Variable::from_raw(self.x);
        let y = U2048Variable::from_raw(self.y);
        let two_to_120 = builder.constant(BN_TWO_TO_120);

        let res = U2048Variable::mul_without_mod_reduction(&x, &y, &two_to_120, builder);

        // builder.display("first limb", x.limbs[0]);

        for i in 0..2 * N_LIMBS {
            builder.assert_is_equal(res[i], self.result[i]);
        }
    }
}

impl MulNoModCircuit<Fr> {
    fn create_circuit(
        x: [[u64; 2]; N_LIMBS],
        y: [[u64; 2]; N_LIMBS],
        result: [[u64; 2]; 2 * N_LIMBS],
    ) -> MulNoModCircuit<Fr> {
        let mut x_limbs = [Fr::zero(); N_LIMBS];
        let mut y_limbs = [Fr::zero(); N_LIMBS];
        let mut result_limbs = [Fr::zero(); 2 * N_LIMBS];

        for i in 0..N_LIMBS {
            x_limbs[i] = Fr::from_raw([x[i][0], x[i][1], 0, 0]);
            y_limbs[i] = Fr::from_raw([y[i][0], y[i][1], 0, 0]);
        }

        for i in 0..2 * N_LIMBS {
            result_limbs[i] = Fr::from_raw([result[i][0], result[i][1], 0, 0]);
        }

        Self {
            x: x_limbs,
            y: y_limbs,
            result: result_limbs,
        }
    }
}

#[test]
fn test_mul_without_mod() {
    let compile_result = compile(&MulNoModCircuit::default(), CompileOptions::default()).unwrap();

    {
        // Test case 1: Simple multiplication with no carries
        let mut x = [[0, 0]; N_LIMBS];
        x[0] = [5, 0];
        let mut y = [[0, 0]; N_LIMBS];
        y[0] = [3, 0];
        let mut result = [[0, 0]; 2 * N_LIMBS];
        result[0] = [15, 0]; // 5 * 3 = 15

        let assignment = MulNoModCircuit::<Fr>::create_circuit(x, y, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case 2: Multiplication with carry in lower limb
        let mut x = [[0, 0]; N_LIMBS];
        x[0] = [(1u64 << 60), 0]; // Large number in first limb
        let mut y = [[0, 0]; N_LIMBS];
        y[0] = [2, 0];
        let mut result = [[0, 0]; 2 * N_LIMBS];
        result[0] = [(1u64 << 61), 0]; // 2^60 * 2 = 2^61

        let assignment = MulNoModCircuit::<Fr>::create_circuit(x, y, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case 3: Cross-limb multiplication
        let mut x = [[0, 0]; N_LIMBS];
        x[0] = [1, 0];
        x[1] = [1, 0];
        let mut y = [[0, 0]; N_LIMBS];
        y[0] = [1, 0];
        y[1] = [1, 0];
        let mut result = [[0, 0]; 2 * N_LIMBS];
        result[0] = [1, 0]; // 1*1
        result[1] = [2, 0]; // 1*1 + 1*1
        result[2] = [1, 0]; // 1*1

        let assignment = MulNoModCircuit::<Fr>::create_circuit(x, y, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case 4: Multiplication near 2^120 boundary
        let mut x = [[0, 0]; N_LIMBS];
        x[0] = [0, 1]; // 2^64
        let mut y = [[0, 0]; N_LIMBS];
        y[0] = [2, 0];
        let mut result = [[0, 0]; 2 * N_LIMBS];
        result[0] = [0, 2]; // 2^64 * 2 = 2^65

        let assignment = MulNoModCircuit::<Fr>::create_circuit(x, y, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case 5: Multiplication by zero
        let mut x = [[0, 0]; N_LIMBS];
        x[0] = [5, 0];
        let y = [[0, 0]; N_LIMBS];
        let result = [[0, 0]; 2 * N_LIMBS];

        let assignment = MulNoModCircuit::<Fr>::create_circuit(x, y, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case 6: Multiple limb interaction
        let mut x = [[0, 0]; N_LIMBS];
        x[0] = [1 << 32, 0];
        x[1] = [1, 0];
        let mut y = [[0, 0]; N_LIMBS];
        y[0] = [1 << 32, 0];
        y[1] = [1, 0];
        let mut result = [[0, 0]; 2 * N_LIMBS];
        result[0] = [0, 1]; // (2^32 * 2^32)
        result[1] = [2 << 32, 0]; // (2^32 * 1 + 1 * 2^32)
        result[2] = [1, 0]; // (1 * 1)

        let assignment = MulNoModCircuit::<Fr>::create_circuit(x, y, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case 7: Multiple limb interaction
        let mut x = [[0, 0]; N_LIMBS];
        x[0] = [1 << 32, 0];
        x[N_LIMBS - 1] = [1, 0];
        let mut y = [[0, 0]; N_LIMBS];
        y[0] = [1 << 32, 0];
        y[N_LIMBS - 1] = [1, 0];
        let mut result = [[0, 0]; 2 * N_LIMBS];
        result[0] = [0, 1]; // (2^32 * 2^32)
        result[N_LIMBS - 1] = [0x200000000, 0];
        result[N_LIMBS * 2 - 2] = [1, 0];

        let assignment = MulNoModCircuit::<Fr>::create_circuit(x, y, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case 8: Large numbers
        let x = [[1, 0]; N_LIMBS];
        let y = [[2, 0]; N_LIMBS];
        let mut result = [[0u64, 0]; 2 * N_LIMBS];

        // Hard coded result from (a*b).hex()
        result[0] = [2, 0]; // 0x2
        result[1] = [4, 0]; // 0x4
        result[2] = [6, 0]; // 0x6
        result[3] = [8, 0]; // 0x8
        result[4] = [10, 0]; // 0xa
        result[5] = [12, 0]; // 0xc
        result[6] = [14, 0]; // 0xe
        result[7] = [16, 0]; // 0x10
        result[8] = [18, 0]; // 0x12
        result[9] = [20, 0]; // 0x14
        result[10] = [22, 0]; // 0x16
        result[11] = [24, 0]; // 0x18
        result[12] = [26, 0]; // 0x1a
        result[13] = [28, 0]; // 0x1c
        result[14] = [30, 0]; // 0x1e
        result[15] = [32, 0]; // 0x20
        result[16] = [34, 0]; // 0x22
        result[17] = [36, 0]; // 0x24
        result[18] = [34, 0]; // 0x22
        result[19] = [32, 0]; // 0x20
        result[20] = [30, 0]; // 0x1e
        result[21] = [28, 0]; // 0x1c
        result[22] = [26, 0]; // 0x1a
        result[23] = [24, 0]; // 0x18
        result[24] = [22, 0]; // 0x16
        result[25] = [20, 0]; // 0x14
        result[26] = [18, 0]; // 0x12
        result[27] = [16, 0]; // 0x10
        result[28] = [14, 0]; // 0xe
        result[29] = [12, 0]; // 0xc
        result[30] = [10, 0]; // 0xa
        result[31] = [8, 0]; // 0x8
        result[32] = [6, 0]; // 0x6
        result[33] = [4, 0]; // 0x4
        result[34] = [2, 0]; // 0x2

        let assignment = MulNoModCircuit::<Fr>::create_circuit(x, y, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    // Negative test cases
    {
        // Test case 9: Incorrect result
        let mut x = [[0, 0]; N_LIMBS];
        x[0] = [5, 0];
        let mut y = [[0, 0]; N_LIMBS];
        y[0] = [3, 0];
        let mut result = [[0, 0]; 2 * N_LIMBS];
        result[0] = [16, 0]; // Wrong result (should be 15)

        let assignment = MulNoModCircuit::<Fr>::create_circuit(x, y, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]);
    }

    {
        // Test case 10: Missing carry
        let mut x = [[0, 0]; N_LIMBS];
        x[0] = [(1u64 << 63), 0];
        let mut y = [[0, 0]; N_LIMBS];
        y[0] = [2, 0];
        let mut result = [[0, 0]; 2 * N_LIMBS];
        result[0] = [0, 0];
        // Missing result[1] = [1, 0] for the carry

        let assignment = MulNoModCircuit::<Fr>::create_circuit(x, y, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]);
    }
}
