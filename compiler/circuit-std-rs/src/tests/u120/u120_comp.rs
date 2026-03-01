use expander_compiler::frontend::*;
use halo2curves::bn256::Fr;
use std::mem::transmute;

use crate::u120::is_less_than_u120;
use crate::MASK120;

declare_circuit!(LessThanCircuit {
    x: Variable,
    y: Variable,
    result: Variable,
});

impl Define<BN254Config> for LessThanCircuit<Variable> {
    fn define<Builder: RootAPI<BN254Config>>(&self, builder: &mut Builder) {
        let res = is_less_than_u120(&self.x, &self.y, builder);
        builder.assert_is_equal(res, self.result);
    }
}

impl LessThanCircuit<Fr> {
    fn create_circuit(x: [u64; 2], y: [u64; 2], result: [u64; 2]) -> LessThanCircuit<Fr> {
        Self {
            x: Fr::from_raw([x[0], x[1], 0, 0]),
            y: Fr::from_raw([y[0], y[1], 0, 0]),
            result: Fr::from_raw([result[0], result[1], 0, 0]),
        }
    }
}

#[test]
fn test_u120_less_than() {
    let compile_result = compile(&LessThanCircuit::default(), CompileOptions::default()).unwrap();

    {
        // Test case: Simple less than
        let x = [5, 0];
        let y = [10, 0];
        let result = [1, 0]; // true: 5 < 10

        let assignment = LessThanCircuit::<Fr>::create_circuit(x, y, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: Equal values
        let x = [42, 0];
        let y = [42, 0];
        let result = [0, 0]; // false: 42 = 42

        let assignment = LessThanCircuit::<Fr>::create_circuit(x, y, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: Greater than
        let x = [100, 0];
        let y = [50, 0];
        let result = [0, 0]; // false: 100 > 50

        let assignment = LessThanCircuit::<Fr>::create_circuit(x, y, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: Using second limb
        let x = [0, 1]; // 2^64
        let y = [u64::MAX, 0];
        let result = [0, 0]; // false: 2^64 > u64::MAX

        let assignment = LessThanCircuit::<Fr>::create_circuit(x, y, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: Large numbers near 120-bit limit
        let x = unsafe { transmute(MASK120 - 1) }; // 2^120 - 2
        let y = unsafe { transmute(MASK120) }; // 2^120 - 1
        let result = [1, 0]; // true: (2^120 - 2) < (2^120 - 1)

        let assignment = LessThanCircuit::<Fr>::create_circuit(x, y, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: Equal large numbers
        let x = unsafe { transmute(MASK120) }; // 2^120 - 1
        let y = unsafe { transmute(MASK120) }; // 2^120 - 1
        let result = [0, 0]; // false: equal values

        let assignment = LessThanCircuit::<Fr>::create_circuit(x, y, result);
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
        let y = [10, 0];
        let result = [0, 0]; // incorrect: should be 1 since 5 < 10

        let assignment = LessThanCircuit::<Fr>::create_circuit(x, y, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]);
    }
    {
        // Test case: Negative case (incorrect result)
        let x = [5, 0];
        let y = [5, 0];
        let result = [1, 0]; // incorrect: should be 0 since 5 = 5

        let assignment = LessThanCircuit::<Fr>::create_circuit(x, y, result);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]);
    }
}
