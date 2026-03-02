use expander_compiler::frontend::*;
use halo2curves::bn256::Fr;

use crate::non_native::u120;
use crate::BN_TWO_TO_120;

declare_circuit!(AccumulateCircuit {
    inputs: [Variable],
    result: Variable,
    carry: Variable,
});

impl AccumulateCircuit<Variable> {
    fn dummy(num_inputs: usize) -> AccumulateCircuit<Variable> {
        let inputs = vec![Variable::default(); num_inputs];
        let result = Variable::default();
        let carry = Variable::default();
        Self {
            inputs,
            result,
            carry,
        }
    }
}

impl Define<BN254Config> for AccumulateCircuit<Variable> {
    fn define<Builder: RootAPI<BN254Config>>(&self, builder: &mut Builder) {
        println!("len: {}", self.inputs.len());
        let two_to_120 = builder.constant(BN_TWO_TO_120);
        let (result, carry) = u120::accumulate_u120(&self.inputs, &two_to_120, builder);

        builder.assert_is_equal(result, self.result);
        builder.assert_is_equal(carry, self.carry);
    }
}

impl AccumulateCircuit<Fr> {
    fn create_circuit(
        inputs: Vec<[u64; 2]>,
        result: [u64; 2],
        carry: [u64; 2],
    ) -> AccumulateCircuit<Fr> {
        println!("inputs: {:?}", inputs);
        let input_vars = inputs
            .into_iter()
            .map(|x| Fr::from_raw([x[0], x[1], 0, 0]))
            .collect();

        Self {
            inputs: input_vars,
            result: Fr::from_raw([result[0], result[1], 0, 0]),
            carry: Fr::from_raw([carry[0], carry[1], 0, 0]),
        }
    }
}

#[test]
fn test_accumulate_u120() {
    {
        let compile_result =
            compile(&AccumulateCircuit::dummy(4), CompileOptions::default()).unwrap();

        // Test case 1: Simple addition without carry
        let inputs = vec![[1, 0], [2, 0], [3, 0], [4, 0]];
        let result = [10, 0]; // 1 + 2 + 3 + 4 = 10
        let carry = [0, 0];

        let assignment = AccumulateCircuit::<Fr>::create_circuit(inputs, result, carry);

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case 2: Overflow within u64
        let compile_result =
            compile(&AccumulateCircuit::dummy(3), CompileOptions::default()).unwrap();
        let inputs = vec![[u64::MAX - 1, 0], [1, 0], [1, 0]];
        let result = [0, 1]; // MAX - 1 + 1 + 1 = 1 (with carry to next limb)
        let carry = [0, 0];

        let assignment = AccumulateCircuit::<Fr>::create_circuit(inputs, result, carry);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case 3: Near 2^120 boundary
        let compile_result =
            compile(&AccumulateCircuit::dummy(2), CompileOptions::default()).unwrap();
        let max_120_bits = [u64::MAX, (1u64 << 56) - 1]; // 2^120 - 1
        let inputs = vec![max_120_bits, [1, 0]];
        let result = [0, 0]; // Wraps to 0
        let carry = [1, 0];

        let assignment = AccumulateCircuit::<Fr>::create_circuit(inputs, result, carry);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case 4: Multiple carries
        let compile_result =
            compile(&AccumulateCircuit::dummy(3), CompileOptions::default()).unwrap();
        let near_max = [u64::MAX - 2, (1u64 << 56) - 1];
        let inputs = vec![near_max, near_max, near_max];
        let result = [0xfffffffffffffff7, 0xffffffffffffff];
        let carry = [2, 0]; // Three numbers near 2^120 will cause multiple carries

        let assignment = AccumulateCircuit::<Fr>::create_circuit(inputs, result, carry);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case 5: Large number of small values
        let compile_result =
            compile(&AccumulateCircuit::dummy(10), CompileOptions::default()).unwrap();
        let inputs = vec![[1, 0]; 10]; // Ten ones
        let result = [10, 0];
        let carry = [0, 0];

        let assignment = AccumulateCircuit::<Fr>::create_circuit(inputs, result, carry);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    {
        // Test case 6: Single value in upper limb
        let compile_result =
            compile(&AccumulateCircuit::dummy(2), CompileOptions::default()).unwrap();
        let inputs = vec![[0, 1], [0, 2]];
        let result = [0, 3]; // Adding values in upper limb
        let carry = [0, 0];

        let assignment = AccumulateCircuit::<Fr>::create_circuit(inputs, result, carry);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    // Negative test cases
    {
        // Test case 7: Incorrect result
        let compile_result =
            compile(&AccumulateCircuit::dummy(3), CompileOptions::default()).unwrap();
        let inputs = vec![[1, 0], [2, 0], [3, 0]];
        let result = [7, 0]; // Wrong result (should be 6)
        let carry = [0, 0];

        let assignment = AccumulateCircuit::<Fr>::create_circuit(inputs, result, carry);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]);
    }

    {
        // Test case 8: Incorrect handling of upper limb
        let compile_result =
            compile(&AccumulateCircuit::dummy(2), CompileOptions::default()).unwrap();
        let inputs = vec![[1, 1], [2, 1]];
        let result = [3, 1]; // Wrong result (should be [3, 2])
        let carry = [0, 0];

        let assignment = AccumulateCircuit::<Fr>::create_circuit(inputs, result, carry);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]);
    }

    {
        // Test case 9: Minimum case (two values)
        let compile_result =
            compile(&AccumulateCircuit::dummy(2), CompileOptions::default()).unwrap();
        let inputs = vec![[5, 0], [7, 0]];
        let result = [12, 0];
        let carry = [0, 0];

        let assignment = AccumulateCircuit::<Fr>::create_circuit(inputs, result, carry);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
}
