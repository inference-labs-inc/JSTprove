use std::mem::transmute;

use expander_compiler::frontend::*;
use halo2curves::bn256::Fr;

use crate::{u120, BN_TWO_TO_120, MASK120};

declare_circuit!(AddCircuit {
    x: Variable,
    y: Variable,
    carry_in: Variable,
    carry_out: Variable,
    result: Variable,
});

impl Define<BN254Config> for AddCircuit<Variable> {
    fn define<Builder: RootAPI<BN254Config>>(&self, builder: &mut Builder) {
        let two_to_120 = builder.constant(BN_TWO_TO_120);

        let (result, carry_out) =
            u120::add_u120(&self.x, &self.y, &self.carry_in, &two_to_120, builder);

        builder.assert_is_equal(result, self.result);
        builder.assert_is_equal(carry_out, self.carry_out);
    }
}
impl AddCircuit<Fr> {
    // Helper function to create circuit instance with given inputs
    fn create_circuit(
        x: [u64; 2],
        y: [u64; 2],
        carry_in: u64,
        carry_out: u64,
        result: [u64; 2],
    ) -> AddCircuit<Fr> {
        Self {
            x: Fr::from_raw([x[0], x[1], 0, 0]),
            y: Fr::from_raw([y[0], y[1], 0, 0]),
            carry_in: Fr::from_raw([carry_in, 0, 0, 0]),
            carry_out: Fr::from(carry_out),
            result: Fr::from_raw([result[0], result[1], 0, 0]),
        }
    }
}

#[test]
fn test_rsa_circuit_120_addition() {
    let compile_result = compile(&AddCircuit::default(), CompileOptions::default()).unwrap();

    {
        // Test case: Simple addition without carries
        let x = [50, 0];
        let y = [30, 0];
        let res = [80, 0];
        let carry_ins = 0;
        let carry_outs = 0;

        let assignment = AddCircuit::<Fr>::create_circuit(x, y, carry_ins, carry_outs, res);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: overflow 64 bits
        let x = [u64::MAX, 0];
        let y = [u64::MAX, 0];
        let res = [u64::MAX - 1, 1];
        let carry_ins = 0;
        let carry_outs = 0;

        let assignment = AddCircuit::<Fr>::create_circuit(x, y, carry_ins, carry_outs, res);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: overflow 120 bits
        let x = unsafe { transmute(MASK120) };
        let y = [2, 0];
        let res = [1, 0];
        let carry_ins = 0;
        let carry_outs = 1;

        let assignment = AddCircuit::<Fr>::create_circuit(x, y, carry_ins, carry_outs, res);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: max
        let x = unsafe { transmute(MASK120) }; // -1
        let y = unsafe { transmute(MASK120) }; // -1
        let res = unsafe { transmute(MASK120 - 1) }; // -2
        let carry_ins = 0;
        let carry_outs = 1;

        let assignment = AddCircuit::<Fr>::create_circuit(x, y, carry_ins, carry_outs, res);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: negative case
        let x = [50, 0];
        let y = [40, 0];
        let res = [80, 0];
        let carry_ins = 0;
        let carry_outs = 0;

        let assignment = AddCircuit::<Fr>::create_circuit(x, y, carry_ins, carry_outs, res);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]);
    }
}
