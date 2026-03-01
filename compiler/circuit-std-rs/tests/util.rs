use expander_compiler::frontend::*;
use halo2curves::bn256::Fr;
use halo2curves::ff::Field;

use circuit_std_rs::byte_decomposition;

declare_circuit!(ByteDecompCircuit {
    input: Variable,
    bytes: [Variable; 256 / 8],
});

impl Define<BN254Config> for ByteDecompCircuit<Variable> {
    fn define<Builder: RootAPI<BN254Config>>(&self, builder: &mut Builder) {
        let two_to_eight = Fr::from(1u64 << 8);
        let constant_scalars = (0..32)
            .map(|i| {
                let scalar = two_to_eight.pow([i as u64, 0, 0, 0]);
                builder.constant(scalar)
            })
            .collect::<Vec<_>>();

        let decomposed = byte_decomposition(&self.input, &constant_scalars, builder);
        assert_eq!(decomposed.len(), self.bytes.len());

        for (actual, expected) in decomposed.iter().zip(self.bytes.iter()) {
            builder.assert_is_equal(actual, expected);
        }
    }
}

impl ByteDecompCircuit<Fr> {
    fn create_circuit(input: [u64; 4], bytes: Vec<[u64; 4]>) -> ByteDecompCircuit<Fr> {
        Self {
            input: Fr::from_raw(input),
            bytes: bytes
                .into_iter()
                .map(Fr::from_raw)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }
}

#[test]
fn test_byte_decomposition() {
    let compile_result = compile(&ByteDecompCircuit::default(), CompileOptions::default()).unwrap();

    {
        // Test case: Zero
        let input = [0, 0, 0, 0];
        let bytes = vec![[0, 0, 0, 0]; 32];

        let assignment = ByteDecompCircuit::<Fr>::create_circuit(input, bytes);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: Simple value (255)
        let input = [255, 0, 0, 0];
        let mut bytes = vec![[0, 0, 0, 0]; 32];
        bytes[0] = [255, 0, 0, 0];

        let assignment = ByteDecompCircuit::<Fr>::create_circuit(input, bytes);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: Two-byte value (0x1234)
        let input = [0x1234, 0, 0, 0];
        let mut bytes = vec![[0, 0, 0, 0]; 32];
        bytes[0] = [0x34, 0, 0, 0];
        bytes[1] = [0x12, 0, 0, 0];

        let assignment = ByteDecompCircuit::<Fr>::create_circuit(input, bytes);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: Large value across multiple bytes
        let input = [0x1234567890ABCDEF, 0, 0, 0];
        let mut bytes = vec![[0, 0, 0, 0]; 32];
        bytes[0] = [0xEF, 0, 0, 0];
        bytes[1] = [0xCD, 0, 0, 0];
        bytes[2] = [0xAB, 0, 0, 0];
        bytes[3] = [0x90, 0, 0, 0];
        bytes[4] = [0x78, 0, 0, 0];
        bytes[5] = [0x56, 0, 0, 0];
        bytes[6] = [0x34, 0, 0, 0];
        bytes[7] = [0x12, 0, 0, 0];

        let assignment = ByteDecompCircuit::<Fr>::create_circuit(input, bytes);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: Value using multiple u64 limbs
        let input = [0xFFFFFFFFFFFFFFFF, 0x123456789, 0, 0];
        let mut bytes = vec![[0, 0, 0, 0]; 32];
        for i in 0..8 {
            bytes[i] = [0xFF, 0, 0, 0];
        }
        bytes[8] = [0x89, 0, 0, 0];
        bytes[9] = [0x67, 0, 0, 0];
        bytes[10] = [0x45, 0, 0, 0];
        bytes[11] = [0x23, 0, 0, 0];
        bytes[12] = [0x01, 0, 0, 0];

        let assignment = ByteDecompCircuit::<Fr>::create_circuit(input, bytes);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: All limbs non-zero
        let input = [
            0xFFFFFFFFFFFFFFFF,
            0xAAAAAAAAAAAAAAAA,
            0x5555555555555555,
            0x1111111111111111,
        ];
        let mut bytes = vec![[0, 0, 0, 0]; 32];

        // Fill expected bytes based on the input limbs
        // First limb (0xFFFFFFFFFFFFFFFF)
        for i in 0..8 {
            bytes[i] = [0xFF, 0, 0, 0];
        }
        // Second limb (0xAAAAAAAAAAAAAAAA)
        for i in 8..16 {
            bytes[i] = [0xAA, 0, 0, 0];
        }
        // Third limb (0x5555555555555555)
        for i in 16..24 {
            bytes[i] = [0x55, 0, 0, 0];
        }
        // Fourth limb (0x1111111111111111)
        for i in 24..32 {
            bytes[i] = [0x11, 0, 0, 0];
        }

        let assignment = ByteDecompCircuit::<Fr>::create_circuit(input, bytes);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
    {
        // Test case: Negative case (incorrect decomposition)
        let input = [0x1234, 0, 0, 0];
        let mut bytes = vec![[0, 0, 0, 0]; 32];
        bytes[0] = [0x35, 0, 0, 0]; // Incorrect value (should be 0x34)
        bytes[1] = [0x12, 0, 0, 0];

        let assignment = ByteDecompCircuit::<Fr>::create_circuit(input, bytes);
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]);
    }
}
