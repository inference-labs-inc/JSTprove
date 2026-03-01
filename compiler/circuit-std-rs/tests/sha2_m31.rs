use circuit_std_rs::sha256::m31::sha256_37bytes;
use expander_compiler::frontend::*;
use extra::*;
use sha2::{Digest, Sha256};

declare_circuit!(SHA25637BYTESCircuit {
    input: [Variable; 37],
    output: [Variable; 32],
});

pub fn check_sha256<C: Config, B: RootAPI<C>>(
    builder: &mut B,
    origin_data: &Vec<Variable>,
) -> Vec<Variable> {
    let output = origin_data[37..].to_vec();
    let result = sha256_37bytes(builder, &origin_data[..37]);
    for i in 0..32 {
        builder.assert_is_equal(result[i], output[i]);
    }
    result
}

impl Define<M31Config> for SHA25637BYTESCircuit<Variable> {
    fn define<Builder: RootAPI<M31Config>>(&self, builder: &mut Builder) {
        for _ in 0..8 {
            let mut data = self.input.to_vec();
            data.append(&mut self.output.to_vec());
            builder.memorized_simple_call(check_sha256, &data);
        }
    }
}

#[test]
fn test_sha256_37bytes() {
    let compile_result =
        compile(&SHA25637BYTESCircuit::default(), CompileOptions::default()).unwrap();
    for i in 0..1 {
        let data = [i; 37];
        let mut hash = Sha256::new();
        hash.update(data);
        let output = hash.finalize();
        let mut assignment = SHA25637BYTESCircuit::default();
        for i in 0..37 {
            assignment.input[i] = M31::from(data[i] as u32);
        }
        for i in 0..32 {
            assignment.output[i] = M31::from(output[i] as u32);
        }
        let witness = compile_result
            .witness_solver
            .solve_witness_with_hints(&assignment, &mut EmptyHintCaller)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
}

#[test]
fn debug_sha256_37bytes() {
    let data = [255; 37];
    let mut hash = Sha256::new();
    hash.update(data);
    let output = hash.finalize();
    let mut assignment = SHA25637BYTESCircuit::default();
    for i in 0..37 {
        assignment.input[i] = M31::from(data[i] as u32);
    }
    for i in 0..32 {
        assignment.output[i] = M31::from(output[i] as u32);
    }
    debug_eval(
        &SHA25637BYTESCircuit::default(),
        &assignment,
        EmptyHintCaller,
    );
}
