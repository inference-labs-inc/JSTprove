use circuit_std_rs::sha256::{gf2::SHA256GF2, gf2_utils::u32_to_bit};
use expander_compiler::frontend::*;
#[allow(unused_imports)]
use extra::debug_eval;
use rand::RngCore;
use sha2::{Digest, Sha256};

mod sha256_debug_utils;
use sha256_debug_utils::{compress, H256_256 as SHA256_INIT_STATE};

const INPUT_LEN: usize = 1024; // input size in bits, must be a multiple of 8
const OUTPUT_LEN: usize = 256; // FIXED 256

declare_circuit!(SHA256CircuitCompressionOnly {
    input: [Variable; 512],
    output: [Variable; 256],
});

impl Define<GF2Config> for SHA256CircuitCompressionOnly<Variable> {
    fn define<Builder: RootAPI<GF2Config>>(&self, api: &mut Builder) {
        let hasher = SHA256GF2::new();
        let mut state = SHA256_INIT_STATE
            .iter()
            .map(|x| u32_to_bit(api, *x))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        hasher.sha256_compress(api, &mut state, &self.input);
        let output = state.iter().flatten().cloned().collect::<Vec<_>>();
        for i in 0..256 {
            api.assert_is_equal(output[i], self.output[i]);
        }
    }
}

#[test]
fn test_sha256_compression_gf2() {
    // let compile_result = compile_generic(
    //     &SHA256CircuitCompressionOnly::default(),
    //     CompileOptions::default(),
    // )
    // .unwrap();

    let compile_result = compile_cross_layer(
        &SHA256CircuitCompressionOnly::default(),
        CompileOptions::default(),
    )
    .unwrap();

    let mut rng = rand::thread_rng();
    let n_tests = 5;
    for _ in 0..n_tests {
        let data = [rng.next_u32() as u8; 512 / 8];
        let mut state = SHA256_INIT_STATE;
        compress(&mut state, &[data.try_into().unwrap()]);
        let output = state
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect::<Vec<_>>();

        let mut assignment = SHA256CircuitCompressionOnly::default();

        for i in 0..64 {
            for j in 0..8 {
                assignment.input[i * 8 + j] = ((data[i] >> (7 - j)) as u32 & 1).into();
            }
        }
        for i in 0..32 {
            for j in 0..8 {
                assignment.output[i * 8 + j] = ((output[i] >> (7 - j)) as u32 & 1).into();
            }
        }

        // debug_eval::<GF2Config, _, _, _>(
        //     &SHA256CircuitCompressionOnly::default(),
        //     &assignment,
        //     EmptyHintCaller::new(),
        // );

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
}

declare_circuit!(SHA256Circuit {
    input: [Variable; INPUT_LEN],
    output: [Variable; OUTPUT_LEN],
});

impl Define<GF2Config> for SHA256Circuit<Variable> {
    fn define<Builder: RootAPI<GF2Config>>(&self, api: &mut Builder) {
        let mut hasher = SHA256GF2::new();
        hasher.update(&self.input);
        let output = hasher.finalize(api);
        (0..OUTPUT_LEN).for_each(|i| api.assert_is_equal(output[i], self.output[i]));
    }
}

#[test]
fn test_sha256_gf2() {
    assert!(INPUT_LEN % 8 == 0);
    // let compile_result =
    //     compile_generic(&SHA256Circuit::default(), CompileOptions::default()).unwrap();

    let compile_result =
        compile_cross_layer(&SHA256Circuit::default(), CompileOptions::default()).unwrap();

    let n_tests = 5;
    let mut rng = rand::thread_rng();
    for _ in 0..n_tests {
        let data = [rng.next_u32() as u8; INPUT_LEN / 8];
        let mut hash = Sha256::new();
        hash.update(data);
        let output = hash.finalize();
        let mut assignment = SHA256Circuit::default();
        for i in 0..INPUT_LEN / 8 {
            for j in 0..8 {
                assignment.input[i * 8 + j] = (((data[i] >> (7 - j)) & 1) as u32).into();
            }
        }
        for i in 0..OUTPUT_LEN / 8 {
            for j in 0..8 {
                assignment.output[i * 8 + j] = (((output[i] >> (7 - j) as u32) & 1) as u32).into();
            }
        }
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
}
