mod common;

use circuit_std_rs::matmul::{MatMulCircuit, MatMulParams};
use expander_compiler::frontend::*;

#[test]
fn matmul_test() {
    let matmul_params = [
        MatMulParams {
            m1: 4,
            n1: 3,
            m2: 3,
            n2: 2,
        },
        MatMulParams {
            m1: 6,
            n1: 6,
            m2: 6,
            n2: 6,
        },
        MatMulParams {
            m1: 10,
            n1: 5,
            m2: 5,
            n2: 1,
        },
        MatMulParams {
            m1: 1,
            n1: 1,
            m2: 1,
            n2: 1,
        },
        MatMulParams {
            m1: 50,
            n1: 35,
            m2: 35,
            n2: 65,
        },
    ];

    for params in matmul_params.iter() {
        common::circuit_test_helper::<BN254Config, MatMulCircuit>(params);
        common::circuit_test_helper::<M31Config, MatMulCircuit>(params);
    }

    //let mut rng = rand::rngs::StdRng::seed_from_u64(1235);
    //debug_eval(&MatMulCircuit::default(), &MatMulCircuit::new_assignment(&matmul_params, rng), EmptyHintCaller);
}
