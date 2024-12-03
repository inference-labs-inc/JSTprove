mod common;

use circuit_std_rs::{LogUpCircuit, LogUpParams};
use expander_compiler::frontend::*;

#[test]
fn logup_test() {
    let logup_params = LogUpParams {
        key_len: 7,
        value_len: 7,
        n_table_rows: 123,
        n_queries: 456,
    };

    common::circuit_test_helper::<BN254Config, LogUpCircuit>(&logup_params);
    common::circuit_test_helper::<M31Config, LogUpCircuit>(&logup_params);
    common::circuit_test_helper::<GF2Config, LogUpCircuit>(&logup_params);
}
