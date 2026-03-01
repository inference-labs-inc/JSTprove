#![allow(clippy::pedantic)]
use std::str::FromStr;

use clap::Parser;
use expander_compiler::zkcuda::proving_system::{
    expander::config::{
        ZKCudaBN254Hyrax, ZKCudaBN254HyraxBatchPCS, ZKCudaBN254KZG, ZKCudaBN254KZGBatchPCS,
    },
    expander_parallelized::server_ctrl::{serve, ExpanderExecArgs},
    ExpanderNoOverSubscribe,
};
use gkr_engine::PolynomialCommitmentType;

#[tokio::main]
pub async fn main() {
    let expander_exec_args = ExpanderExecArgs::parse();
    assert_eq!(
        expander_exec_args.fiat_shamir_hash, "SHA256",
        "Only SHA256 is supported for now"
    );

    let pcs_type = PolynomialCommitmentType::from_str(&expander_exec_args.poly_commit).unwrap();

    match (expander_exec_args.field_type.as_str(), pcs_type) {
        ("BN254", PolynomialCommitmentType::Hyrax) => {
            if expander_exec_args.batch_pcs {
                serve::<_, _, ExpanderNoOverSubscribe<ZKCudaBN254HyraxBatchPCS>>(
                    expander_exec_args.port_number,
                )
                .await;
            } else {
                serve::<_, _, ExpanderNoOverSubscribe<ZKCudaBN254Hyrax>>(
                    expander_exec_args.port_number,
                )
                .await;
            }
        }
        ("BN254", PolynomialCommitmentType::KZG) => {
            if expander_exec_args.batch_pcs {
                serve::<_, _, ExpanderNoOverSubscribe<ZKCudaBN254KZGBatchPCS>>(
                    expander_exec_args.port_number,
                )
                .await;
            } else {
                serve::<_, _, ExpanderNoOverSubscribe<ZKCudaBN254KZG>>(
                    expander_exec_args.port_number,
                )
                .await;
            }
        }
        (field_type, pcs_type) => {
            panic!("Combination of {field_type:?} and {pcs_type:?} not supported for no oversubscribe expander proving system.");
        }
    }
}
