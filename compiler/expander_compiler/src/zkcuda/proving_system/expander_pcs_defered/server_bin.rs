#![allow(clippy::pedantic)]
use std::str::FromStr;

use clap::Parser;
use expander_compiler::{
    frontend::BN254Config,
    zkcuda::proving_system::{
        expander_parallelized::server_ctrl::{serve, ExpanderExecArgs},
        expander_pcs_defered::BN254ConfigSha2UniKZG,
        ExpanderPCSDefered,
    },
};
use gkr::BN254ConfigSha2Hyrax;
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
            serve::<BN254ConfigSha2Hyrax, BN254Config, ExpanderPCSDefered<_>>(
                expander_exec_args.port_number,
            )
            .await;
        }
        ("BN254", PolynomialCommitmentType::KZG) => {
            serve::<BN254ConfigSha2UniKZG, BN254Config, ExpanderPCSDefered<_>>(
                expander_exec_args.port_number,
            )
            .await;
        }
        (field_type, pcs_type) => {
            panic!("Combination of {field_type:?} and {pcs_type:?} not supported for pcs defered expander proving system.");
        }
    }
}
