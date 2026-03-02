use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct ExpanderExecArgs {
    /// M31, GF2, BN254, Goldilocks, BabyBear
    #[arg(short, long, default_value = "M31")]
    pub field_type: String,

    /// Fiat-Shamir Hash: SHA256, or Poseidon, or MiMC5
    #[arg(short, long, default_value = "SHA256")]
    pub fiat_shamir_hash: String,

    /// Polynomial Commitment Scheme: Raw, or Orion
    #[arg(short, long, default_value = "Raw")]
    pub poly_commit: String,
}
