// Copyright 2025 Irreducible Inc.
use std::{fs, path::PathBuf};

use anyhow::{Context, Result, bail};
use binius_core::constraint_system::{ConstraintSystem, Proof, ValuesData};
use binius_examples::StdVerifier;
use binius_utils::serialization::DeserializeBytes;
use binius_verifier::{
    Verifier,
    config::{ChallengerWithName, StdChallenger},
    hash::StdCompression,
    transcript::VerifierTranscript,
};
use clap::Parser;

/// Verifier CLI: load CS, public witness and proof, then verify.
#[derive(Debug, Parser)]
#[command(
    name = "verifier",
    about = "Verify a proof from a constraint system, public witness, and proof binary"
)]
struct Args {
    /// Path to the constraint system binary
    #[arg(long = "cs-path")]
    cs_path: PathBuf,

    /// Path to the public values (ValuesData) binary
    #[arg(long = "pub-witness-path")]
    pub_witness_path: PathBuf,

    /// Path to the proof binary
    #[arg(long = "proof-path")]
    proof_path: PathBuf,

    /// Log of the inverse rate for the proof system (must match what was used for proving)
    #[arg(short = 'l', long = "log-inv-rate", default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    log_inv_rate: u32,
}

fn main() -> Result<()> {
    let _tracing_guard = tracing_profile::init_tracing().ok();
    let args = Args::parse();

    // Read and deserialize constraint system
    let cs_bytes = fs::read(&args.cs_path).with_context(|| {
        format!(
            "Failed to read constraint system from {}",
            args.cs_path.display()
        )
    })?;
    let cs = ConstraintSystem::deserialize(&mut cs_bytes.as_slice())
        .context("Failed to deserialize ConstraintSystem")?;

    // Read and deserialize public values
    let pub_bytes = fs::read(&args.pub_witness_path).with_context(|| {
        format!(
            "Failed to read public values from {}",
            args.pub_witness_path.display()
        )
    })?;
    let public = ValuesData::deserialize(&mut pub_bytes.as_slice())
        .context("Failed to deserialize public ValuesData")?;

    // Read and deserialize proof
    let proof_bytes = fs::read(&args.proof_path)
        .with_context(|| format!("Failed to read proof from {}", args.proof_path.display()))?;
    let proof =
        Proof::deserialize(&mut proof_bytes.as_slice()).context("Failed to deserialize Proof")?;

    // Validate challenger type matches our verifier configuration
    let expected_challenger = StdChallenger::NAME;
    if proof.challenger_type() != expected_challenger {
        bail!(
            "Challenger type mismatch: expected '{}', found '{}'",
            expected_challenger,
            proof.challenger_type()
        );
    }

    // Set up the verifier
    let verifier: StdVerifier =
        Verifier::setup(cs, args.log_inv_rate as usize, StdCompression::default())
            .context("Failed to setup verifier")?;

    // Create a verifier transcript from the serialized proof data
    let (data, _) = proof.into_owned();
    let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), data);

    // Verify
    verifier
        .verify(public.as_slice(), &mut verifier_transcript)
        .context("Verification failed")?;
    verifier_transcript
        .finalize()
        .context("Transcript not fully consumed after verification")?;

    tracing::info!("Proof verified successfully");
    Ok(())
}
