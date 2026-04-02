// Copyright 2025 Irreducible Inc.
use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use binius_core::constraint_system::{ConstraintSystem, Proof, ValueVec, ValuesData};
use binius_examples::setup_sha256;
use binius_utils::serialization::{DeserializeBytes, SerializeBytes};
use binius_verifier::{
    config::{ChallengerWithName, StdChallenger},
    transcript::ProverTranscript,
};
use clap::Parser;

/// Prover CLI: generate a proof from a serialized constraint system and witnesses.
#[derive(Debug, Parser)]
#[command(
    name = "prover",
    about = "Generate and save a proof from CS and witnesses"
)]
struct Args {
    /// Path to the constraint system binary
    #[arg(long = "cs-path")]
    cs_path: PathBuf,

    /// Path to the public values (ValuesData) binary
    #[arg(long = "pub-witness-path")]
    pub_witness_path: PathBuf,

    /// Path to the non-public values (ValuesData) binary
    #[arg(long = "non-pub-data-path")]
    non_pub_data_path: PathBuf,

    /// Path to write the proof binary
    #[arg(long = "proof-path")]
    proof_path: PathBuf,

    /// Log of the inverse rate for the proof system
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

    // Read and deserialize non-public values
    let non_pub_bytes = fs::read(&args.non_pub_data_path).with_context(|| {
        format!(
            "Failed to read non-public values from {}",
            args.non_pub_data_path.display()
        )
    })?;
    let non_public = ValuesData::deserialize(&mut non_pub_bytes.as_slice())
        .context("Failed to deserialize non-public ValuesData")?;

    // Reconstruct the full ValueVec
    // Take ownership of the underlying vectors without extra copies
    let public: Vec<_> = public.into();
    let non_public: Vec<_> = non_public.into();
    let witness = ValueVec::new_from_data(cs.value_vec_layout.clone(), public, non_public)
        .context("Failed to reconstruct ValueVec from provided values")?;

    // Setup prover (verifier is not used here)
    let (_verifier, prover) = setup_sha256(cs, args.log_inv_rate as usize, None)?;

    // Prove
    let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
    prover
        .prove(witness, &mut prover_transcript)
        .context("Proving failed")?;
    let transcript = prover_transcript.finalize();

    // Wrap into serializable Proof with a stable challenger type identifier.
    // NOTE: Avoid std::any::type_name for cross-platform stability; use a constant instead.
    let proof = Proof::owned(transcript, StdChallenger::NAME.to_string());

    // Serialize and save the proof
    if let Some(parent) = args.proof_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create parent directory {}", parent.display()))?;
    }
    let mut buf = Vec::new();
    proof
        .serialize(&mut buf)
        .context("Failed to serialize proof")?;
    fs::write(&args.proof_path, &buf)
        .with_context(|| format!("Failed to write proof to {}", args.proof_path.display()))?;

    tracing::info!(
        "Saved proof to {} ({} bytes)",
        args.proof_path.display(),
        buf.len()
    );
    Ok(())
}
