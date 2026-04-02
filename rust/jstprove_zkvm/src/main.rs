#[allow(dead_code)]
mod canonical;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::info;

use canonical::CanonicalModel;

#[derive(Parser)]
#[command(
    name = "jstprove-zkvm",
    about = "Prove circuit compilation correctness via Jolt zkVM"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Prove {
        #[arg(long)]
        model: PathBuf,
        #[arg(
            long,
            default_value = "/tmp/jolt-guest-targets"
        )]
        target_dir: PathBuf,
    },
    Hash {
        #[arg(long)]
        model: PathBuf,
    },
}

fn load_canonical(model_path: &PathBuf) -> Result<(CanonicalModel, Vec<u8>)> {
    let raw = std::fs::read(model_path).context("reading model file")?;
    let msgpack = decompress_model_bytes(&raw)?;
    let model = CanonicalModel::from_quantized_model_bytes(&msgpack)?;
    let encoded = model.encode();

    info!(
        "Canonical model: {} layers, {} bytes ({:.1} KB)",
        model.layers.len(),
        encoded.len(),
        encoded.len() as f64 / 1024.0,
    );

    Ok((model, encoded))
}

fn decompress_model_bytes(raw: &[u8]) -> Result<Vec<u8>> {
    if raw.len() >= 20 && &raw[..4] == b"JST\x01" {
        let flags =
            u32::from_le_bytes(raw[4..8].try_into().unwrap());
        let payload_len =
            u64::from_le_bytes(raw[8..16].try_into().unwrap()) as usize;
        let compressed = flags & 1 != 0;
        let payload = &raw[20..20 + payload_len];
        if compressed {
            Ok(zstd::decode_all(payload)?)
        } else {
            Ok(payload.to_vec())
        }
    } else if raw.len() >= 4
        && raw[..4] == [0x28, 0xB5, 0x2F, 0xFD]
    {
        Ok(zstd::decode_all(raw.as_ref())?)
    } else {
        Ok(raw.to_vec())
    }
}

fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

fn cmd_prove(
    model_path: &PathBuf,
    target_dir: &PathBuf,
) -> Result<()> {
    let (_, model_bytes) = load_canonical(model_path)?;

    let input_hash = sha256_bytes(&model_bytes);
    info!("Input hash (canonical): {}", hex::encode(input_hash));

    let target_str = target_dir.to_str().unwrap();

    info!("Compiling guest program...");
    let now = Instant::now();
    let mut program = guest::compile_verify_compilation(target_str);
    info!("Guest compilation: {:.2}s", now.elapsed().as_secs_f64());

    info!("Preprocessing...");
    let now = Instant::now();
    let shared = guest::preprocess_shared_verify_compilation(
        &mut program,
    )
    .context("shared preprocessing")?;
    let prover_pp =
        guest::preprocess_prover_verify_compilation(shared.clone());
    let verifier_setup = prover_pp.generators.to_verifier_setup();
    let verifier_pp =
        guest::preprocess_verifier_verify_compilation(
            shared,
            verifier_setup,
            None,
        );
    info!(
        "Preprocessing: {:.2}s",
        now.elapsed().as_secs_f64()
    );

    let prove_fn = guest::build_prover_verify_compilation(
        program,
        prover_pp,
    );
    let verify_fn =
        guest::build_verifier_verify_compilation(verifier_pp);

    info!("Proving compilation...");
    let now = Instant::now();
    let (output_hash, proof, program_io) = prove_fn(&model_bytes);
    let prove_secs = now.elapsed().as_secs_f64();
    info!("Prove time: {:.2}s", prove_secs);
    info!("Trace length: {} cycles", proof.trace_length);
    info!(
        "Compilation output hash: {}",
        hex::encode(output_hash)
    );

    info!("Verifying proof...");
    let now = Instant::now();
    let valid = verify_fn(
        &model_bytes,
        output_hash,
        program_io.panic,
        proof,
    );
    let verify_secs = now.elapsed().as_secs_f64();
    info!("Verify time: {:.2}s", verify_secs);

    if valid {
        info!("VALID: compilation proof verified");
        let result = serde_json::json!({
            "status": "valid",
            "input_hash": hex::encode(input_hash),
            "compilation_hash": hex::encode(output_hash),
            "prove_time_s": prove_secs,
            "verify_time_s": verify_secs,
        });
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        anyhow::bail!("compilation proof verification failed");
    }

    Ok(())
}

fn cmd_hash(model_path: &PathBuf) -> Result<()> {
    let (_, model_bytes) = load_canonical(model_path)?;
    let hash = sha256_bytes(&model_bytes);
    println!("{}", hex::encode(hash));
    Ok(())
}

pub fn main() {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    let result = match &cli.command {
        Command::Prove { model, target_dir } => {
            cmd_prove(model, target_dir)
        }
        Command::Hash { model } => cmd_hash(model),
    };

    if let Err(e) = result {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}
