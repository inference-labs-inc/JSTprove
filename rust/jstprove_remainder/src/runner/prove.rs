use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use remainder::mle::evals::MultilinearExtension;
use shared_types::config::global_config::global_prover_circuit_description_hash_type;
use shared_types::config::{GKRCircuitProverConfig, ProofConfig};
use shared_types::transcript::poseidon_sponge::PoseidonSponge;
use shared_types::transcript::{Transcript, TranscriptWriter};
use shared_types::{perform_function_under_prover_config, Fr};

use crate::runner::circuit_builder;

pub fn run(model_path: &Path, witness_path: &Path, output_path: &Path) -> Result<()> {
    tracing::info!("loading model from {}", model_path.display());
    let model = super::compile::load_model(model_path)?;

    tracing::info!("loading witness from {}", witness_path.display());
    let witness = super::witness::load_witness(witness_path)?;

    tracing::info!("building circuit");
    let input_name = model.graph.input_names.first()
        .cloned()
        .unwrap_or_else(|| "input".to_string());
    let input_size = witness.get(&input_name)
        .map(|v| v.len())
        .ok_or_else(|| anyhow::anyhow!("witness missing input shred '{}'", input_name))?;

    let build_result = circuit_builder::build_circuit(&model, input_size)?;
    let mut circuit = build_result.circuit;

    tracing::info!("setting {} witness shreds on circuit", witness.len());
    for (name, values) in &witness {
        let mle = MultilinearExtension::new(
            values.iter().map(|&v| i64_to_fr(v)).collect(),
        );
        circuit.set_input(name, mle);
    }

    tracing::info!("generating proof");
    let provable = circuit.gen_provable_circuit()?;

    let (proof_config, proof_transcript) = prove_and_get_transcript(&provable)?;

    let serializable = SerializableProof {
        proof_config,
        transcript: proof_transcript,
    };

    let serialized = bincode::serialize(&serializable)?;
    let compressed = zstd::encode_all(serialized.as_slice(), 3)?;
    std::fs::write(output_path, &compressed)?;

    tracing::info!(
        "proof written to {} ({} bytes compressed)",
        output_path.display(),
        compressed.len()
    );
    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SerializableProof {
    pub proof_config: ProofConfig,
    pub transcript: Transcript<Fr>,
}

pub fn load_proof(path: &Path) -> Result<SerializableProof> {
    let compressed = std::fs::read(path)?;
    let decompressed = zstd::decode_all(compressed.as_slice())?;
    let proof: SerializableProof = bincode::deserialize(&decompressed)?;
    Ok(proof)
}

fn prove_and_get_transcript(
    provable: &remainder::provable_circuit::ProvableCircuit<Fr>,
) -> Result<(ProofConfig, Transcript<Fr>)> {
    let runtime_optimized_config = GKRCircuitProverConfig::runtime_optimized_default();

    let result: (ProofConfig, Transcript<Fr>) = perform_function_under_prover_config!(
        prove_internal,
        &runtime_optimized_config,
        provable
    );

    Ok(result)
}

fn prove_internal(
    provable: &remainder::provable_circuit::ProvableCircuit<Fr>,
) -> (ProofConfig, Transcript<Fr>) {
    let mut transcript_writer =
        TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new("GKR Prover Transcript");

    let proof_config = provable
        .prove(
            global_prover_circuit_description_hash_type(),
            &mut transcript_writer,
        )
        .expect("proof generation failed");

    let transcript = transcript_writer.get_transcript();
    (proof_config, transcript)
}

fn i64_to_fr(val: i64) -> Fr {
    use shared_types::Field;
    if val >= 0 {
        Fr::from(val as u64)
    } else {
        Fr::from(val.unsigned_abs()).neg()
    }
}
