use std::path::Path;
use std::collections::HashMap;

use anyhow::Result;
use remainder::mle::evals::MultilinearExtension;
use shared_types::config::global_config::global_prover_circuit_description_hash_type;
use shared_types::config::{GKRCircuitProverConfig, ProofConfig};
use shared_types::transcript::poseidon_sponge::PoseidonSponge;
use shared_types::transcript::{Transcript, TranscriptWriter};
use shared_types::{perform_function_under_prover_config, Fr};

use crate::runner::circuit_builder;
use crate::padding::num_vars_for;
use crate::util::i64_to_fr;

use super::serialization;

pub fn run(model_path: &Path, witness_path: &Path, output_path: &Path, compress: bool) -> Result<()> {
    tracing::info!("loading model from {}", model_path.display());
    let model = super::compile::load_model(model_path)?;

    tracing::info!("loading witness from {}", witness_path.display());
    let witness = super::witness::load_witness(witness_path)?;

    let proof = generate_proof(&model, &witness)?;

    let size = serialization::serialize_to_file(&proof, output_path, compress)?;
    tracing::info!("proof written to {} ({} bytes)", output_path.display(), size);
    Ok(())
}

pub fn generate_proof(
    model: &crate::onnx::quantizer::QuantizedModel,
    witness: &HashMap<String, Vec<i64>>,
) -> Result<SerializableProof> {
    let input_name = model.graph.input_names.first()
        .ok_or_else(|| anyhow::anyhow!("model has no input names defined"))?
        .clone();
    let input_size = witness.get(&input_name)
        .map(|v| v.len())
        .ok_or_else(|| anyhow::anyhow!("witness missing input shred '{}'", input_name))?;

    tracing::info!("building circuit");
    let build_result = circuit_builder::build_circuit(model, input_size)?;
    let mut circuit = build_result.circuit;

    tracing::info!("validating witness completeness against manifest");
    for (name, entry) in &build_result.manifest {
        match witness.get(name) {
            None => anyhow::bail!("witness missing shred '{}' ({:?})", name, entry.visibility),
            Some(values) => {
                let expected_nv = entry.num_vars;
                let actual_nv = num_vars_for(values.len());
                if actual_nv != expected_nv {
                    anyhow::bail!(
                        "witness shred '{}' has {} vars (len {}), circuit expects {} vars",
                        name, actual_nv, values.len(), expected_nv
                    );
                }
            }
        }
    }
    tracing::info!("witness validation passed ({} shreds verified)", build_result.manifest.len());

    tracing::info!("setting {} witness shreds on circuit", witness.len());
    for (name, values) in witness {
        let mle = MultilinearExtension::new(
            values.iter().map(|&v| i64_to_fr(v)).collect(),
        );
        circuit.set_input(name, mle);
    }

    tracing::info!("generating proof");
    let provable = circuit.gen_provable_circuit()?;

    let (proof_config, proof_transcript) = prove_and_get_transcript(&provable)?;

    Ok(SerializableProof {
        proof_config,
        transcript: proof_transcript,
    })
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SerializableProof {
    pub proof_config: ProofConfig,
    pub transcript: Transcript<Fr>,
}

pub fn load_proof(path: &Path) -> Result<SerializableProof> {
    serialization::deserialize_from_file(path)
}

fn prove_and_get_transcript(
    provable: &remainder::provable_circuit::ProvableCircuit<Fr>,
) -> Result<(ProofConfig, Transcript<Fr>)> {
    let runtime_optimized_config = GKRCircuitProverConfig::runtime_optimized_default();

    let result: Result<(ProofConfig, Transcript<Fr>)> = perform_function_under_prover_config!(
        prove_internal,
        &runtime_optimized_config,
        provable
    );

    result
}

fn prove_internal(
    provable: &remainder::provable_circuit::ProvableCircuit<Fr>,
) -> Result<(ProofConfig, Transcript<Fr>)> {
    let mut transcript_writer =
        TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new("GKR Prover Transcript");

    let proof_config = provable
        .prove(
            global_prover_circuit_description_hash_type(),
            &mut transcript_writer,
        )
        .map_err(|e| anyhow::anyhow!("proof generation failed: {}", e))?;

    let transcript = transcript_writer.get_transcript();
    Ok((proof_config, transcript))
}
