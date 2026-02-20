use std::path::Path;

use anyhow::Result;
use remainder::mle::evals::MultilinearExtension;
use shared_types::config::global_config::global_verifier_circuit_description_hash_type;
use shared_types::config::GKRCircuitVerifierConfig;
use shared_types::transcript::poseidon_sponge::PoseidonSponge;
use shared_types::transcript::TranscriptReader;
use shared_types::{perform_function_under_verifier_config, Fr};

use crate::runner::circuit_builder::{self, Visibility};

pub fn run(model_path: &Path, proof_path: &Path, input_path: &Path) -> Result<()> {
    tracing::info!("loading model from {}", model_path.display());
    let model = super::compile::load_model(model_path)?;

    tracing::info!("loading proof from {}", proof_path.display());
    let proof = super::prove::load_proof(proof_path)?;

    tracing::info!("loading input from {}", input_path.display());
    let input_json: serde_json::Value = serde_json::from_reader(std::fs::File::open(input_path)?)?;

    let raw_input: Vec<f64> = input_json.get("input")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("input JSON must have an \"input\" array field"))?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0))
        .collect();

    let alpha = model.scale_config.alpha;
    let quantized_input: Vec<i64> = raw_input.iter()
        .map(|&v| (v * alpha as f64).round() as i64)
        .collect();

    let witness = super::witness::compute_witness(&model, &quantized_input)?;

    let input_name = model.graph.input_names.first()
        .cloned()
        .unwrap_or_else(|| "input".to_string());
    let input_size = witness.get(&input_name)
        .map(|v| v.len())
        .ok_or_else(|| anyhow::anyhow!("witness missing input shred '{}'", input_name))?;

    tracing::info!("building verifier circuit");
    let build_result = circuit_builder::build_circuit(&model, input_size)?;
    let mut circuit = build_result.circuit;

    for (name, entry) in &build_result.manifest {
        if entry.visibility == Visibility::Public {
            if let Some(values) = witness.get(name) {
                let mle = MultilinearExtension::new(
                    values.iter().map(|&v| i64_to_fr(v)).collect(),
                );
                circuit.set_input(name, mle);
            }
        }
    }

    tracing::info!("verifying proof");
    let verifiable = circuit.gen_verifiable_circuit()?;

    let mut transcript_reader =
        TranscriptReader::<Fr, PoseidonSponge<Fr>>::new(proof.transcript);

    let verifier_config =
        GKRCircuitVerifierConfig::new_from_proof_config(&proof.proof_config, false);

    let result: std::result::Result<(), _> = perform_function_under_verifier_config!(
        verify_internal,
        &verifier_config,
        &verifiable,
        &mut transcript_reader,
        &proof.proof_config
    );

    match result {
        Ok(()) => {
            tracing::info!("verification PASSED");
            println!("Verification: PASS");
            Ok(())
        }
        Err(e) => {
            tracing::error!("verification FAILED: {}", e);
            println!("Verification: FAIL");
            anyhow::bail!("verification failed: {}", e)
        }
    }
}

fn verify_internal(
    verifiable: &remainder::verifiable_circuit::VerifiableCircuit<Fr>,
    transcript: &mut TranscriptReader<Fr, PoseidonSponge<Fr>>,
    proof_config: &shared_types::config::ProofConfig,
) -> Result<(), anyhow::Error> {
    verifiable
        .verify(
            global_verifier_circuit_description_hash_type(),
            transcript,
            proof_config,
        )
        .map_err(|e| anyhow::anyhow!("{}", e))
}

fn i64_to_fr(val: i64) -> Fr {
    use shared_types::Field;
    if val >= 0 {
        Fr::from(val as u64)
    } else {
        Fr::from(val.unsigned_abs()).neg()
    }
}
