use std::path::Path;

use anyhow::Result;
use remainder::mle::evals::MultilinearExtension;
use shared_types::config::global_config::global_verifier_circuit_description_hash_type;
use shared_types::config::GKRCircuitVerifierConfig;
use shared_types::transcript::poseidon_sponge::PoseidonSponge;
use shared_types::transcript::TranscriptReader;
use shared_types::{perform_function_under_verifier_config, Fr};

use crate::runner::circuit_builder::{self, Visibility};
use crate::util::i64_to_fr;

pub fn run(model_path: &Path, proof_path: &Path, input_path: &Path) -> Result<()> {
    tracing::info!("loading model from {}", model_path.display());
    let model = super::compile::load_model(model_path)?;

    tracing::info!("loading proof from {}", proof_path.display());
    let proof = super::prove::load_proof(proof_path)?;

    tracing::info!("loading input from {}", input_path.display());
    let quantized_input = super::witness::load_and_quantize_input(input_path, model.scale_config.alpha)?;

    let result = verify_with_model(&model, &proof, &quantized_input);

    if result.is_ok() {
        tracing::info!("verification PASSED");
    } else {
        tracing::error!("verification FAILED: {}", result.as_ref().unwrap_err());
    }

    result
}

pub fn verify_with_model(
    model: &crate::onnx::quantizer::QuantizedModel,
    proof: &super::prove::SerializableProof,
    quantized_input: &[i64],
) -> Result<()> {
    let witness = super::witness::compute_witness(model, quantized_input)?;

    let input_name = model.graph.input_names.first()
        .ok_or_else(|| anyhow::anyhow!("model has no input names defined"))?
        .clone();
    let input_size = witness.get(&input_name)
        .map(|v| v.len())
        .ok_or_else(|| anyhow::anyhow!("witness missing input shred '{}'", input_name))?;

    let build_result = circuit_builder::build_circuit(model, input_size)?;
    let mut circuit = build_result.circuit;

    for (name, entry) in &build_result.manifest {
        if entry.visibility == Visibility::Public {
            let values = witness.get(name)
                .ok_or_else(|| anyhow::anyhow!("missing public input '{}'", name))?;
            let mle = MultilinearExtension::new(
                values.iter().map(|&v| i64_to_fr(v)).collect(),
            );
            circuit.set_input(name, mle);
        }
    }

    let verifiable = circuit.gen_verifiable_circuit()?;

    let mut transcript_reader =
        TranscriptReader::<Fr, PoseidonSponge<Fr>>::new(proof.transcript.clone());

    let verifier_config =
        GKRCircuitVerifierConfig::new_from_proof_config(&proof.proof_config, false);

    let result: std::result::Result<(), _> = perform_function_under_verifier_config!(
        verify_internal,
        &verifier_config,
        &verifiable,
        &mut transcript_reader,
        &proof.proof_config
    );

    result.map_err(|e| anyhow::anyhow!("verification failed: {}", e))
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
