use std::path::Path;

use anyhow::Result;
use remainder::mle::evals::MultilinearExtension;
use shared_types::config::global_config::global_verifier_circuit_description_hash_type;
use shared_types::config::GKRCircuitVerifierConfig;
use shared_types::transcript::poseidon_sponge::PoseidonSponge;
use shared_types::transcript::TranscriptReader;
use shared_types::{perform_function_under_verifier_config, Fr};

use crate::cli::{self, OutputMode, StepPrinter};
use crate::padding::next_power_of_two;
use crate::runner::circuit_builder::{self, Visibility};
use crate::util::i64_to_fr;

pub fn run(
    model_path: &Path,
    proof_path: &Path,
    input_path: &Path,
    mode: OutputMode,
) -> Result<()> {
    let mut steps = StepPrinter::new(4, mode);

    steps.step("Loading model");
    let mut model = super::compile::load_model(model_path)?;
    steps.detail(&format!("{} layers", model.graph.layers.len()));

    steps.step("Loading proof");
    let proof = super::prove::load_proof(proof_path)?;
    if !proof.observed_n_bits.is_empty() {
        for layer in &mut model.graph.layers {
            if let Some(&obs) = proof.observed_n_bits.get(&layer.name) {
                layer.n_bits = Some(obs);
            }
        }
    }

    steps.step("Quantizing input");
    let quantized_input =
        super::witness::load_and_quantize_input(input_path, model.scale_config.alpha)?;
    steps.detail(&format!("{} elements", quantized_input.len()));

    steps.step("Verifying proof");
    let input_padded_size = next_power_of_two(quantized_input.len());
    let public_shreds = super::witness::prepare_public_shreds(
        &model,
        &quantized_input,
        &proof.expected_output,
        &proof.observed_n_bits,
    )?;
    let build_result = circuit_builder::build_circuit(&model, input_padded_size)?;
    let mut circuit = build_result.circuit;
    let public_count = build_result
        .manifest
        .values()
        .filter(|e| e.visibility == Visibility::Public)
        .count();
    let committed_count = build_result.manifest.len() - public_count;
    steps.detail(&format!(
        "{public_count} public, {committed_count} committed shreds"
    ));

    for (name, entry) in &build_result.manifest {
        if entry.visibility == Visibility::Public {
            let values = public_shreds
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("missing public input '{name}'"))?;
            let mle = MultilinearExtension::new(values.iter().map(|&v| i64_to_fr(v)).collect());
            circuit.set_input(name, mle);
        }
    }

    let verifiable = circuit.gen_verifiable_circuit()?;
    let sp = cli::spinner("checking GKR sumcheck transcript", mode);
    let verify_result = run_verify(&verifiable, &proof);
    sp.finish_and_clear();
    let result = verify_result;

    match result {
        Ok(()) => {
            steps.finish_ok("Verification passed");
            Ok(())
        }
        Err(ref e) => {
            steps.finish_err("Verification failed");
            steps.detail(&describe_verification_error(e));
            result
        }
    }
}

pub fn verify_with_model(
    model: &crate::onnx::quantizer::QuantizedModel,
    proof: &super::prove::SerializableProof,
    quantized_input: &[i64],
) -> Result<()> {
    let input_padded_size = next_power_of_two(quantized_input.len());

    let public_shreds = super::witness::prepare_public_shreds(
        model,
        quantized_input,
        &proof.expected_output,
        &proof.observed_n_bits,
    )?;

    let build_result = circuit_builder::build_circuit(model, input_padded_size)?;
    let mut circuit = build_result.circuit;

    for (name, entry) in &build_result.manifest {
        if entry.visibility == Visibility::Public {
            let values = public_shreds
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("missing public input '{name}'"))?;
            let mle = MultilinearExtension::new(values.iter().map(|&v| i64_to_fr(v)).collect());
            circuit.set_input(name, mle);
        }
    }

    let verifiable = circuit.gen_verifiable_circuit()?;
    run_verify(&verifiable, proof)
}

fn run_verify(
    verifiable: &remainder::verifiable_circuit::VerifiableCircuit<Fr>,
    proof: &super::prove::SerializableProof,
) -> Result<()> {
    let mut transcript_reader =
        TranscriptReader::<Fr, PoseidonSponge<Fr>>::new(proof.transcript.clone());

    let verifier_config =
        GKRCircuitVerifierConfig::new_from_proof_config(&proof.proof_config, false);

    let result: std::result::Result<(), _> = perform_function_under_verifier_config!(
        verify_internal,
        &verifier_config,
        verifiable,
        &mut transcript_reader,
        &proof.proof_config
    );

    result.map_err(|e| anyhow::anyhow!("verification failed: {e}"))
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
        .map_err(|e| anyhow::anyhow!("{e}"))
}

fn describe_verification_error(err: &anyhow::Error) -> String {
    let msg = format!("{err}");
    if msg.contains("PublicInputLayerValuesMismatch") || msg.contains("not as expected") {
        "public inputs do not match the proof transcript".into()
    } else if msg.contains("EvaluationMismatch") {
        "input layer evaluation mismatch at GKR claim point".into()
    } else if msg.contains("ErrorWhenVerifyingLayer") {
        "sumcheck rejected at an intermediate layer — proof may be corrupted or from a different circuit".into()
    } else if msg.contains("ErrorWhenVerifyingOutputLayer") {
        "output layer invalid — claimed output does not match circuit computation".into()
    } else if msg.contains("ClaimTrackerNotEmpty") {
        "proof structurally incomplete — unresolved claims remain after verification".into()
    } else if msg.contains("InputShredLengthMismatch") {
        "input dimensions do not match circuit — witness may be from a different model".into()
    } else {
        msg
    }
}
