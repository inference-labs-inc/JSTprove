#![allow(clippy::doc_markdown)]

use std::collections::HashSet;
use std::hash::BuildHasher;
use std::path::Path;
use std::sync::LazyLock;

use crate::circuit_functions::utils::onnx_model::{Architecture, CircuitParams, WANDB};
use crate::circuit_functions::utils::onnx_types::ONNXIO;
use crate::io::io_reader::onnx_context::OnnxContext;
use crate::proof_config::ProofConfig;
use crate::proof_system::ProofSystem;
use crate::runner::errors::RunError;
use crate::runner::schema::{CompiledCircuit, WitnessBundle, WitnessRequest};
use crate::runner::verify_extract::{ExtractedOutput, VerifiedOutput};

pub use crate::circuit_functions::utils::onnx_model::{
    Architecture as ArchitectureType, CircuitParams as CircuitParamsType, WANDB as WANDBType,
};
pub use crate::circuit_functions::utils::onnx_types::ONNXIO as ONNXIOType;
pub use crate::expander_metadata::ExpanderMetadata;
pub use crate::proof_config::{
    Field as FieldType, ProofConfig as ProofConfigType, ProofConfigError,
    StampedProofConfig as StampedProofConfigType,
};
pub use crate::proof_system::{ProofSystem as ProofSystemType, ProofSystemParseError};
pub use crate::runner::errors::RunError as ApiError;
pub use crate::runner::schema::{
    CompiledCircuit as CompiledCircuitType, WitnessBundle as WitnessBundleType,
};
pub use crate::runner::verify_extract::ExtractedOutput as ExtractedOutputType;
pub use crate::runner::verify_extract::VerifiedOutput as VerifiedOutputType;
pub use crate::runner::version::{ArtifactVersion, jstprove_artifact_version};

/// Invoke the per-config implementation closure for the given proof
/// configuration.
///
/// Centralizes the [`ProofConfig`] → concrete-implementation mapping
/// so that all per-config dispatchers (`compile`, `witness`,
/// `witness_f64`, `prove`, `verify`, `verify_and_extract`) share a
/// single authoritative routing table. Adding a new variant requires
/// updating one place.
fn dispatch_by_proof_config<R>(
    config: ProofConfig,
    bn254_raw: impl FnOnce() -> R,
    goldilocks_raw: impl FnOnce() -> R,
    goldilocks_basefold: impl FnOnce() -> R,
    goldilocks_ext2_basefold: impl FnOnce() -> R,
    goldilocks_ext3_whir: impl FnOnce() -> R,
    goldilocks_ext4_whir: impl FnOnce() -> R,
) -> R {
    match config {
        ProofConfig::Bn254Raw => bn254_raw(),
        ProofConfig::GoldilocksRaw => goldilocks_raw(),
        ProofConfig::GoldilocksBasefold => goldilocks_basefold(),
        ProofConfig::GoldilocksExt2Basefold => goldilocks_ext2_basefold(),
        ProofConfig::GoldilocksExt3Whir => goldilocks_ext3_whir(),
        ProofConfig::GoldilocksExt4Whir => goldilocks_ext4_whir(),
    }
}

/// # Errors
/// Returns `RunError` on compilation or serialization failure.
pub fn compile(
    circuit_path: &str,
    config: ProofConfig,
    params: CircuitParams,
    architecture: Architecture,
    wandb: WANDB,
    compress: bool,
) -> Result<(), RunError> {
    OnnxContext::set_all(architecture, params.clone(), Some(wandb));
    let p = Some(params);
    dispatch_by_proof_config(
        config,
        || crate::onnx::compile_bn254(circuit_path, compress, p.clone()),
        || crate::onnx::compile_goldilocks(circuit_path, compress, p.clone()),
        || crate::onnx::compile_goldilocks_basefold(circuit_path, compress, p.clone()),
        || crate::onnx::compile_goldilocks_ext2(circuit_path, compress, p.clone()),
        || crate::onnx::compile_goldilocks_whir(circuit_path, compress, p.clone()),
        || crate::onnx::compile_goldilocks_whir_pq(circuit_path, compress, p.clone()),
    )
}

fn unsupported_witness(config: ProofConfig) -> Result<WitnessBundle, RunError> {
    Err(RunError::Unsupported(format!(
        "{config} raw JSON witness not supported; use witness_f64"
    )))
}

/// # Errors
/// Returns `RunError` on witness generation failure.
pub fn witness(
    config: ProofConfig,
    req: &WitnessRequest,
    compress: bool,
) -> Result<WitnessBundle, RunError> {
    if let Some(ref params) = req.metadata {
        OnnxContext::set_params(params.clone());
    }

    dispatch_by_proof_config(
        config,
        || crate::onnx::witness_bn254(req, compress),
        || crate::onnx::witness_goldilocks(req, compress),
        || unsupported_witness(config),
        || unsupported_witness(config),
        || unsupported_witness(config),
        || unsupported_witness(config),
    )
}

/// # Errors
/// Returns `RunError` on witness generation or activation mismatch.
#[allow(clippy::too_many_arguments)]
pub fn witness_f64(
    config: ProofConfig,
    circuit_bytes: &[u8],
    solver_bytes: &[u8],
    params: &CircuitParams,
    activations: &[f64],
    initializers: &[(Vec<f64>, Vec<usize>)],
    compress: bool,
) -> Result<WitnessBundle, RunError> {
    dispatch_by_proof_config(
        config,
        || {
            crate::onnx::witness_bn254_from_f64(
                circuit_bytes,
                solver_bytes,
                params,
                activations,
                initializers,
                compress,
            )
        },
        || {
            crate::onnx::witness_goldilocks_from_f64(
                circuit_bytes,
                solver_bytes,
                params,
                activations,
                initializers,
                compress,
            )
        },
        || {
            crate::onnx::witness_goldilocks_basefold_from_f64(
                circuit_bytes,
                solver_bytes,
                params,
                activations,
                initializers,
                compress,
            )
        },
        || {
            crate::onnx::witness_goldilocks_ext2_from_f64(
                circuit_bytes,
                solver_bytes,
                params,
                activations,
                initializers,
                compress,
            )
        },
        || {
            crate::onnx::witness_goldilocks_whir_from_f64(
                circuit_bytes,
                solver_bytes,
                params,
                activations,
                initializers,
                compress,
            )
        },
        || {
            crate::onnx::witness_goldilocks_whir_pq_from_f64(
                circuit_bytes,
                solver_bytes,
                params,
                activations,
                initializers,
                compress,
            )
        },
    )
}

/// # Errors
/// Returns `RunError` on proof generation failure.
pub fn prove(
    config: ProofConfig,
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    compress: bool,
) -> Result<Vec<u8>, RunError> {
    dispatch_by_proof_config(
        config,
        || crate::onnx::prove_bn254(circuit_bytes, witness_bytes, compress),
        || crate::onnx::prove_goldilocks(circuit_bytes, witness_bytes, compress),
        || crate::onnx::prove_goldilocks_basefold(circuit_bytes, witness_bytes, compress),
        || crate::onnx::prove_goldilocks_ext2(circuit_bytes, witness_bytes, compress),
        || crate::onnx::prove_goldilocks_whir(circuit_bytes, witness_bytes, compress),
        || crate::onnx::prove_goldilocks_whir_pq(circuit_bytes, witness_bytes, compress),
    )
}

/// # Errors
/// Returns `RunError` on verification failure.
pub fn verify(
    config: ProofConfig,
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
) -> Result<bool, RunError> {
    dispatch_by_proof_config(
        config,
        || crate::onnx::verify_bn254(circuit_bytes, witness_bytes, proof_bytes),
        || crate::onnx::verify_goldilocks(circuit_bytes, witness_bytes, proof_bytes),
        || crate::onnx::verify_goldilocks_basefold(circuit_bytes, witness_bytes, proof_bytes),
        || crate::onnx::verify_goldilocks_ext2(circuit_bytes, witness_bytes, proof_bytes),
        || crate::onnx::verify_goldilocks_whir(circuit_bytes, witness_bytes, proof_bytes),
        || crate::onnx::verify_goldilocks_whir_pq(circuit_bytes, witness_bytes, proof_bytes),
    )
}

/// # Errors
/// Returns `RunError` on verification or output extraction failure.
pub fn verify_and_extract(
    config: ProofConfig,
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
    num_inputs: usize,
    expected_inputs: Option<&[f64]>,
) -> Result<VerifiedOutput, RunError> {
    dispatch_by_proof_config(
        config,
        || {
            crate::onnx::verify_and_extract_bn254(
                circuit_bytes,
                witness_bytes,
                proof_bytes,
                num_inputs,
                expected_inputs,
            )
        },
        || {
            crate::onnx::verify_and_extract_goldilocks(
                circuit_bytes,
                witness_bytes,
                proof_bytes,
                num_inputs,
                expected_inputs,
            )
        },
        || {
            crate::onnx::verify_and_extract_goldilocks_basefold(
                circuit_bytes,
                witness_bytes,
                proof_bytes,
                num_inputs,
                expected_inputs,
            )
        },
        || {
            crate::onnx::verify_and_extract_goldilocks_ext2(
                circuit_bytes,
                witness_bytes,
                proof_bytes,
                num_inputs,
                expected_inputs,
            )
        },
        || {
            crate::onnx::verify_and_extract_goldilocks_whir(
                circuit_bytes,
                witness_bytes,
                proof_bytes,
                num_inputs,
                expected_inputs,
            )
        },
        || {
            crate::onnx::verify_and_extract_goldilocks_whir_pq(
                circuit_bytes,
                witness_bytes,
                proof_bytes,
                num_inputs,
                expected_inputs,
            )
        },
    )
}

/// # Errors
/// Returns `RunError` on deserialization or extraction failure.
pub fn extract_outputs(
    witness_bytes: &[u8],
    num_model_inputs: usize,
) -> Result<ExtractedOutput, RunError> {
    crate::runner::verify_extract::extract_outputs_from_witness(witness_bytes, num_model_inputs)
}

/// # Errors
/// Returns `RunError` on deserialization failure.
pub fn read_circuit_bundle(path: &str) -> Result<CompiledCircuit, RunError> {
    crate::runner::main_runner::read_circuit_msgpack(path)
}

#[must_use]
pub fn try_load_metadata(circuit_path: &str) -> Option<CircuitParams> {
    crate::runner::main_runner::try_load_metadata_from_circuit(circuit_path)
}

/// # Errors
/// Returns `RunError` if ONNX parsing, quantization, or shape inference fails.
pub fn generate_metadata(
    onnx_path: &Path,
) -> Result<crate::expander_metadata::ExpanderMetadata, RunError> {
    crate::expander_metadata::generate_from_onnx(onnx_path)
        .map_err(|e| RunError::Compile(format!("{e:#}")))
}

/// # Errors
/// Returns `RunError` if ONNX parsing, quantization, or shape inference fails.
pub fn generate_metadata_with_options(
    onnx_path: &Path,
    weights_as_inputs: bool,
) -> Result<crate::expander_metadata::ExpanderMetadata, RunError> {
    crate::expander_metadata::generate_from_onnx_with_options(onnx_path, weights_as_inputs)
        .map_err(|e| RunError::Compile(format!("{e:#}")))
}

/// Generate metadata using externally-resolved tensor shapes, bypassing
/// jstprove's internal shape inference. The provided shape map is trusted
/// as the single source of truth.
///
/// # Errors
/// Returns `RunError` if ONNX parsing or quantization fails.
#[allow(clippy::implicit_hasher)]
pub fn generate_metadata_with_shapes(
    onnx_path: &Path,
    precomputed_shapes: std::collections::HashMap<String, Vec<usize>>,
) -> Result<crate::expander_metadata::ExpanderMetadata, RunError> {
    crate::expander_metadata::generate_from_onnx_with_shapes(onnx_path, precomputed_shapes)
        .map_err(|e| RunError::Compile(format!("{e:#}")))
}

#[must_use]
pub fn supported_ops(proof_system: ProofSystem) -> &'static [&'static str] {
    proof_system.supported_ops()
}

#[derive(Debug, Clone)]
pub struct OpInfo {
    pub name: &'static str,
    pub is_spatial: bool,
    pub is_elementwise: bool,
}

use crate::circuit_functions::layers::LayerKind;

pub const SPATIAL_OPS: &[&str] = &[
    LayerKind::Conv.name(),
    LayerKind::ConvTranspose.name(),
    LayerKind::MaxPool.name(),
    LayerKind::AveragePool.name(),
    LayerKind::GlobalAveragePool.name(),
];

pub const ELEMENTWISE_OPS: &[&str] = &[
    LayerKind::Add.name(),
    LayerKind::Sub.name(),
    LayerKind::Mul.name(),
    LayerKind::Div.name(),
    LayerKind::Pow.name(),
    LayerKind::Max.name(),
    LayerKind::Min.name(),
    LayerKind::ReLU.name(),
    LayerKind::LeakyRelu.name(),
    LayerKind::Sigmoid.name(),
    LayerKind::Tanh.name(),
    LayerKind::Exp.name(),
    LayerKind::Log.name(),
    LayerKind::Sqrt.name(),
    LayerKind::Erf.name(),
    LayerKind::Neg.name(),
    LayerKind::Clip.name(),
    LayerKind::HardSwish.name(),
    LayerKind::Gelu.name(),
    LayerKind::Sin.name(),
    LayerKind::Cos.name(),
    LayerKind::Not.name(),
    LayerKind::And.name(),
    LayerKind::Equal.name(),
    LayerKind::Greater.name(),
    LayerKind::Less.name(),
    LayerKind::Identity.name(),
    LayerKind::Cast.name(),
];

static OP_REGISTRY: LazyLock<Vec<OpInfo>> = LazyLock::new(|| {
    LayerKind::SUPPORTED_OP_NAMES
        .iter()
        .map(|&name| OpInfo {
            name,
            is_spatial: SPATIAL_OPS.contains(&name),
            is_elementwise: ELEMENTWISE_OPS.contains(&name),
        })
        .collect()
});

#[must_use]
pub fn op_registry() -> &'static [OpInfo] {
    &OP_REGISTRY
}

const ONNX_ELEM_TYPE_FLOAT: i16 = 1;

/// # Errors
/// Returns `RunError` if a WANDB entry is missing its shape.
pub fn populate_wai_inputs<S: BuildHasher>(
    params: &mut CircuitParams,
    wandb: &WANDB,
    exclude: &HashSet<String, S>,
) -> Result<(), RunError> {
    params.weights_as_inputs = true;
    for wb in &wandb.w_and_b {
        if exclude.contains(&wb.name) {
            continue;
        }
        let shape = wb.shape.get(&wb.name).cloned().ok_or_else(|| {
            RunError::Compile(format!(
                "wandb entry '{}' missing shape for WAI input population",
                wb.name
            ))
        })?;
        params.inputs.push(ONNXIO {
            name: wb.name.clone(),
            elem_type: ONNX_ELEM_TYPE_FLOAT,
            shape,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_ops_nonempty() {
        let ops = supported_ops(ProofSystem::Expander);
        assert!(!ops.is_empty());
    }

    #[test]
    fn op_registry_populated() {
        let registry = op_registry();
        assert!(!registry.is_empty());
        let conv = registry.iter().find(|o| o.name == "Conv");
        assert!(conv.is_some());
        assert!(conv.unwrap().is_spatial);
        assert!(!conv.unwrap().is_elementwise);
    }

    #[test]
    fn op_registry_relu_is_elementwise() {
        let registry = op_registry();
        let relu = registry.iter().find(|o| o.name == "ReLU").unwrap();
        assert!(relu.is_elementwise);
        assert!(!relu.is_spatial);
    }

    #[test]
    fn remainder_ops_subset() {
        let expander = supported_ops(ProofSystem::Expander);
        for &op in supported_ops(ProofSystem::Remainder) {
            assert!(expander.contains(&op), "{op} missing from Expander");
        }
    }

    #[test]
    fn hardcoded_ops_exist_in_supported() {
        use crate::circuit_functions::layers::LayerKind;

        let supported = LayerKind::SUPPORTED_OP_NAMES;
        for &op in SPATIAL_OPS {
            assert!(
                supported.contains(&op),
                "SPATIAL_OPS entry {op:?} not in LayerKind::SUPPORTED_OP_NAMES"
            );
        }
        for &op in ELEMENTWISE_OPS {
            assert!(
                supported.contains(&op),
                "ELEMENTWISE_OPS entry {op:?} not in LayerKind::SUPPORTED_OP_NAMES"
            );
        }
    }
}
