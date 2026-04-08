#![allow(clippy::doc_markdown)]

use std::collections::HashSet;
use std::hash::BuildHasher;
use std::path::Path;
use std::sync::LazyLock;

use crate::circuit_functions::utils::onnx_model::{Architecture, CircuitParams, WANDB};
use crate::circuit_functions::utils::onnx_types::ONNXIO;
use crate::curve::Curve;
use crate::io::io_reader::onnx_context::OnnxContext;
use crate::proof_system::ProofSystem;
use crate::runner::errors::RunError;
use crate::runner::schema::{CompiledCircuit, WitnessBundle, WitnessRequest};
use crate::runner::verify_extract::ExtractedOutput;

pub use crate::circuit_functions::utils::onnx_model::{
    Architecture as ArchitectureType, CircuitParams as CircuitParamsType, WANDB as WANDBType,
};
pub use crate::circuit_functions::utils::onnx_types::ONNXIO as ONNXIOType;
pub use crate::curve::{Curve as CurveType, CurveParseError};
pub use crate::expander_metadata::ExpanderMetadata;
pub use crate::proof_system::{ProofSystem as ProofSystemType, ProofSystemParseError};
pub use crate::runner::errors::RunError as ApiError;
pub use crate::runner::schema::{
    CompiledCircuit as CompiledCircuitType, WitnessBundle as WitnessBundleType,
};
pub use crate::runner::verify_extract::ExtractedOutput as ExtractedOutputType;
pub use crate::runner::version::{ArtifactVersion, jstprove_artifact_version};

/// # Errors
/// Returns `RunError` on compilation or serialization failure.
pub fn compile(
    circuit_path: &str,
    curve: Curve,
    params: CircuitParams,
    architecture: Architecture,
    wandb: WANDB,
    compress: bool,
) -> Result<(), RunError> {
    OnnxContext::set_params(params.clone());
    OnnxContext::set_architecture(architecture);
    OnnxContext::set_wandb(wandb);

    match curve {
        Curve::Bn254 => crate::onnx::compile_bn254(circuit_path, compress, Some(params)),
        Curve::Goldilocks | Curve::GoldilocksBasefold => {
            crate::onnx::compile_goldilocks(circuit_path, compress, Some(params))
        }
        Curve::GoldilocksExt2 => {
            crate::onnx::compile_goldilocks_ext2(circuit_path, compress, Some(params))
        }
        Curve::GoldilocksWhir => {
            crate::onnx::compile_goldilocks_whir(circuit_path, compress, Some(params))
        }
        Curve::GoldilocksWhirPQ => {
            crate::onnx::compile_goldilocks_whir_pq(circuit_path, compress, Some(params))
        }
    }
}

/// # Errors
/// Returns `RunError` on witness generation failure.
pub fn witness(
    curve: Curve,
    req: &WitnessRequest,
    compress: bool,
) -> Result<WitnessBundle, RunError> {
    if let Some(ref params) = req.metadata {
        OnnxContext::set_params(params.clone());
    }

    match curve {
        Curve::Bn254 => crate::onnx::witness_bn254(req, compress),
        Curve::Goldilocks | Curve::GoldilocksBasefold => {
            crate::onnx::witness_goldilocks(req, compress)
        }
        Curve::GoldilocksExt2 | Curve::GoldilocksWhir | Curve::GoldilocksWhirPQ => {
            Err(RunError::Unsupported(format!(
                "{curve} raw JSON witness not supported; use witness_f64"
            )))
        }
    }
}

/// # Errors
/// Returns `RunError` on witness generation or activation mismatch.
pub fn witness_f64(
    curve: Curve,
    circuit_bytes: &[u8],
    solver_bytes: &[u8],
    params: &CircuitParams,
    activations: &[f64],
    initializers: &[(Vec<f64>, Vec<usize>)],
    compress: bool,
) -> Result<WitnessBundle, RunError> {
    match curve {
        Curve::Bn254 => crate::onnx::witness_bn254_from_f64(
            circuit_bytes,
            solver_bytes,
            params,
            activations,
            initializers,
            compress,
        ),
        Curve::Goldilocks | Curve::GoldilocksBasefold => crate::onnx::witness_goldilocks_from_f64(
            circuit_bytes,
            solver_bytes,
            params,
            activations,
            initializers,
            compress,
        ),
        Curve::GoldilocksExt2 => crate::onnx::witness_goldilocks_ext2_from_f64(
            circuit_bytes,
            solver_bytes,
            params,
            activations,
            initializers,
            compress,
        ),
        Curve::GoldilocksWhir => crate::onnx::witness_goldilocks_whir_from_f64(
            circuit_bytes,
            solver_bytes,
            params,
            activations,
            initializers,
            compress,
        ),
        Curve::GoldilocksWhirPQ => crate::onnx::witness_goldilocks_whir_pq_from_f64(
            circuit_bytes,
            solver_bytes,
            params,
            activations,
            initializers,
            compress,
        ),
    }
}

/// # Errors
/// Returns `RunError` on proof generation failure.
pub fn prove(
    curve: Curve,
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    compress: bool,
) -> Result<Vec<u8>, RunError> {
    match curve {
        Curve::Bn254 => crate::onnx::prove_bn254(circuit_bytes, witness_bytes, compress),
        Curve::Goldilocks | Curve::GoldilocksBasefold => {
            crate::onnx::prove_goldilocks(circuit_bytes, witness_bytes, compress)
        }
        Curve::GoldilocksExt2 => {
            crate::onnx::prove_goldilocks_ext2(circuit_bytes, witness_bytes, compress)
        }
        Curve::GoldilocksWhir => {
            crate::onnx::prove_goldilocks_whir(circuit_bytes, witness_bytes, compress)
        }
        Curve::GoldilocksWhirPQ => {
            crate::onnx::prove_goldilocks_whir_pq(circuit_bytes, witness_bytes, compress)
        }
    }
}

/// # Errors
/// Returns `RunError` on verification failure.
pub fn verify(
    curve: Curve,
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
) -> Result<bool, RunError> {
    match curve {
        Curve::Bn254 => crate::onnx::verify_bn254(circuit_bytes, witness_bytes, proof_bytes),
        Curve::Goldilocks | Curve::GoldilocksBasefold => {
            crate::onnx::verify_goldilocks(circuit_bytes, witness_bytes, proof_bytes)
        }
        Curve::GoldilocksExt2 => {
            crate::onnx::verify_goldilocks_ext2(circuit_bytes, witness_bytes, proof_bytes)
        }
        Curve::GoldilocksWhir => {
            crate::onnx::verify_goldilocks_whir(circuit_bytes, witness_bytes, proof_bytes)
        }
        Curve::GoldilocksWhirPQ => {
            crate::onnx::verify_goldilocks_whir_pq(circuit_bytes, witness_bytes, proof_bytes)
        }
    }
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

static OP_REGISTRY: LazyLock<Vec<OpInfo>> = LazyLock::new(|| {
    use crate::circuit_functions::layers::LayerKind;

    const SPATIAL_OPS: &[&str] = &[
        "Conv",
        "ConvTranspose",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
    ];

    const ELEMENTWISE_OPS: &[&str] = &[
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        "Max",
        "Min",
        "ReLU",
        "LeakyRelu",
        "Sigmoid",
        "Tanh",
        "Exp",
        "Log",
        "Sqrt",
        "Erf",
        "Neg",
        "Clip",
        "HardSwish",
        "Gelu",
        "Sin",
        "Cos",
        "Not",
        "And",
        "Equal",
        "Greater",
        "Less",
        "Identity",
        "Cast",
    ];

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

        const SPATIAL_OPS: &[&str] = &[
            "Conv",
            "ConvTranspose",
            "MaxPool",
            "AveragePool",
            "GlobalAveragePool",
        ];

        const ELEMENTWISE_OPS: &[&str] = &[
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Pow",
            "Max",
            "Min",
            "ReLU",
            "LeakyRelu",
            "Sigmoid",
            "Tanh",
            "Exp",
            "Log",
            "Sqrt",
            "Erf",
            "Neg",
            "Clip",
            "HardSwish",
            "Gelu",
            "Sin",
            "Cos",
            "Not",
            "And",
            "Equal",
            "Greater",
            "Less",
            "Identity",
            "Cast",
        ];

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
