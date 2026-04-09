use serde::{Deserialize, Serialize};

use super::version::ArtifactVersion;
use crate::circuit_functions::utils::onnx_model::CircuitParams;
use crate::curve::Curve;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledCircuit {
    #[serde(with = "serde_bytes")]
    pub circuit: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub witness_solver: Vec<u8>,
    #[serde(default)]
    pub metadata: Option<CircuitParams>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curve: Option<Curve>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<ArtifactVersion>,
}

impl CompiledCircuit {
    /// Resolve the curve this circuit was compiled with. Prefers the
    /// top-level `curve` field, falls back to `metadata.curve`, then to
    /// field-level detection from the serialized circuit bytes. Returns
    /// `None` only for bundles that predate curve stamping and whose
    /// serialized field modulus does not match any known curve.
    #[must_use]
    pub fn resolved_curve(&self) -> Option<Curve> {
        if let Some(c) = self.curve {
            return Some(c);
        }
        if let Some(c) = self.metadata.as_ref().and_then(|m| m.curve) {
            return Some(c);
        }
        Curve::detect_base_field(&self.circuit).ok()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessRequest {
    #[serde(with = "serde_bytes")]
    pub circuit: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub witness_solver: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub inputs: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub outputs: Vec<u8>,
    #[serde(default)]
    pub metadata: Option<CircuitParams>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessBundle {
    #[serde(with = "serde_bytes")]
    pub witness: Vec<u8>,
    #[serde(default)]
    pub output_data: Option<Vec<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<ArtifactVersion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProveRequest {
    #[serde(with = "serde_bytes")]
    pub circuit: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub witness: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofBundle {
    #[serde(with = "serde_bytes")]
    pub proof: Vec<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<ArtifactVersion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyRequest {
    #[serde(with = "serde_bytes")]
    pub circuit: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub witness: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub proof: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub inputs: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub outputs: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyResponse {
    pub valid: bool,
    #[serde(default)]
    pub error: Option<String>,
}
