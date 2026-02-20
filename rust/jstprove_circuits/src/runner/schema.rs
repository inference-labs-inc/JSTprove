use serde::{Deserialize, Serialize};

use crate::circuit_functions::utils::onnx_model::CircuitParams;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledCircuit {
    #[serde(with = "serde_bytes")]
    pub circuit: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub witness_solver: Vec<u8>,
    #[serde(default)]
    pub metadata: Option<CircuitParams>,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessBundle {
    #[serde(with = "serde_bytes")]
    pub witness: Vec<u8>,
    #[serde(default)]
    pub output_data: Option<Vec<i64>>,
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
