use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct CompiledCircuit {
    pub circuit: Vec<u8>,
    pub witness_solver: Vec<u8>,
    #[serde(default)]
    pub metadata: Option<Metadata>,
}

#[derive(Serialize, Deserialize, Default)]
pub struct Metadata {
    #[serde(default)]
    pub input_shapes: Vec<Vec<usize>>,
    #[serde(default)]
    pub output_shapes: Vec<Vec<usize>>,
    #[serde(default)]
    pub scale: Option<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct WitnessRequest {
    pub circuit: Vec<u8>,
    pub witness_solver: Vec<u8>,
    pub inputs: Vec<u8>,
    pub outputs: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
pub struct WitnessBundle {
    pub witness: Vec<u8>,
    #[serde(default)]
    pub output_data: Option<Vec<i64>>,
}

#[derive(Serialize, Deserialize)]
pub struct ProveRequest {
    pub circuit: Vec<u8>,
    pub witness: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
pub struct ProofBundle {
    pub proof: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
pub struct VerifyRequest {
    pub circuit: Vec<u8>,
    pub witness: Vec<u8>,
    pub proof: Vec<u8>,
    pub inputs: Vec<u8>,
    pub outputs: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
pub struct VerifyResponse {
    pub valid: bool,
    #[serde(default)]
    pub error: Option<String>,
}
