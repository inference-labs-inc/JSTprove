use thiserror::Error;

use crate::circuit_functions::utils::onnx_types::ONNXLayer;

#[derive(Debug, Clone)]
pub enum LayerKind {
    Constant,
    Conv,
    Flatten,
    Gemm,
    MaxPool,
    ReLU,
    Reshape,
}

impl std::fmt::Display for LayerKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayerKind::Constant => write!(f, "Constant"),
            LayerKind::Conv => write!(f, "Conv"),
            LayerKind::Flatten => write!(f, "Flatten"),
            LayerKind::Gemm => write!(f, "Gemm"),
            LayerKind::MaxPool => write!(f, "MaxPool"),
            LayerKind::ReLU => write!(f, "ReLU"),
            LayerKind::Reshape => write!(f, "Reshape"),
        }
    }
}

impl TryFrom<&ONNXLayer> for LayerKind {
    type Error = LayerError;

    fn try_from(layer: &ONNXLayer) -> Result<Self, Self::Error> {
        match layer.op_type.as_str() {
            "Constant" => Ok(LayerKind::Constant),
            "Conv" => Ok(LayerKind::Conv),
            "Flatten" => Ok(LayerKind::Flatten),
            "Gemm" => Ok(LayerKind::Gemm),
            "MaxPool" => Ok(LayerKind::MaxPool),
            "Relu" => Ok(LayerKind::ReLU),
            "Reshape" => Ok(LayerKind::Reshape),
            other      => Err(LayerError::UnknownOp { op_type: other.to_string() }),
            // _ => Err(())
        }
    }
}

#[derive(Debug, Error)]
pub enum LayerError {
    #[error("{layer} is missing input: {name}")]
    MissingInput { layer: LayerKind, name: String },

    #[error("Shape mismatch in {layer}: expected {expected:?}, got {got:?}")]
    ShapeMismatch { layer: LayerKind, expected: Vec<usize>, got: Vec<usize> },

    #[error("{layer} is missing parameter: {param}")]
    MissingParameter { layer: LayerKind, param: String },

    #[error("Unsupported config in {layer}: {msg}")]
    UnsupportedConfig { layer: LayerKind, msg: String },

    #[error("Invalid shape in {layer}: {msg}")]
    InvalidShape { layer: LayerKind, msg: String },

    #[error("Unknown operator type: {op_type}")]
    UnknownOp { op_type: String },

    #[error("Other error in {layer}: {msg}")]
    Other{layer: LayerKind, msg: String},
}
