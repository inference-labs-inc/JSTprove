use crate::circuit_functions::{
    layers::LayerError,
    utils::{ArrayConversionError, BuildError, PatternError, RescaleError, UtilsError},
};
use crate::io::io_reader::onnx_context::OnnxContextError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CircuitError {
    #[error("Layer failed: {0}")]
    Layer(#[from] LayerError),

    #[error(transparent)]
    UtilsError(#[from] UtilsError),

    #[error("Failed to parse weights: {0}")]
    InvalidWeightsFormat(#[from] rmpv::ext::Error),

    #[error("Architecture definition is empty")]
    EmptyArchitecture,

    #[error("Graph error: {0}")]
    GraphPatternError(#[from] PatternError),

    #[error("Array conversion error: {0}")]
    ArrayConversionError(#[from] ArrayConversionError),

    #[error("Rescaling error: {0}")]
    RescaleError(#[from] RescaleError),

    #[error("Error building layers: {0}")]
    BuildError(#[from] BuildError),

    #[error("ONNX context error: {0}")]
    OnnxContext(#[from] OnnxContextError),

    #[error("Other circuit error: {0}")]
    Other(String),
}
