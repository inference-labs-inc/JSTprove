use thiserror::Error;
use crate::circuit_functions::{layers::LayerError, utils::{ArrayConversionError, BuildError, PatternError, RescaleError, UtilsError}};

#[derive(Debug, Error)]
pub enum CircuitError {
    #[error("Layer failed: {0}")]
    Layer(#[from] LayerError),

    #[error(transparent)]
    UtilsError(#[from] UtilsError),

    #[error("Failed to parse weights JSON: {0}")]
    InvalidWeightsFormat(#[from] serde_json::Error),

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

    #[error("Other circuit error: {0}")]
    Other(String),
}
