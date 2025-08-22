use thiserror::Error;
use crate::circuit_functions::{layers::LayerError, utils::{ArrayConversionError, PatternError, RescaleError, UtilsError}};

#[derive(Debug, Error)]
pub enum CircuitError {
    #[error("Layer failed: {0}")]
    Layer(#[from] LayerError),

    #[error("Circuit failed: {0}")]
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

    #[error("Other circuit error: {0}")]
    Other(String),
}
