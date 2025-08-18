use thiserror::Error;

#[derive(Debug, Error)]
pub enum UtilsError {
    #[error("Missing param '{param}' for layer '{layer}'")]
    MissingParam { layer: String, param: String },

    #[error("Failed to parse param '{param}' for layer '{layer}': {source}")]
    ParseError {
        layer: String,
        param: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("Missing tensor '{tensor}' in weights map")]
    MissingTensor { tensor: String },

    #[error("Bitstring too long: bit index {value} exceeds limit {max}")]
    ValueTooLarge { value: usize, max: u128 },
}


#[derive(Debug, Error)]
pub enum PatternError {
    #[error("Optimization matches have inconsistent patterns: expected {expected}, got {got}")]
    InconsistentPattern { expected: String, got: String },

    #[error("Outputs do not match: expected all in {expected:?}, but got {actual:?}")]
    OutputMismatch { expected: Vec<String>, actual: Vec<String> },

    #[error("Optimization pattern has no layers")]
    EmptyMatch,
}