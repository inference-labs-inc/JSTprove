use thiserror::Error;

#[derive(Debug, Error)]
pub enum UtilsError {
    #[error("Missing param '{param}' for layer '{layer}'")]
    MissingParam { layer: String, param: String },

    #[error("{layer_name} is missing input: {name}")]
    MissingInput { layer_name: String, name: String },

    #[error("Failed to parse param '{param}' for layer '{layer}': {source}")]
    ParseError {
        layer: String,
        param: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("Cannot convert variable of type '{initial_var_type}' to '{converted_var_type}'")]
    ValueConversionError {
        initial_var_type: String,
        converted_var_type: String,
    },

    #[error("Inputs length mismatch: got {got}, required {required}")]
    InputDataLengthMismatch { got: usize, required: usize },

    #[error("Missing tensor '{tensor}' in weights map")]
    MissingTensor { tensor: String },

    #[error("Bitstring too long: bit index {value} exceeds limit {max}")]
    ValueTooLarge { value: usize, max: u128 },

    #[error("Expected number, but got {value}")]
    InvalidNumber { value: serde_json::Value },

    #[error("Graph error: {0}")]
    GraphPatternError(#[from] PatternError),

    #[error("Array conversion error: {0}")]
    ArrayConversionError(#[from] ArrayConversionError),

    #[error("Rescaling error: {0}")]
    RescaleError(#[from] RescaleError),

    #[error("Build error: {0}")]
    BuildError(#[from] BuildError),
}

#[derive(Debug, thiserror::Error)]
pub enum ArrayConversionError {
    #[error("Invalid array structure: expected {expected}, found {found}")]
    InvalidArrayStructure { expected: String, found: String },

    #[error("Invalid number for target type")]
    InvalidNumber,

    #[error("Shape error: {0}")]
    ShapeError(#[from] ndarray::ShapeError),

    #[error("Invalid axis {axis} for rank {rank}")]
    InvalidAxis { axis: usize, rank: usize },
}

#[derive(Debug, Error)]
pub enum PatternError {
    #[error("Optimization matches have inconsistent patterns: expected {expected}, got {got}")]
    InconsistentPattern { expected: String, got: String },

    #[error("Outputs do not match: expected all in {expected:?}, but got {actual:?}")]
    OutputMismatch {
        expected: Vec<String>,
        actual: Vec<String>,
    },

    #[error("Optimization pattern has no layers")]
    EmptyMatch,

    #[error(
        "Developer error: Empty optimization pattern {pattern} has been attempted to be created. This is not allowed"
    )]
    EmptyPattern { pattern: String },
}

#[derive(Debug, Error)]
pub enum RescaleError {
    #[error("Exponent too large for {type_name} shift: scaling_exponent={exp}")]
    ScalingExponentTooLargeError { exp: usize, type_name: &'static str },

    #[error("Exponent too large for {type_name} shift: shift_exponent={exp}")]
    ShiftExponentTooLargeError { exp: usize, type_name: &'static str },

    #[error("Bit decomposition failed for {var_name} into {n_bits} bits")]
    BitDecompositionError { var_name: String, n_bits: usize },

    #[error("Bit reconstruction failed for {var_name} into {n_bits} bits")]
    BitReconstructionError { var_name: String, n_bits: usize },
}

#[derive(thiserror::Error, Debug)]
pub enum BuildError {
    #[error("Pattern matcher failed: {0}")]
    PatternMatcher(#[from] PatternError),

    #[error("Unsupported layer type: {0}")]
    UnsupportedLayer(String),

    #[error("Layer build failed: {0}")]
    LayerBuild(String),
}
