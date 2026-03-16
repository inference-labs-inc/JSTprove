use thiserror::Error;

use crate::circuit_functions::layers::LayerKind;

#[derive(Debug, Error)]
pub enum LayerError {
    #[error("{layer} is missing input: {name}")]
    MissingInput { layer: LayerKind, name: String },

    #[error("Shape mismatch in {layer} for {var_name}: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        layer: LayerKind,
        expected: Vec<usize>,
        got: Vec<usize>,
        var_name: String,
    },

    #[error("{layer} is missing parameter: {param}")]
    MissingParameter { layer: LayerKind, param: String },

    #[error("{layer} layer '{layer_name}' has an invalid value for {param_name}: {value}")]
    InvalidParameterValue {
        layer: LayerKind,
        layer_name: String,
        param_name: String,
        value: String,
    },

    #[error("Unsupported config in {layer}: {msg}")]
    UnsupportedConfig { layer: LayerKind, msg: String },

    #[error("Invalid shape in {layer}: {msg}")]
    InvalidShape { layer: LayerKind, msg: String },

    #[error("Unknown operator type: {op_type}")]
    UnknownOp { op_type: String },

    #[error("Other error in {layer}: {msg}")]
    Other { layer: LayerKind, msg: String },
}

impl LayerError {
    #[must_use]
    pub fn with_layer(self, new_layer: LayerKind) -> Self {
        match self {
            Self::MissingInput { name, .. } => Self::MissingInput {
                layer: new_layer,
                name,
            },
            Self::ShapeMismatch {
                expected,
                got,
                var_name,
                ..
            } => Self::ShapeMismatch {
                layer: new_layer,
                expected,
                got,
                var_name,
            },
            Self::MissingParameter { param, .. } => Self::MissingParameter {
                layer: new_layer,
                param,
            },
            Self::InvalidParameterValue {
                layer_name,
                param_name,
                value,
                ..
            } => Self::InvalidParameterValue {
                layer: new_layer,
                layer_name,
                param_name,
                value,
            },
            Self::UnsupportedConfig { msg, .. } => Self::UnsupportedConfig {
                layer: new_layer,
                msg,
            },
            Self::InvalidShape { msg, .. } => Self::InvalidShape {
                layer: new_layer,
                msg,
            },
            Self::UnknownOp { op_type } => Self::UnknownOp { op_type },
            Self::Other { msg, .. } => Self::Other {
                layer: new_layer,
                msg,
            },
        }
    }
}
