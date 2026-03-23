// ONNX `ConvTranspose` layer for ZK circuits.
//
// # ZK approach
// ConvTranspose (a.k.a. transposed convolution or fractionally-strided convolution)
// computes the gradient of Conv with respect to its input. It is equivalent to
// inserting (stride-1) zeros between input elements, then applying a regular
// convolution with the flipped kernel.
//
// In the ZK circuit this would require the same approach as Conv (Freivalds-based
// matrix-vector multiplication with rescaling), but the upsampling step significantly
// increases circuit complexity.
//
// # Current status
// ConvTranspose is not yet supported in the Expander backend. It is rejected at
// build time. Shape inference and quantizer bounds are registered so that models
// containing ConvTranspose can be parsed and analysed; only circuit compilation
// will fail.

use std::collections::HashMap;

use ndarray::ArrayD;

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
};

// -------- Struct --------

pub struct ConvTransposeLayer;

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConvTransposeLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        _input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        Err(LayerError::Other {
            layer: LayerKind::ConvTranspose,
            msg: "ConvTranspose is not yet supported in the Expander backend.".to_string(),
        }
        .into())
    }

    fn build(
        _layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        _layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        Err(LayerError::Other {
            layer: LayerKind::ConvTranspose,
            msg: "ConvTranspose is not yet supported in the Expander backend.".to_string(),
        }
        .into())
    }
}
