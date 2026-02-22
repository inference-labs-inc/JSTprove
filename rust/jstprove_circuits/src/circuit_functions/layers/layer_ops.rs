use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::ArrayD;

use crate::circuit_functions::CircuitError;
use crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry;
use crate::circuit_functions::utils::onnx_model::CircuitParams;
use crate::circuit_functions::utils::onnx_types::ONNXLayer;

pub trait LayerOp<C: Config, Builder: RootAPI<C>> {
    /// Instantiated by each layer op.
    /// Applies the operation relevant operation for that layer
    ///
    /// # Arguments
    /// - `api`: Mutable reference to the circuit builder.
    /// - `input`: Mapping from input names to inputs of
    ///   the layer.
    ///
    /// # Returns
    /// A tuple `(output_names, output_tensor)` containing:
    /// - The ordered list of output names for this layer.
    /// - The computed output tensor as an `ArrayD<Variable>`.
    ///
    /// # Errors
    /// - [`CircuitError`] if tensor operations or constraints fail.
    ///   Or typically if a layer is missing.
    ///   Additionally, any error propogated from underlying computation.
    ///
    fn apply(
        &self,
        api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError>;
    /// Instantiated by each layer op.
    /// Builds a circuit layer from an ONNX definition.
    ///
    /// # Arguments
    /// - `layer`: The ONNX layer specification (op type, attributes, inputs, outputs, etc.).
    /// - `circuit_params`: Global parameters controlling scaling and rescaling.
    /// - `optimization_pattern`: Find any optimization patterns involved.
    /// - `is_rescale`: Flag indicating whether rescaling logic should be applied.
    /// - `index`: The index of this layer in the network.
    /// - `layer_context`: Additional shared state for building layers.
    ///
    /// # Returns
    /// A boxed `LayerOp` implementing the logic for the given ONNX layer.
    ///
    /// # Errors
    /// - [`CircuitError`] if the layer cannot be instantiated in the circuit.
    fn build(
        layer: &ONNXLayer,
        circuit_params: &CircuitParams,
        optimization_pattern: PatternRegistry,
        is_rescale: bool,
        index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError>
    where
        Self: Sized;
}
