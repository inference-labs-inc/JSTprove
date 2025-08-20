use std::collections::HashMap;

use expander_compiler::frontend::*;
use ndarray::ArrayD;

use crate::circuit_functions::utils::graph_pattern_matching::GraphPattern;
use crate::circuit_functions::utils::onnx_types::ONNXLayer;
use crate::circuit_functions::utils::onnx_model::CircuitParams;

pub trait LayerOp<C: Config, Builder: RootAPI<C>> {
    fn apply(&self, api: &mut Builder, input: HashMap<String,ArrayD<Variable>>)
        -> Result<(Vec<String>,ArrayD<Variable>), String>;
fn build(
        layer: &ONNXLayer,
        circuit_params: &CircuitParams,
        optimization_pattern: GraphPattern,
        is_rescale: bool,
        index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, Error>
    where
        Self: Sized;
}
