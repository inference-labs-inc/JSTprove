use std::collections::HashMap;

use crate::circuit_functions::{
    layers::{LayerKind, layer_ops::LayerOp},
    utils::{
        errors::BuildError,
        graph_pattern_matching::PatternRegistry,
        onnx_model::{Architecture, CircuitParams, WANDB, collect_all_shapes},
        onnx_types::ONNXLayer,
    },
};

use expander_compiler::frontend::{Config, RootAPI};

pub struct BuildLayerContext {
    pub w_and_b_map: HashMap<String, ONNXLayer>,
    pub shapes_map: HashMap<String, Vec<usize>>,
    pub n_bits: usize,
    pub two_v: u32,
    pub alpha_two_v: u64,
}

/// Creates the layer context needed for building layers.
/// This should be called once and reused for all layer builds.
pub fn create_layer_context(
    circuit_params: &CircuitParams,
    w_and_b: &WANDB,
) -> Result<BuildLayerContext, BuildError> {
    const N_BITS: usize = 32;
    const V_PLUS_ONE: usize = N_BITS;
    const TWO_V: u32 = 1 << (V_PLUS_ONE - 1);
    let alpha_two_v: u64 = u64::from((1 << circuit_params.scale_exponent) * TWO_V);

    let w_and_b_map: HashMap<String, ONNXLayer> = w_and_b
        .w_and_b
        .clone()
        .into_iter()
        .map(|layer| (layer.name.clone(), layer))
        .collect();

    // This requires architecture, which we need to pass in
    // For now, return error if not available
    let shapes_map = HashMap::new(); // Will be populated by caller if needed

    Ok(BuildLayerContext {
        w_and_b_map,
        shapes_map,
        n_bits: N_BITS,
        two_v: TWO_V,
        alpha_two_v,
    })
}

/// Alternative that takes architecture to build complete context
pub fn create_layer_context_with_shapes(
    circuit_params: &CircuitParams,
    architecture: &Architecture,
    w_and_b: &WANDB,
) -> Result<BuildLayerContext, BuildError> {
    const N_BITS: usize = 32;
    const V_PLUS_ONE: usize = N_BITS;
    const TWO_V: u32 = 1 << (V_PLUS_ONE - 1);
    let alpha_two_v: u64 = u64::from((1 << circuit_params.scale_exponent) * TWO_V);

    let w_and_b_map: HashMap<String, ONNXLayer> = w_and_b
        .w_and_b
        .clone()
        .into_iter()
        .map(|layer| (layer.name.clone(), layer))
        .collect();

    let inputs = &circuit_params.inputs;
    let shapes_map: HashMap<String, Vec<usize>> =
        collect_all_shapes(&architecture.architecture, inputs);

    Ok(BuildLayerContext {
        w_and_b_map,
        shapes_map,
        n_bits: N_BITS,
        two_v: TWO_V,
        alpha_two_v,
    })
}

/// Builds a single layer on-demand.
///
/// This function is designed to be called in a streaming fashion where each
/// layer is built, applied, and then dropped, avoiding storing all layers in memory.
///
/// # Arguments
/// - `layer`: The layer definition to build
/// - `circuit_params`: Global circuit parameters
/// - `optimization_pattern`: Pattern optimization to apply (if any)
/// - `layer_index`: Index of this layer in the architecture
/// - `layer_context`: Pre-built context containing weights, shapes, etc.
///
/// # Returns
/// A boxed `LayerOp` trait object for the built layer
pub fn build_single_layer<C: Config, Builder: RootAPI<C>>(
    layer: &ONNXLayer,
    circuit_params: &CircuitParams,
    optimization_pattern: PatternRegistry,
    layer_index: usize,
    layer_context: &BuildLayerContext,
) -> Result<Box<dyn LayerOp<C, Builder>>, BuildError> {
    let is_rescale = circuit_params
        .rescale_config
        .get(&layer.name)
        .unwrap_or(&true);

    let layer_kind = LayerKind::try_from(layer.op_type.as_str())
        .map_err(|_| BuildError::UnsupportedLayer(layer.op_type.clone()))?;

    let builder = layer_kind.builder::<C, Builder>();

    builder(
        layer,
        circuit_params,
        optimization_pattern,
        *is_rescale,
        layer_index,
        layer_context,
    )
    .map_err(|e| BuildError::LayerBuild(format!("Failed to build {}: {}", layer.name, e)))
}

/// Legacy function kept for backward compatibility.
/// Consider migrating to the streaming approach for better memory efficiency.
///
/// Builds all layers upfront and returns them as a vector.
/// This can use significant memory for large models.
#[deprecated(note = "Consider using build_single_layer for better memory efficiency")]
pub fn build_layers<C: Config, Builder: RootAPI<C>>(
    circuit_params: &CircuitParams,
    architecture: &Architecture,
    w_and_b: &WANDB,
) -> Result<Vec<Box<dyn LayerOp<C, Builder>>>, BuildError> {
    use crate::circuit_functions::utils::graph_pattern_matching::{
        PatternMatcher, optimization_skip_layers,
    };

    let mut layers: Vec<Box<dyn LayerOp<C, Builder>>> = vec![];
    let layer_context = create_layer_context_with_shapes(circuit_params, architecture, w_and_b)?;

    let mut skip_next_layer: HashMap<String, bool> = HashMap::new();
    let matcher = PatternMatcher::new();
    let opt_patterns_by_layername = matcher.run(&architecture.architecture)?;

    for (i, original_layer) in architecture.architecture.iter().enumerate() {
        let mut layer = original_layer.clone();
        if *skip_next_layer.get(&layer.name).unwrap_or(&false) {
            continue;
        }
        let outputs = layer.outputs.clone();

        let optimization_pattern_match = opt_patterns_by_layername.get(&layer.name);
        let (optimization_pattern, outputs, layers_to_skip) =
            match optimization_skip_layers(optimization_pattern_match, &outputs).unwrap_or(None) {
                Some(opt) => opt,
                None => (PatternRegistry::None, outputs.clone(), vec![]),
            };

        for item in layers_to_skip {
            skip_next_layer.insert(item.to_string(), true);
        }
        layer.outputs = outputs;

        let built = build_single_layer::<C, Builder>(
            &layer,
            circuit_params,
            optimization_pattern,
            i,
            &layer_context,
        )?;
        layers.push(built);
    }
    Ok(layers)
}
