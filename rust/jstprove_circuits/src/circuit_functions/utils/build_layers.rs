use std::collections::HashMap;

use crate::circuit_functions::{
    layers::{LayerKind, layer_ops::LayerOp},
    utils::{
        errors::BuildError,
        graph_pattern_matching::{PatternMatcher, PatternRegistry, optimization_skip_layers},
        onnx_model::{Architecture, CircuitParams, WANDB, collect_all_shapes},
        onnx_types::ONNXLayer,
    },
};

use expander_compiler::frontend::{Config, RootAPI};

type BoxedDynLayer<C, B> = Box<dyn LayerOp<C, B>>;
pub struct BuildLayerContext {
    pub w_and_b_map: HashMap<String, ONNXLayer>,
    pub shapes_map: HashMap<String, Vec<usize>>,
    pub n_bits: usize,
    pub two_v: u32,
    pub alpha_two_v: u64,
    pub weights_as_inputs: bool,
}

/// Builds a sequence of circuit layers from an architecture and parameters.
///
/// Iterates over the architecture definition, applies pattern-based optimizations
/// to skip redundant layers, and constructs each remaining layer using its
/// corresponding builder.
///
/// # Arguments
/// - `circuit_params`: Global parameters for circuit scaling and rescaling.
/// - `arcitecture`: Network architecture definition, including inputs and layers.
/// - `w_and_b`: Weights and biases for each layer.
///
/// # Returns
/// A vector of boxed `LayerOp` trait objects representing the built layers.
///
/// # Errors
/// - [`BuildError::UnsupportedLayer`] if a layer’s `op_type` is not recognized.
/// - [`BuildError::LayerBuild`] if construction of an individual layer fails.
/// - Propagates any error from the pattern matcher used for optimizations.
pub fn build_layers<C: Config, Builder: RootAPI<C>>(
    circuit_params: &CircuitParams,
    architecture: &Architecture,
    w_and_b: &WANDB,
) -> Result<Vec<Box<dyn LayerOp<C, Builder>>>, BuildError> {
    const N_BITS: usize = 32;
    const V_PLUS_ONE: usize = N_BITS;
    const TWO_V: u32 = 1 << (V_PLUS_ONE - 1);
    let mut layers: Vec<BoxedDynLayer<C, Builder>> = vec![];
    let alpha_two_v: u64 = u64::from((1 << circuit_params.scale_exponent) * TWO_V);

    /*
    TODO: Inject weights + bias data with external functions instead of regular assignment in function.
     */

    let w_and_b_map: HashMap<String, ONNXLayer> = w_and_b
        .w_and_b
        .clone()
        .into_iter()
        .map(|layer| (layer.name.clone(), layer))
        .collect();

    let mut skip_next_layer: HashMap<String, bool> = HashMap::new();

    let inputs = &circuit_params.inputs;

    // TODO havent figured out how but this can maybe go in build layers?
    let shapes_map: HashMap<String, Vec<usize>> =
        collect_all_shapes(&architecture.architecture, inputs);

    let layer_context = BuildLayerContext {
        w_and_b_map: w_and_b_map.clone(),
        shapes_map: shapes_map.clone(),
        n_bits: N_BITS,
        two_v: TWO_V,
        alpha_two_v,
        weights_as_inputs: circuit_params.weights_as_inputs,
    };

    let matcher = PatternMatcher::new();
    let opt_patterns_by_layername = matcher.run(&architecture.architecture)?;

    for (i, original_layer) in architecture.architecture.iter().enumerate() {
        /*
           Track layer combo optimizations
        */
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
        // Save layers to skip
        for item in layers_to_skip {
            skip_next_layer.insert(item.to_string(), true);
        }

        layer.outputs = outputs;
        /*
           End tracking layer combo optimizations
        */
        let is_rescale = circuit_params
            .rescale_config
            .get(&layer.name)
            .unwrap_or(&true);

        let layer_kind = LayerKind::try_from(layer.op_type.as_str())
            .map_err(|_| BuildError::UnsupportedLayer(layer.op_type.clone()))?;

        let builder = layer_kind.builder::<C, Builder>();

        let built = builder(
            &layer,
            circuit_params,
            optimization_pattern,
            *is_rescale,
            i,
            &layer_context,
        )
        .map_err(|e| BuildError::LayerBuild(format!("Failed to build {}: {}", layer.name, e)))?;
        layers.push(built);
        // tracing::info!("Layer added: {}", layer.op_type);
    }
    Ok(layers)
}
