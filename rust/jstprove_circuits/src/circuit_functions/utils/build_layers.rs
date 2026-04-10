use std::collections::HashMap;

use ndarray::ArrayD;

use crate::circuit_functions::{
    layers::{LayerKind, layer_ops::LayerOp},
    utils::{
        errors::BuildError,
        graph_pattern_matching::{PatternMatcher, PatternRegistry, optimization_skip_layers},
        onnx_model::{Architecture, CircuitParams, WANDB, collect_all_shapes, get_w_or_b},
        onnx_types::ONNXLayer,
    },
};

use expander_compiler::frontend::{Config, RootAPI};

const DEFAULT_N_BITS_BN254: usize = 64;
const DEFAULT_N_BITS_GOLDILOCKS: usize = 31;

#[must_use]
pub fn default_n_bits_for_config<C: Config>() -> usize {
    match C::CONFIG_ID {
        4 | 6 | 8 => DEFAULT_N_BITS_GOLDILOCKS,
        _ => DEFAULT_N_BITS_BN254,
    }
}

pub struct BuiltLayer<C: Config, Builder: RootAPI<C>> {
    pub name: String,
    pub op_type: String,
    pub arch_index: usize,
    pub layer: Box<dyn LayerOp<C, Builder>>,
}

pub struct BuildLayerContext<'a> {
    pub w_and_b_map: &'a HashMap<String, &'a ONNXLayer>,
    pub shapes_map: &'a HashMap<String, Vec<usize>>,
    pub n_bits_config: &'a HashMap<String, usize>,
    pub default_n_bits: usize,
    pub weights_as_inputs: bool,
    pub constants_map: &'a HashMap<String, ArrayD<i64>>,
}

impl BuildLayerContext<'_> {
    #[must_use]
    pub fn n_bits_for(&self, layer_name: &str) -> usize {
        self.n_bits_config
            .get(layer_name)
            .copied()
            .unwrap_or(self.default_n_bits)
    }

    #[must_use]
    pub fn get_constant(&self, name: &str) -> Option<&ArrayD<i64>> {
        self.constants_map.get(name)
    }
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
) -> Result<Vec<BuiltLayer<C, Builder>>, BuildError> {
    let mut layers: Vec<BuiltLayer<C, Builder>> = vec![];

    let w_and_b_map: HashMap<String, &ONNXLayer> = w_and_b
        .w_and_b
        .iter()
        .map(|layer| (layer.name.clone(), layer))
        .collect();

    let mut skip_next_layer: HashMap<String, bool> = HashMap::new();

    let inputs = &circuit_params.inputs;

    let shapes_map: HashMap<String, Vec<usize>> =
        collect_all_shapes(&architecture.architecture, inputs);

    let constants_map = extract_architecture_constants(&architecture.architecture, &w_and_b_map)?;

    let layer_context = BuildLayerContext {
        w_and_b_map: &w_and_b_map,
        shapes_map: &shapes_map,
        n_bits_config: &circuit_params.n_bits_config,
        default_n_bits: default_n_bits_for_config::<C>(),
        weights_as_inputs: circuit_params.weights_as_inputs,
        constants_map: &constants_map,
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
        layers.push(BuiltLayer {
            name: layer.name.clone(),
            op_type: layer.op_type.clone(),
            arch_index: i,
            layer: built,
        });
    }
    Ok(layers)
}

fn extract_architecture_constants(
    architecture: &[ONNXLayer],
    w_and_b_map: &HashMap<String, &ONNXLayer>,
) -> Result<HashMap<String, ArrayD<i64>>, BuildError> {
    let mut constants_map = HashMap::new();
    for layer in architecture {
        if layer.op_type != "Constant" || layer.tensor.is_none() {
            continue;
        }
        for out_name in &layer.outputs {
            if w_and_b_map.contains_key(out_name) {
                continue;
            }
            let tmp_map: HashMap<String, &ONNXLayer> =
                [(out_name.clone(), layer)].into_iter().collect();
            let arr = get_w_or_b::<i64, _>(&tmp_map, out_name).map_err(|e| {
                BuildError::LayerBuild(format!(
                    "Failed to extract Constant tensor '{out_name}': {e}"
                ))
            })?;
            constants_map.insert(out_name.clone(), arr);
        }
    }
    Ok(constants_map)
}
