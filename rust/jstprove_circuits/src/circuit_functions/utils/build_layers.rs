use std::collections::HashMap;

use crate::circuit_functions::{layers::{layer_ops::LayerOp,LayerKind}, utils::{errors::BuildError, graph_pattern_matching::{optimization_skip_layers, GraphPattern, PatternMatcher}, onnx_model::{collect_all_shapes, Architecture, CircuitParams, WANDB}, onnx_types::ONNXLayer}};

use expander_compiler::frontend::*;


type BoxedDynLayer<C, B> = Box<dyn LayerOp<C, B>>;
pub struct BuildLayerContext{
    pub w_and_b_map: HashMap<String, ONNXLayer>,
    pub shapes_map: HashMap<String, Vec<usize>>,
    pub n_bits: usize,
    pub two_v: u32,
    pub alpha_two_v: u64,
}

pub fn build_layers<C: Config, Builder: RootAPI<C>>(
    circuit_params: &CircuitParams,
    arcitecture: &Architecture,
    w_and_b: &WANDB
) -> Result<Vec<Box<dyn LayerOp<C, Builder>>>, BuildError> {
    let mut layers: Vec<BoxedDynLayer<C, Builder>> = vec![];
    const N_BITS: usize = 32;
    const V_PLUS_ONE: usize = N_BITS;
    const TWO_V: u32 = 1 << (V_PLUS_ONE - 1);
    let alpha_two_v: u64 = ((1 << circuit_params.scaling) * TWO_V) as u64;

    /*
    TODO: Inject weights + bias data with external functions instead of regular assignment in function.
     */

    let w_and_b_map: HashMap<String, ONNXLayer> = w_and_b.w_and_b.clone()
        .into_iter()
        .map(|layer| (layer.name.clone(), layer))
        .collect();

    

    let mut skip_next_layer: HashMap<String, bool>  = HashMap::new();

    let inputs = &arcitecture.inputs;

    // TODO havent figured out how but this can maybe go in build layers?
    let shapes_map: HashMap<String, Vec<usize>> = collect_all_shapes(&arcitecture.architecture, inputs);

    let layer_context = BuildLayerContext{
        w_and_b_map: w_and_b_map.clone(),
        shapes_map: shapes_map.clone(),
        n_bits: N_BITS,
        two_v: TWO_V,
        alpha_two_v: alpha_two_v
    };

    let matcher = PatternMatcher::new();
    let opt_patterns_by_layername = matcher.run(&arcitecture.architecture)?;
    
    for (i, original_layer) in arcitecture.architecture.iter().enumerate() {
        /*
            Track layer combo optimizations
         */
        let mut layer = original_layer.clone();
        if *skip_next_layer.get(&layer.name).unwrap_or(&false){
            continue
        }
        let outputs = layer.outputs.to_vec();

        let optimization_pattern_match = opt_patterns_by_layername.get(&layer.name);
        let (optimization_pattern, outputs, layers_to_skip) =  match optimization_skip_layers(optimization_pattern_match, outputs.clone()).unwrap_or(None) {
            Some(opt) => opt,
            None => (GraphPattern::default(), outputs.clone(), vec![])
        };
        // Save layers to skip
        layers_to_skip.into_iter()
            .for_each(|item| {
                skip_next_layer.insert(item.to_string(), true);
            });

        layer.outputs = outputs;
        /*
            End tracking layer combo optimizations
         */
        let is_rescale = match  circuit_params.rescale_config.get(&layer.name){
            Some(config) => config,
            None => &true
        };

        let layer_kind = LayerKind::try_from(layer.op_type.as_str())
            .map_err(|_| BuildError::UnsupportedLayer(layer.op_type.clone()))?;

        let builder = layer_kind.builder::<C, Builder>();

        let built = builder(&layer, circuit_params, optimization_pattern, *is_rescale, i, &layer_context)
            .map_err(|e| BuildError::LayerBuild(format!("Failed to build {}: {}", layer.name, e)))?;
        layers.push(built);
        // tracing::info!("Layer added: {}", layer.op_type);
    }
    Ok(layers)
}
