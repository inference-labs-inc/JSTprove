use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context as _, Result};
use ndarray::ArrayD;

use expander_compiler::frontend::{API, Config, GoldilocksConfig, HintRegistry, Variable};
use expander_compiler::zkcuda::context::{Context, DeviceMemoryHandle};
use expander_compiler::zkcuda::kernel::{IOVecSpec, KernelPrimitive, compile_with_spec_and_shapes};
use expander_compiler::zkcuda::proving_system::{Expander, ProvingSystem};
use expander_compiler::zkcuda::shape::Reshape;

use circuit_std_rs::logup::{query_count_by_key_hint, query_count_hint, rangeproof_hint};

use crate::circuit_functions::gadgets::LogupRangeCheckContext;
use crate::circuit_functions::layers::LayerKind;
use crate::circuit_functions::utils::build_layers::{BuildLayerContext, default_n_bits_for_config};
use crate::circuit_functions::utils::graph_pattern_matching::{
    PatternMatcher, PatternRegistry, optimization_skip_layers,
};
use crate::circuit_functions::utils::onnx_model::{
    Architecture, CircuitParams, WANDB, collect_all_shapes,
};
use crate::circuit_functions::utils::onnx_types::ONNXLayer;
use crate::circuit_functions::utils::tensor_ops::convert_val_to_field_element;
use crate::expander_metadata;

use jstprove_onnx::quantizer::N_BITS_GOLDILOCKS;

use expander_compiler::circuit::config::CircuitField;

fn cap_n_bits_config(params: &mut CircuitParams, n_bits_field: u32) {
    let max_n_bits = 63_usize;
    for nb in params.n_bits_config.values_mut() {
        if *nb > max_n_bits {
            *nb = n_bits_field as usize;
        }
    }
}

struct LayerDescriptor {
    name: String,
    output_activation_len: usize,
    weight_entries: Vec<(String, usize)>,
}

struct CompiledPipeline<C: Config> {
    kernels: Vec<KernelPrimitive<C>>,
    layer_descriptors: Vec<LayerDescriptor>,
    input_len: usize,
}

pub struct PipelineResult {
    pub compile_ms: f64,
    pub witness_ms: f64,
    pub prove_ms: f64,
    pub verify_ms: f64,
    pub num_kernels: usize,
    pub verified: bool,
}

/// # Errors
/// Returns an error if ONNX parsing, kernel compilation, or proving fails.
pub fn run_pipeline_goldilocks(onnx_path: &Path, input_data: &[f64]) -> Result<PipelineResult> {
    run_pipeline::<GoldilocksConfig>(onnx_path, input_data, N_BITS_GOLDILOCKS)
}

fn run_pipeline<C: Config>(
    onnx_path: &Path,
    input_data: &[f64],
    n_bits: u32,
) -> Result<PipelineResult>
where
    Expander<C>: ProvingSystem<C>,
{
    let metadata = expander_metadata::generate_from_onnx_for_field(onnx_path, n_bits, None)
        .context("generating ONNX metadata")?;

    let mut params = metadata.circuit_params.clone();
    let architecture = &metadata.architecture;
    let wandb = &metadata.wandb;

    cap_n_bits_config(&mut params, n_bits);

    let t = Instant::now();
    let pipeline = compile_kernels::<C>(&params, architecture, wandb)?;
    let compile_ms = t.elapsed().as_secs_f64() * 1000.0;

    let num_kernels = pipeline.kernels.len();

    let t = Instant::now();
    let mut hint_registry = HintRegistry::<CircuitField<C>>::new();
    hint_registry.register("myhint.querycounthint", query_count_hint);
    hint_registry.register("myhint.querycountbykeyhint", query_count_by_key_hint);
    hint_registry.register("myhint.rangeproofhint", rangeproof_hint);
    let mut ctx: Context<C, _> = Context::new(hint_registry);

    anyhow::ensure!(
        input_data.len() == pipeline.input_len,
        "input_data length ({}) does not match model input length ({})",
        input_data.len(),
        pipeline.input_len
    );
    let input_vals: Vec<CircuitField<C>> = input_data
        .iter()
        .map(|&v| {
            #[allow(clippy::cast_possible_truncation)]
            let i = v as i64;
            convert_val_to_field_element::<C>(i)
        })
        .collect();
    let mut activations: DeviceMemoryHandle = ctx.copy_to_device(&input_vals);
    activations = activations.reshape(&[1, pipeline.input_len]);

    for (i, (kernel, desc)) in pipeline
        .kernels
        .iter()
        .zip(pipeline.layer_descriptors.iter())
        .enumerate()
    {
        let mut io_handles: Vec<_> = vec![activations.clone()];

        for (_wname, wlen) in &desc.weight_entries {
            let wvals = vec![CircuitField::<C>::default(); *wlen];
            let whandle = ctx.copy_to_device(&wvals);
            io_handles.push(whandle.reshape(&[1, *wlen]));
        }

        let mut out_handle = None;
        io_handles.push(out_handle.clone());

        let io_slice: &mut [_] = &mut io_handles;
        ctx.call_kernel(kernel, 1, io_slice)
            .map_err(|e| anyhow::anyhow!("kernel {i} ({}) call failed: {e}", desc.name))?;

        out_handle = io_slice.last().cloned().unwrap_or(None);
        anyhow::ensure!(out_handle.is_some(), "kernel {i} produced no output");
        activations = out_handle.reshape(&[1, desc.output_activation_len]);

        tracing::info!(kernel = i, name = %desc.name, "kernel executed");
    }

    let computation_graph = ctx
        .compile_computation_graph()
        .map_err(|e| anyhow::anyhow!("compile_computation_graph: {e}"))?;
    ctx.solve_witness()
        .map_err(|e| anyhow::anyhow!("solve_witness: {e}"))?;
    let witness_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let (prover_setup, verifier_setup) =
        <Expander<C> as ProvingSystem<C>>::setup(&computation_graph);
    let proof = <Expander<C> as ProvingSystem<C>>::prove(
        &prover_setup,
        &computation_graph,
        ctx.export_device_memories(),
    );
    let prove_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let verified =
        <Expander<C> as ProvingSystem<C>>::verify(&verifier_setup, &computation_graph, &proof);
    let verify_ms = t.elapsed().as_secs_f64() * 1000.0;

    Ok(PipelineResult {
        compile_ms,
        witness_ms,
        prove_ms,
        verify_ms,
        num_kernels,
        verified,
    })
}

#[allow(clippy::too_many_lines)]
fn compile_kernels<C: Config>(
    params: &CircuitParams,
    architecture: &Architecture,
    wandb: &WANDB,
) -> Result<CompiledPipeline<C>> {
    let w_and_b_map: HashMap<String, &ONNXLayer> = wandb
        .w_and_b
        .iter()
        .map(|layer| (layer.name.clone(), layer))
        .collect();

    let inputs = &params.inputs;
    let shapes_map: HashMap<String, Vec<usize>> =
        collect_all_shapes(&architecture.architecture, inputs);

    let layer_context = BuildLayerContext {
        w_and_b_map: &w_and_b_map,
        shapes_map: &shapes_map,
        n_bits_config: &params.n_bits_config,
        default_n_bits: default_n_bits_for_config::<C>(),
        weights_as_inputs: false,
    };

    let matcher = PatternMatcher::new();
    let opt_patterns_by_layername = matcher.run(&architecture.architecture)?;

    let mut skip_next_layer: HashMap<String, bool> = HashMap::new();
    let mut kernels: Vec<KernelPrimitive<C>> = Vec::new();
    let mut descriptors: Vec<LayerDescriptor> = Vec::new();

    let input_len: usize = params
        .inputs
        .iter()
        .map(|io| io.shape.iter().product::<usize>())
        .sum();
    let mut prev_output_len = input_len;

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

        let is_rescale = *params.rescale_config.get(&layer.name).unwrap_or(&true);

        let layer_kind = LayerKind::try_from(layer.op_type.as_str())
            .map_err(|_| anyhow::anyhow!("unsupported layer: {}", layer.op_type))?;

        let act_in_len = prev_output_len;

        let mut weight_entries: Vec<(String, usize)> = Vec::new();
        for name in layer.inputs.iter().skip(1) {
            let wb = w_and_b_map.get(name.as_str()).ok_or_else(|| {
                anyhow::anyhow!(
                    "layer '{}' references input '{}' not found in initializer map",
                    layer.name,
                    name
                )
            })?;
            let len: usize = wb
                .shape
                .get(name.as_str())
                .map_or(1, |s| s.iter().product());
            weight_entries.push((name.clone(), len));
        }

        let act_out_shape: Vec<usize> = layer
            .outputs
            .first()
            .and_then(|oname| shapes_map.get(oname))
            .cloned()
            .unwrap_or_else(|| vec![act_in_len]);
        let act_out_len: usize = act_out_shape.iter().product();

        let mut io_specs = vec![IOVecSpec {
            len: act_in_len,
            is_input: true,
            is_output: false,
        }];
        let mut io_shapes: Vec<Vec<usize>> = vec![vec![act_in_len]];

        for (_wname, wlen) in &weight_entries {
            io_specs.push(IOVecSpec {
                len: *wlen,
                is_input: true,
                is_output: false,
            });
            io_shapes.push(vec![*wlen]);
        }

        io_specs.push(IOVecSpec {
            len: act_out_len,
            is_input: false,
            is_output: true,
        });
        io_shapes.push(vec![act_out_len]);

        let layer_clone = layer.clone();
        let params_clone = params.clone();
        let layer_name = layer.name.clone();

        let builder = layer_kind.builder::<C, API<C>>();
        let built_layer = builder(
            &layer_clone,
            &params_clone,
            optimization_pattern,
            is_rescale,
            i,
            &layer_context,
        )
        .map_err(|e| anyhow::anyhow!("building layer {layer_name}: {e}"))?;

        let input_activation_name = layer
            .inputs
            .first()
            .cloned()
            .unwrap_or_else(|| "input".into());

        let chunk_bits = params
            .logup_chunk_bits
            .unwrap_or(crate::circuit_functions::gadgets::DEFAULT_LOGUP_CHUNK_BITS);

        let act_in_tensor_shape: Vec<usize> = shapes_map
            .get(&input_activation_name)
            .cloned()
            .unwrap_or_else(|| vec![act_in_len]);

        let weight_shapes: Vec<(String, Vec<usize>)> = weight_entries
            .iter()
            .map(|(wname, wlen)| {
                let shape = w_and_b_map
                    .get(wname.as_str())
                    .and_then(|wb| wb.shape.get(wname.as_str()))
                    .cloned()
                    .unwrap_or_else(|| vec![*wlen]);
                (wname.clone(), shape)
            })
            .collect();

        let kernel = compile_with_spec_and_shapes::<C, _>(
            |api: &mut API<C>, io_vars: &mut Vec<Vec<Variable>>| {
                let act_in = &io_vars[0];
                let act_in_arr =
                    ArrayD::from_shape_vec(ndarray::IxDyn(&act_in_tensor_shape), act_in.clone())
                        .unwrap_or_else(|e| panic!(
                            "reshape input activations for layer '{}' with shape {:?} and {} elements: {e}",
                            layer_name, act_in_tensor_shape, act_in.len()
                        ));

                let mut tensor_map: HashMap<String, ArrayD<Variable>> = HashMap::new();
                tensor_map.insert(input_activation_name.clone(), act_in_arr);

                for (idx, (wname, wshape)) in weight_shapes.iter().enumerate() {
                    let wvars = &io_vars[1 + idx];
                    let warr = ArrayD::from_shape_vec(ndarray::IxDyn(wshape), wvars.clone())
                        .unwrap_or_else(|e| panic!(
                            "reshape weights '{}' for layer '{}' with shape {:?} and {} elements: {e}",
                            wname, layer_name, wshape, wvars.len()
                        ));
                    tensor_map.insert(wname.clone(), warr);
                }

                let mut logup_ctx = LogupRangeCheckContext::new(chunk_bits);
                logup_ctx.init::<C, API<C>>(api);

                let (_, output_arr) = built_layer
                    .apply(api, &mut logup_ctx, &tensor_map)
                    .unwrap_or_else(|e| panic!(
                        "apply failed for layer '{layer_name}': {e}"
                    ));

                logup_ctx.finalize::<C, API<C>>(api);

                let out_flat: Vec<Variable> = output_arr.iter().copied().collect();
                let out_idx = io_vars.len() - 1;
                io_vars[out_idx] = out_flat;
            },
            &io_specs,
            &io_shapes,
        )
        .map_err(|e| anyhow::anyhow!("compile kernel for layer {layer_name}: {e}"))?;

        tracing::info!(
            layer = i,
            name = %layer_name,
            act_in = act_in_len,
            act_out = act_out_len,
            weights = weight_entries.len(),
            "compiled kernel"
        );

        prev_output_len = act_out_len;

        descriptors.push(LayerDescriptor {
            name: layer_name,
            output_activation_len: act_out_len,
            weight_entries,
        });
        kernels.push(kernel);
    }

    Ok(CompiledPipeline {
        kernels,
        layer_descriptors: descriptors,
        input_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn lenet_zkcuda_pipeline() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../jstprove_remainder/models/lenet.onnx");
        assert!(model_path.exists(), "lenet.onnx not found");

        let num_act: usize = 3072;
        let activations: Vec<f64> = (0..num_act).map(|i| i as f64 / num_act as f64).collect();

        let result = run_pipeline_goldilocks(&model_path, &activations).unwrap();

        println!("kernels:  {}", result.num_kernels);
        println!("compile:  {:.2}s", result.compile_ms / 1000.0);
        println!("witness:  {:.1}ms", result.witness_ms);
        println!("prove:    {:.1}ms", result.prove_ms);
        println!("verify:   {:.1}ms", result.verify_ms);
        println!("verified: {}", result.verified);
        assert!(result.verified);
    }

    #[test]
    #[ignore]
    fn vgg16_zkcuda_pipeline() {
        let model_path = Path::new("/tmp/vgg16_dsperse/model.onnx");
        if !model_path.exists() {
            eprintln!("VGG-16 model not found at /tmp/vgg16_dsperse/model.onnx, skipping");
            return;
        }

        let num_act: usize = 3072;
        let activations: Vec<f64> = (0..num_act).map(|i| i as f64 / num_act as f64).collect();

        let result = run_pipeline_goldilocks(model_path, &activations).unwrap();

        println!("kernels:  {}", result.num_kernels);
        println!("compile:  {:.2}s", result.compile_ms / 1000.0);
        println!("witness:  {:.1}ms", result.witness_ms);
        println!("prove:    {:.1}ms", result.prove_ms);
        println!("verify:   {:.1}ms", result.verify_ms);
        println!("verified: {}", result.verified);
        assert!(result.verified);
    }
}
