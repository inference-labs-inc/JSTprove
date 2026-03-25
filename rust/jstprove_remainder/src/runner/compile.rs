use std::path::Path;

use anyhow::Result;

use crate::cli::{self, StepPrinter};
use crate::onnx::graph::LayerGraph;
use crate::onnx::parser;
use crate::onnx::quantizer::{self, QuantizedModel, ScaleConfig};
use crate::onnx::shape_inference;

pub fn run(model_path: &Path, output_path: &Path, compress: bool) -> Result<()> {
    let mut steps = StepPrinter::new(4);

    steps.step("Parsing ONNX model");
    steps.detail(&format!("source: {}", model_path.display()));
    let parsed = parser::parse_onnx(model_path)?;
    steps.detail(&format!("{} graph nodes found", parsed.nodes.len()));

    steps.step("Building layer graph");
    let graph = LayerGraph::from_parsed(&parsed)?;
    steps.detail(&format!("{} layers constructed", graph.layers.len()));

    steps.step("Quantizing model");
    let config = ScaleConfig::default();
    steps.detail(&format!(
        "scale = {}^{} (alpha = {})",
        config.base, config.exponent, config.alpha
    ));
    let mut quantized = quantizer::quantize_model(graph, &config)?;

    let shapes = shape_inference::infer_all_shapes(&parsed, &quantized.graph)?;
    for layer in &mut quantized.graph.layers {
        if let Some(out_name) = layer.outputs.first() {
            if let Some(shape) = shapes.get(out_name) {
                layer.output_shape = shape.clone();
            }
        }
    }
    steps.detail(&format!(
        "{} layers, {} n_bits entries",
        quantized.graph.layers.len(),
        quantized.n_bits_config.len()
    ));

    steps.step("Writing compiled model");
    let size = jstprove_io::serialize_to_file(&quantized, output_path, compress)?;
    steps.detail(&format!(
        "{} -> {}{}",
        output_path.display(),
        cli::fmt_bytes(size as u64),
        if compress { " (compressed)" } else { "" }
    ));

    steps.finish_ok("Model compiled");
    Ok(())
}

pub fn load_model(path: &Path) -> Result<QuantizedModel> {
    jstprove_io::deserialize_from_file(path)
}
