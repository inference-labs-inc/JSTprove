use std::path::Path;

use anyhow::Result;

use crate::onnx::graph::LayerGraph;
use crate::onnx::parser;
use crate::onnx::quantizer::{self, QuantizedModel, ScaleConfig};

pub fn run(model_path: &Path, output_path: &Path, compress: bool) -> Result<()> {
    tracing::info!("parsing ONNX model: {}", model_path.display());
    let parsed = parser::parse_onnx(model_path)?;

    tracing::info!("building layer graph ({} nodes)", parsed.nodes.len());
    let graph = LayerGraph::from_parsed(&parsed)?;

    tracing::info!("quantizing model (scale=2^18)");
    let config = ScaleConfig::default();
    let quantized = quantizer::quantize_model(graph, &config)?;

    tracing::info!(
        "quantized {} layers, {} n_bits entries",
        quantized.graph.layers.len(),
        quantized.n_bits_config.len()
    );

    let size = jstprove_io::serialize_to_file(&quantized, output_path, compress)?;
    tracing::info!(
        "model written to {} ({} bytes)",
        output_path.display(),
        size
    );
    Ok(())
}

pub fn load_model(path: &Path) -> Result<QuantizedModel> {
    jstprove_io::deserialize_from_file(path)
}
