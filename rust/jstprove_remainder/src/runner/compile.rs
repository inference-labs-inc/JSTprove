use std::path::Path;

use anyhow::Result;

use crate::onnx::parser;
use crate::onnx::graph::LayerGraph;
use crate::onnx::quantizer::{self, ScaleConfig};

pub fn run(model_path: &Path, output_path: &Path) -> Result<()> {
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

    let serialized = bincode::serialize(&quantized)?;
    let compressed = zstd::encode_all(serialized.as_slice(), 3)?;
    std::fs::write(output_path, &compressed)?;

    tracing::info!(
        "model written to {} ({} bytes compressed)",
        output_path.display(),
        compressed.len()
    );
    Ok(())
}

pub fn load_model(path: &Path) -> Result<crate::onnx::quantizer::QuantizedModel> {
    let compressed = std::fs::read(path)?;
    let decompressed = zstd::decode_all(compressed.as_slice())?;
    let model: crate::onnx::quantizer::QuantizedModel = bincode::deserialize(&decompressed)?;
    Ok(model)
}
