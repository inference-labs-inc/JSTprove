use crate::onnx::graph::{LayerGraph, OpType};

pub enum FusionPattern {
    GemmRelu(usize, usize),
    ConvRelu(usize, usize),
}

pub fn detect_fusion_patterns(graph: &LayerGraph) -> Vec<FusionPattern> {
    let mut patterns = Vec::new();
    let layers = &graph.layers;

    for i in 0..layers.len().saturating_sub(1) {
        let current = &layers[i];
        let next_idx = find_consumer(graph, i);
        if let Some(j) = next_idx {
            let next = &layers[j];
            match (current.op_type, next.op_type) {
                (OpType::Gemm, OpType::Relu) => {
                    patterns.push(FusionPattern::GemmRelu(i, j));
                }
                (OpType::Conv, OpType::Relu) => {
                    patterns.push(FusionPattern::ConvRelu(i, j));
                }
                _ => {}
            }
        }
    }

    patterns
}

fn find_consumer(graph: &LayerGraph, producer_idx: usize) -> Option<usize> {
    let producer = &graph.layers[producer_idx];
    let output_name = producer.outputs.first()?;

    for (i, layer) in graph.layers.iter().enumerate() {
        if i != producer_idx && layer.inputs.contains(output_name) {
            return Some(i);
        }
    }
    None
}
