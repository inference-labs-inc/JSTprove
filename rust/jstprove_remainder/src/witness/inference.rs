use std::collections::HashMap;

use anyhow::Result;

use crate::gadgets::rescale;
use crate::onnx::graph::{LayerGraph, LayerNode, OpType};
use crate::onnx::quantizer::ScaleConfig;

pub struct InferenceEngine {
    pub scale_config: ScaleConfig,
}

impl InferenceEngine {
    pub fn new(scale_config: ScaleConfig) -> Self {
        Self { scale_config }
    }

    pub fn run(
        &self,
        graph: &LayerGraph,
        input_data: &HashMap<String, Vec<i64>>,
    ) -> Result<InferenceResult> {
        let mut tensors: HashMap<String, Vec<i64>> = input_data.clone();
        let mut all_hints: HashMap<String, LayerHints> = HashMap::new();

        for layer in graph.iter_topo() {
            let layer_inputs: HashMap<String, Vec<i64>> = layer
                .inputs
                .iter()
                .filter_map(|name| tensors.get(name).map(|v| (name.clone(), v.clone())))
                .collect();

            let (output, hints) = self.execute_layer(layer, &layer_inputs)?;

            for out_name in &layer.outputs {
                tensors.insert(out_name.clone(), output.clone());
            }

            all_hints.insert(layer.name.clone(), hints);
        }

        let final_outputs: HashMap<String, Vec<i64>> = graph
            .output_names
            .iter()
            .filter_map(|name| tensors.get(name).map(|v| (name.clone(), v.clone())))
            .collect();

        Ok(InferenceResult {
            layer_outputs: tensors,
            layer_hints: all_hints,
            final_outputs,
        })
    }

    fn execute_layer(
        &self,
        layer: &LayerNode,
        inputs: &HashMap<String, Vec<i64>>,
    ) -> Result<(Vec<i64>, LayerHints)> {
        let alpha = self.scale_config.alpha;
        let offset = 1i64 << 30;

        let first_input = layer
            .inputs
            .first()
            .and_then(|name| inputs.get(name))
            .cloned()
            .unwrap_or_default();

        match layer.op_type {
            OpType::Add => {
                let b = layer
                    .inputs
                    .get(1)
                    .and_then(|name| inputs.get(name))
                    .cloned()
                    .unwrap_or_default();
                let output: Vec<i64> = first_input
                    .iter()
                    .zip(b.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                Ok((output, LayerHints::default()))
            }
            OpType::Sub => {
                let b = layer
                    .inputs
                    .get(1)
                    .and_then(|name| inputs.get(name))
                    .cloned()
                    .unwrap_or_default();
                let output: Vec<i64> = first_input
                    .iter()
                    .zip(b.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                Ok((output, LayerHints::default()))
            }
            OpType::Mul => {
                let b = layer
                    .inputs
                    .get(1)
                    .and_then(|name| inputs.get(name))
                    .cloned()
                    .unwrap_or_default();
                let product: Vec<i64> = first_input
                    .iter()
                    .zip(b.iter())
                    .map(|(a, b)| a * b)
                    .collect();
                let (quotients, remainders) =
                    rescale::compute_rescale_array(&product, alpha, offset);
                let hints = LayerHints {
                    quotients: Some(quotients.clone()),
                    remainders: Some(remainders),
                    ..Default::default()
                };
                Ok((quotients, hints))
            }
            OpType::Relu => {
                let output: Vec<i64> = first_input.iter().map(|&x| x.max(0)).collect();
                let delta_input: Vec<i64> =
                    output.iter().zip(first_input.iter()).map(|(o, x)| o - x).collect();
                let delta_zero: Vec<i64> = output.clone();
                let hints = LayerHints {
                    max_candidates: Some(output.clone()),
                    deltas: Some(vec![delta_input, delta_zero]),
                    ..Default::default()
                };
                Ok((output, hints))
            }
            OpType::Gemm => {
                let weight = layer
                    .inputs
                    .get(1)
                    .and_then(|name| {
                        layer.weights.get(name).map(|w| w.as_i64_vec())
                            .or_else(|| inputs.get(name).cloned())
                    })
                    .unwrap_or_default();
                let bias = layer
                    .inputs
                    .get(2)
                    .and_then(|name| {
                        layer.weights.get(name).map(|w| w.as_i64_vec())
                            .or_else(|| inputs.get(name).cloned())
                    });

                let m = first_input.len();
                let n = weight.len();
                let k = if m > 0 && n > 0 { n / (n / m.max(1)).max(1) } else { 0 };

                let result = vec![0i64; m];
                // TODO: proper shape inference from attributes
                // This is a placeholder - actual implementation needs proper matrix dims

                let (quotients, remainders) =
                    rescale::compute_rescale_array(&result, alpha, offset);
                let hints = LayerHints {
                    quotients: Some(quotients.clone()),
                    remainders: Some(remainders),
                    ..Default::default()
                };
                Ok((quotients, hints))
            }
            OpType::Conv => {
                // TODO: proper convolution with attribute extraction
                let (quotients, remainders) =
                    rescale::compute_rescale_array(&first_input, alpha, offset);
                let hints = LayerHints {
                    quotients: Some(quotients.clone()),
                    remainders: Some(remainders),
                    ..Default::default()
                };
                Ok((quotients, hints))
            }
            OpType::MaxPool => {
                let output: Vec<i64> = first_input.iter().map(|&x| x.max(0)).collect();
                let hints = LayerHints {
                    max_candidates: Some(output.clone()),
                    ..Default::default()
                };
                Ok((output, hints))
            }
            OpType::BatchNormalization => {
                let (quotients, remainders) =
                    rescale::compute_rescale_array(&first_input, alpha, offset);
                let hints = LayerHints {
                    quotients: Some(quotients.clone()),
                    remainders: Some(remainders),
                    ..Default::default()
                };
                Ok((quotients, hints))
            }
            OpType::Max => {
                let b = layer
                    .inputs
                    .get(1)
                    .and_then(|name| inputs.get(name))
                    .cloned()
                    .unwrap_or_default();
                let output: Vec<i64> = first_input
                    .iter()
                    .zip(b.iter())
                    .map(|(&a, &b)| a.max(b))
                    .collect();
                let hints = LayerHints {
                    max_candidates: Some(output.clone()),
                    ..Default::default()
                };
                Ok((output, hints))
            }
            OpType::Min => {
                let b = layer
                    .inputs
                    .get(1)
                    .and_then(|name| inputs.get(name))
                    .cloned()
                    .unwrap_or_default();
                let output: Vec<i64> = first_input
                    .iter()
                    .zip(b.iter())
                    .map(|(&a, &b)| a.min(b))
                    .collect();
                let hints = LayerHints {
                    max_candidates: Some(output.clone()),
                    ..Default::default()
                };
                Ok((output, hints))
            }
            OpType::Clip => {
                let output: Vec<i64> = first_input.iter().map(|&x| x.max(0)).collect();
                Ok((output, LayerHints::default()))
            }
            OpType::Reshape | OpType::Flatten | OpType::Squeeze | OpType::Unsqueeze => {
                Ok((first_input, LayerHints::default()))
            }
            OpType::Constant => {
                let data = layer
                    .weights
                    .values()
                    .next()
                    .map(|w| w.as_i64_vec())
                    .unwrap_or_default();
                Ok((data, LayerHints::default()))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub layer_outputs: HashMap<String, Vec<i64>>,
    pub layer_hints: HashMap<String, LayerHints>,
    pub final_outputs: HashMap<String, Vec<i64>>,
}

#[derive(Debug, Clone, Default)]
pub struct LayerHints {
    pub quotients: Option<Vec<i64>>,
    pub remainders: Option<Vec<i64>>,
    pub max_candidates: Option<Vec<i64>>,
    pub deltas: Option<Vec<Vec<i64>>>,
    pub digit_decompositions: Option<Vec<Vec<u64>>>,
    pub multiplicities: Option<Vec<u64>>,
}
