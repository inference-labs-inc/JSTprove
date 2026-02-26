use std::collections::HashMap;

use anyhow::{Result, bail};

use super::parser::{AttrValue, ParsedModel, TensorData};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum OpType {
    Add,
    Div,
    Sub,
    Mul,
    Gemm,
    Conv,
    Relu,
    MaxPool,
    BatchNormalization,
    Max,
    Min,
    Cast,
    Clip,
    Reshape,
    Flatten,
    Squeeze,
    Unsqueeze,
    Constant,
}

impl OpType {
    pub fn from_str(s: &str) -> Result<Self> {
        match s {
            "Add" => Ok(Self::Add),
            "Div" => Ok(Self::Div),
            "Sub" => Ok(Self::Sub),
            "Mul" => Ok(Self::Mul),
            "Gemm" | "Int64Gemm" => Ok(Self::Gemm),
            "Conv" | "Int64Conv" => Ok(Self::Conv),
            "Relu" | "Int64Relu" => Ok(Self::Relu),
            "MaxPool" | "Int64MaxPool" => Ok(Self::MaxPool),
            "BatchNormalization" | "Int64BatchNormalization" => Ok(Self::BatchNormalization),
            "Max" | "Int64Max" => Ok(Self::Max),
            "Min" | "Int64Min" => Ok(Self::Min),
            "Cast" => Ok(Self::Cast),
            "Clip" | "Int64Clip" => Ok(Self::Clip),
            "Reshape" => Ok(Self::Reshape),
            "Flatten" => Ok(Self::Flatten),
            "Squeeze" => Ok(Self::Squeeze),
            "Unsqueeze" => Ok(Self::Unsqueeze),
            "Constant" => Ok(Self::Constant),
            other => bail!("unsupported ONNX op: {}", other),
        }
    }

    pub fn is_shape_only(&self) -> bool {
        matches!(
            self,
            Self::Cast | Self::Reshape | Self::Flatten | Self::Squeeze | Self::Unsqueeze
        )
    }

    pub fn needs_rescale(&self) -> bool {
        matches!(
            self,
            Self::Gemm | Self::Conv | Self::BatchNormalization | Self::Mul
        )
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerNode {
    pub id: usize,
    pub name: String,
    pub op_type: OpType,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub weights: HashMap<String, TensorData>,
    pub attributes: HashMap<String, AttrValue>,
    pub output_shape: Vec<usize>,
    pub needs_rescale: bool,
    pub n_bits: Option<usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerGraph {
    pub layers: Vec<LayerNode>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub topo_order: Vec<usize>,
    pub input_shapes: HashMap<String, Vec<usize>>,
}

impl LayerGraph {
    pub fn from_parsed(model: &ParsedModel) -> Result<Self> {
        let initializer_names: std::collections::HashSet<&str> =
            model.initializers.keys().map(String::as_str).collect();

        let mut layers = Vec::new();

        for (idx, node) in model.nodes.iter().enumerate() {
            let op_type = OpType::from_str(&node.op_type)?;

            let mut weights = HashMap::new();
            for input_name in &node.inputs {
                if let Some(td) = model.initializers.get(input_name) {
                    weights.insert(input_name.clone(), td.clone());
                }
            }

            let needs_rescale = op_type.needs_rescale();

            layers.push(LayerNode {
                id: idx,
                name: node.name.clone(),
                op_type,
                inputs: node.inputs.clone(),
                outputs: node.outputs.clone(),
                weights,
                attributes: node.attributes.clone(),
                output_shape: vec![],
                needs_rescale,
                n_bits: None,
            });
        }

        let input_names: Vec<String> = model
            .inputs
            .iter()
            .filter(|io| !initializer_names.contains(io.name.as_str()))
            .map(|io| io.name.clone())
            .collect();

        let output_names: Vec<String> = model.outputs.iter().map(|io| io.name.clone()).collect();

        let topo_order = topological_sort(&layers)?;

        let input_shapes: HashMap<String, Vec<usize>> = model
            .inputs
            .iter()
            .filter(|io| !initializer_names.contains(io.name.as_str()))
            .map(|io| {
                let shape: Vec<usize> = io
                    .shape
                    .iter()
                    .map(|&d| if d <= 0 { 1 } else { d as usize })
                    .collect();
                let shape = if shape.len() >= 2 {
                    shape[1..].to_vec()
                } else {
                    shape
                };
                (io.name.clone(), shape)
            })
            .collect();

        Ok(LayerGraph {
            layers,
            input_names,
            output_names,
            topo_order,
            input_shapes,
        })
    }

    pub fn iter_topo(&self) -> impl Iterator<Item = &LayerNode> {
        self.topo_order.iter().map(|&idx| &self.layers[idx])
    }
}

impl LayerNode {
    pub fn get_ints_attr(&self, name: &str) -> Option<&[i64]> {
        match self.attributes.get(name)? {
            AttrValue::Ints(v) => Some(v),
            _ => None,
        }
    }

    pub fn get_int_attr(&self, name: &str) -> Option<i64> {
        match self.attributes.get(name)? {
            AttrValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_float_attr(&self, name: &str) -> Option<f32> {
        match self.attributes.get(name)? {
            AttrValue::Float(v) => Some(*v),
            _ => None,
        }
    }
}

fn topological_sort(layers: &[LayerNode]) -> Result<Vec<usize>> {
    let mut output_to_layer: HashMap<&str, usize> = HashMap::new();
    for (idx, layer) in layers.iter().enumerate() {
        for out in &layer.outputs {
            output_to_layer.insert(out.as_str(), idx);
        }
    }

    let mut in_degree = vec![0usize; layers.len()];
    let mut adj: Vec<Vec<usize>> = vec![vec![]; layers.len()];

    for (idx, layer) in layers.iter().enumerate() {
        for input in &layer.inputs {
            if let Some(&dep_idx) = output_to_layer.get(input.as_str()) {
                if dep_idx != idx {
                    adj[dep_idx].push(idx);
                    in_degree[idx] += 1;
                }
            }
        }
    }

    let mut queue: std::collections::VecDeque<usize> = in_degree
        .iter()
        .enumerate()
        .filter(|(_, &d)| d == 0)
        .map(|(i, _)| i)
        .collect();

    let mut order = Vec::with_capacity(layers.len());
    while let Some(idx) = queue.pop_front() {
        order.push(idx);
        for &next in &adj[idx] {
            in_degree[next] -= 1;
            if in_degree[next] == 0 {
                queue.push_back(next);
            }
        }
    }

    if order.len() != layers.len() {
        bail!("cycle detected in layer graph");
    }

    Ok(order)
}
