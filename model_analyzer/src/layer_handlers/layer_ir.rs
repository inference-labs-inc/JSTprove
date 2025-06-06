use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct SerializableTensor {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub values: Vec<f32>,
}

impl SerializableTensor {
    pub fn from_tensor(t: &tract_onnx::prelude::Tensor) -> Option<Self> {
        let shape = t.shape().to_vec();
        let dtype = format!("{:?}", t.datum_type());
        if let Ok(view) = t.to_array_view::<f32>() {
            Some(Self {
                shape,
                dtype,
                values: view.iter().copied().collect(),
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Serialize)]
pub enum LayerKind {
    Conv,
    MatMul,
    Add,
    Relu,
    Reshape,
    Input,
    Unknown(String),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", content = "params")]
pub enum LayerParams {
    Conv {
        group: usize,
        strides: Vec<usize>,
        dilations: Vec<usize>,
        padding: (Vec<usize>,Vec<usize>),
        kernel_shape: Vec<usize>,
        input_channels: usize,
        output_channels: usize,
    },
    EinSum {
        equation: String,
    },
    // Add more later
}

#[derive(Debug, Serialize)]
pub struct LayerConstant {
    pub input_index: usize,
    pub name: String,
    pub value: SerializableTensor,
}

#[derive(Debug, Serialize)]
pub struct LayerIR {
    pub id: usize,
    pub name: String,
    pub kind: LayerKind,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub shape: Vec<usize>,
    pub constants: Vec<LayerConstant>,
    pub params: Option<LayerParams>,
}