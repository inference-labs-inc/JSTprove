use serde::Serialize;

use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Serialize, Clone)]
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

#[derive(Debug, Serialize, PartialEq, Eq, Clone)]
#[serde(tag = "type")]
pub enum LayerKind {
    Source,
    Const,
    Conv,
    MatMul,
    Add,
    Relu,
    Reshape,
    Unknown { op: String },
}

#[pyclass(name = "LayerKind")]
#[derive(Debug, Clone, Serialize)]
pub struct LayerKindWrapper {
    pub kind: LayerKind,
}

impl From<LayerKind> for LayerKindWrapper {
    fn from(kind: LayerKind) -> Self {
        Self { kind }
    }
}

#[pymethods]
impl LayerKindWrapper {
    #[getter]
    pub fn name(&self) -> String {
        match &self.kind {
            LayerKind::Unknown { .. } => "Unknown".to_string(),
            _ => format!("{:?}", self.kind),
        }
    }

    pub fn __str__(&self) -> String {
        self.name()
    }

    pub fn __repr__(&self) -> String {
        format!("<LayerKind '{}'>", self.name())
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self.kind, LayerKind::Unknown { .. })
    }
}

#[pyclass]
#[derive(Debug, Serialize, Clone)]

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

// #[derive(Debug, Serialize)]
// pub struct LayerIR {
//     pub id: usize,
//     pub name: String,
//     pub kind: LayerKind,
//     pub inputs: Vec<usize>,
//     pub outputs: Vec<usize>,
//     pub shape: Vec<usize>,
//     pub constants: Vec<LayerConstant>,
//     pub params: Option<LayerParams>,
// }

// #[derive(Debug, Serialize)]
// pub struct LayerIR {
//     pub id: usize,
//     pub name: String,
//     pub kind: LayerKind,
//     pub inputs: Vec<usize>,
//     pub outputs: Vec<usize>,
//     pub shape: Vec<usize>,
//     pub tensor: Option<SerializableTensor>,  // for Const layers only
//     pub params: Option<LayerParams>,         // op-specific fields
// }



#[pyclass]
#[derive(Clone, Debug, Serialize)]
pub struct LayerIR {
    #[pyo3(get)]
    pub id: usize,

    #[pyo3(get)]
    pub name: String,

    #[pyo3(get)]
    pub kind: LayerKindWrapper,

    #[pyo3(get)]
    pub inputs: Vec<usize>,

    #[pyo3(get)]
    pub outputs: Vec<usize>,

    #[pyo3(get)]
    pub shape: Vec<usize>,

    #[pyo3(get)]
    pub params: Option<LayerParams>,  

    #[pyo3(get)]
    pub tensor: Option<SerializableTensor>,
}
