use pyo3::IntoPyObjectExt;
use serde::Serialize;
use serde_json::json;

use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyList};
use pyo3::conversion::IntoPyObject;

use serde_json::Value;
// use tract_core::prelude::DatumType;

#[pyclass]
#[derive(Debug, Serialize, Clone)]
pub struct SerializableTensor {
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub dtype: String,
    
    pub values: Vec<serde_json::Value>,  // not exposed directly
}

impl SerializableTensor {
    pub fn from_tensor(t: &tract_onnx::prelude::Tensor) -> Option<Self> {
        let shape = t.shape().to_vec();
        let dtype = format!("{:?}", t.datum_type());

        macro_rules! try_convert {
            ($ty:ty) => {
                if let Ok(view) = t.to_array_view::<$ty>() {
                    let values = view.iter().map(|v| json!(v)).collect();
                    return Some(Self {
                        shape,
                        dtype,
                        values,
                    });
                }
            };
        }

        try_convert!(f32);
        try_convert!(f64);
        try_convert!(i32);
        try_convert!(i64);
        try_convert!(u8);
        try_convert!(bool);

        None
    }
}

#[pymethods]
impl SerializableTensor {
    pub fn get_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>>{
        let pyvals: Vec<PyObject> = self.values
            .iter()
            .map(|v| json_to_py(py, v))
            .collect::<Result<_, _>>()?;

        PyList::new(py, pyvals)
    }

    pub fn __str__(&self) -> String {
        format!("<Tensor shape={:?} dtype={} values.len={}>", self.shape, self.dtype, self.values.len())
    }
}

fn json_to_py(py: Python<'_>, val: &Value) -> PyResult<PyObject> {
    use Value::*;
    match val {
        Null => Ok(py.None()),
        Bool(b) => b.into_py_any(py),
        Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py_any(py)
            } else if let Some(f) = n.as_f64() {
                f.into_py_any(py)
            } else {
                Err(PyRuntimeError::new_err("Unsupported number in tensor"))
            }
        }
        String(s) => s.into_py_any(py),
        Array(arr) => {
            let pyarr: Vec<_> = arr.iter().map(|v| json_to_py(py, v)).collect::<Result<_, _>>()?;
            let pylist = PyList::new(py, pyarr).unwrap();
            pylist.into_py_any(py)
        }
        Object(_) => Err(PyRuntimeError::new_err("Tensor value cannot be object")),
    }
}


#[derive(Debug, Serialize, PartialEq, Eq, Clone)]
#[serde(tag = "type")]
pub enum LayerKind {
    Source,
    Const,
    Cast,
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
            LayerKind::Unknown { op } => format!("Unknown(op='{}')", op),
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
        axes: String,
        operating_dt: String,
        q_params: String
    },
    Cast {
        to: String
    }
    // Add more later
}

#[pyclass(name = "LayerParams")]
#[derive(Debug, Clone, Serialize)]
pub struct LayerParamsWrapper {
    pub inner: LayerParams,
}

#[pymethods]
impl LayerParamsWrapper {
    pub fn to_dict(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {}", e)))
    }

    pub fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    pub fn __repr__(&self) -> String {
        format!("<LayerParams {:?}>", self.inner)
    }
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
    pub params: Option<LayerParamsWrapper>,  

    #[pyo3(get)]
    pub tensor: Option<SerializableTensor>,
}
