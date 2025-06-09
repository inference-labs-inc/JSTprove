use pyo3::{exceptions::PyRuntimeError, prelude::*};
use crate::model_analyzer::{analyze_model_internal, get_architecture_internal};
use crate::layer_handlers::layer_ir::LayerIR;

#[pyfunction]
fn analyze_model(path: &str) -> PyResult<Vec<LayerIR>> {
    analyze_model_internal(path)
        .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
}

#[pyfunction]
fn analyze_model_json(path: &str) -> PyResult<String> {
    let result = analyze_model_internal(path)
        .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;
    serde_json::to_string_pretty(&result)
        .map_err(|e| PyRuntimeError::new_err(format!("JSON error: {}", e)))
}

// #[pyfunction]
// fn get_architecture(model: Vec<LayerIR>) -> PyResult<Vec<LayerIR>> {
//     get_architecture_internal(model)
//         .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
// }


#[pymodule]
fn model_analyzer(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_model, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_model_json, m)?)?;
    Ok(())
}