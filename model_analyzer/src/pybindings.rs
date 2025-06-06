use pyo3::{exceptions::PyRuntimeError, prelude::*};
use crate::model_analyzer::analyze_model_internal;

#[pyfunction]
fn analyze_model(path: &str) -> PyResult<String> {
    analyze_model_internal(path)
        .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
        .and_then(|analysis| {
            serde_json::to_string_pretty(&analysis)
                .map_err(|e| PyRuntimeError::new_err(format!("JSON error: {}", e)))
        })
}

#[pymodule]
fn model_analyzer(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_model, m)?)?;
    Ok(())
}