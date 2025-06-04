use pyo3::prelude::*;
use crate::core::analyze_model_internal;

#[pyfunction]
fn analyze_model(path: &str) -> PyResult<Vec<String>> {
    analyze_model_internal(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
}

#[pymodule]
fn model_analyzer(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_model, m)?)?;
    Ok(())
}