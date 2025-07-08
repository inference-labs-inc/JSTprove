use pyo3::types::PyAny;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use crate::model_analyzer::{analyze_model_internal, get_architecture_internal, get_w_and_b_internal, quantize_model_internal};
use crate::layer_handlers::layer_ir::LayerIR;
use crate::model_runner::{run_model_from_f32_vec, run_model_from_i64_vec};

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

#[pyfunction]
fn get_architecture(model: Vec<LayerIR>) -> PyResult<Vec<LayerIR>> {
    get_architecture_internal(model)
        .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
}

#[pyfunction]
fn get_w_and_b(model: Vec<LayerIR>) -> PyResult<Vec<LayerIR>> {
    get_w_and_b_internal(model)
        .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
}

#[pyfunction]
fn quantize_model(input_path: &str, output_path: &str, scale: i64) {
    quantize_model_internal(input_path, output_path, scale)
        // .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
}

// #[pyfunction]
// fn run_quantize_inference(model_path: &str, inputs: ) {
//     run_model(input_path, output_path, scale)
//         // .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
// }
#[pyfunction]
fn run_model_from_f32(
    _py: Python<'_>,
    model_path: &str,
    input_data: Vec<f32>,
    shape: Vec<usize>,
) -> PyResult<Vec<f32>> {
    let input_vec: Vec<f32> = input_data;
        // .map_err(|e| PyRuntimeError::new_err(format!("Input extract error: {e}")))?;

    run_model_from_f32_vec(model_path, input_vec, shape)
        .map_err(|e| PyRuntimeError::new_err(format!("Inference error: {e}")))
}

#[pyfunction]
fn run_model_from_i64(
    _py: Python<'_>,
    model_path: &str,
    input_data: Vec<i64>,
    shape: Vec<usize>,
) -> PyResult<Vec<i64>> {
    let input_vec: Vec<i64> = input_data;
        // .map_err(|e| PyRuntimeError::new_err(format!("Input extract error: {e}")))?;

    run_model_from_i64_vec(model_path, input_vec, shape)
        .map_err(|e| PyRuntimeError::new_err(format!("Inference error: {e}")))
}

#[pymodule]
fn model_analyzer(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_model, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_model_json, m)?)?;
    m.add_function(wrap_pyfunction!(get_architecture, m)?)?;
    m.add_function(wrap_pyfunction!(get_w_and_b, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_model, m)?)?;
    m.add_function(wrap_pyfunction!(run_model_from_f32, m)?)?;
    m.add_function(wrap_pyfunction!(run_model_from_i64, m)?)?;



    Ok(())
}