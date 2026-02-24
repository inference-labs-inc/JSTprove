use std::path::Path;

use pyo3::prelude::*;

fn anyhow_to_pyerr(e: anyhow::Error) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("{:#}", e))
}

#[pyclass]
#[derive(Clone)]
struct WitnessResult {
    #[pyo3(get)]
    witness_path: String,
    #[pyo3(get)]
    output_path: String,
}

#[pyclass]
#[derive(Clone)]
struct BatchResult {
    #[pyo3(get)]
    succeeded: usize,
    #[pyo3(get)]
    failed: usize,
    #[pyo3(get)]
    errors: Vec<(usize, String)>,
}

#[pyclass]
struct Circuit {
    model_path: String,
}

#[pymethods]
impl Circuit {
    #[staticmethod]
    fn compile(model_path: &str, output_path: &str) -> PyResult<Circuit> {
        jstprove_remainder::runner::compile::run(
            Path::new(model_path),
            Path::new(output_path),
            true,
        )
        .map_err(anyhow_to_pyerr)?;
        Ok(Circuit {
            model_path: output_path.to_string(),
        })
    }

    #[new]
    fn new(compiled_model_path: &str) -> Self {
        Circuit {
            model_path: compiled_model_path.to_string(),
        }
    }

    fn generate_witness(
        &self,
        input_path: &str,
        witness_path: &str,
    ) -> PyResult<WitnessResult> {
        jstprove_remainder::runner::witness::run(
            Path::new(&self.model_path),
            Path::new(input_path),
            Path::new(witness_path),
            true,
        )
        .map_err(anyhow_to_pyerr)?;
        Ok(WitnessResult {
            witness_path: witness_path.to_string(),
            output_path: String::new(),
        })
    }

    fn prove(
        &self,
        witness_path: &str,
        proof_path: &str,
    ) -> PyResult<String> {
        jstprove_remainder::runner::prove::run(
            Path::new(&self.model_path),
            Path::new(witness_path),
            Path::new(proof_path),
            true,
        )
        .map_err(anyhow_to_pyerr)?;
        Ok(proof_path.to_string())
    }

    fn verify(
        &self,
        proof_path: &str,
        input_path: &str,
    ) -> PyResult<bool> {
        jstprove_remainder::runner::verify::run(
            Path::new(&self.model_path),
            Path::new(proof_path),
            Path::new(input_path),
        )
        .map_err(anyhow_to_pyerr)?;
        Ok(true)
    }

    fn generate_witness_batch(
        &self,
        manifest_path: &str,
    ) -> PyResult<BatchResult> {
        let r = jstprove_remainder::runner::batch::run_batch_witness(
            Path::new(&self.model_path),
            Path::new(manifest_path),
            true,
        )
        .map_err(anyhow_to_pyerr)?;
        Ok(BatchResult {
            succeeded: r.succeeded,
            failed: r.failed,
            errors: r.errors,
        })
    }

    fn prove_batch(
        &self,
        manifest_path: &str,
    ) -> PyResult<BatchResult> {
        let r = jstprove_remainder::runner::batch::run_batch_prove(
            Path::new(&self.model_path),
            Path::new(manifest_path),
            true,
        )
        .map_err(anyhow_to_pyerr)?;
        Ok(BatchResult {
            succeeded: r.succeeded,
            failed: r.failed,
            errors: r.errors,
        })
    }

    fn verify_batch(
        &self,
        manifest_path: &str,
    ) -> PyResult<BatchResult> {
        let r = jstprove_remainder::runner::batch::run_batch_verify(
            Path::new(&self.model_path),
            Path::new(manifest_path),
        )
        .map_err(anyhow_to_pyerr)?;
        Ok(BatchResult {
            succeeded: r.succeeded,
            failed: r.failed,
            errors: r.errors,
        })
    }

    #[staticmethod]
    fn is_compatible(model_path: &str) -> PyResult<(bool, Vec<String>)> {
        match jstprove_remainder::onnx::compat::is_compatible(Path::new(model_path)) {
            Ok((compatible, issues)) => Ok((compatible, issues)),
            Err(e) => Ok((false, vec![format!("{:#}", e)])),
        }
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Circuit>()?;
    m.add_class::<WitnessResult>()?;
    m.add_class::<BatchResult>()?;
    Ok(())
}
