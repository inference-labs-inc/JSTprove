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
    fn compile(py: Python<'_>, model_path: &str, output_path: &str) -> PyResult<Circuit> {
        let model = model_path.to_string();
        let out = output_path.to_string();
        py.allow_threads(move || {
            jstprove_remainder::runner::compile::run(Path::new(&model), Path::new(&out), true)
        })
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
        py: Python<'_>,
        input_path: &str,
        witness_path: &str,
    ) -> PyResult<WitnessResult> {
        let model = self.model_path.clone();
        let input = input_path.to_string();
        let witness = witness_path.to_string();
        py.allow_threads(move || {
            jstprove_remainder::runner::witness::run(
                Path::new(&model),
                Path::new(&input),
                Path::new(&witness),
                true,
            )
        })
        .map_err(anyhow_to_pyerr)?;
        Ok(WitnessResult {
            witness_path: witness_path.to_string(),
            output_path: String::new(),
        })
    }

    fn prove(
        &self,
        py: Python<'_>,
        witness_path: &str,
        proof_path: &str,
    ) -> PyResult<String> {
        let model = self.model_path.clone();
        let witness = witness_path.to_string();
        let proof = proof_path.to_string();
        py.allow_threads(move || {
            jstprove_remainder::runner::prove::run(
                Path::new(&model),
                Path::new(&witness),
                Path::new(&proof),
                true,
            )
        })
        .map_err(anyhow_to_pyerr)?;
        Ok(proof_path.to_string())
    }

    fn verify(
        &self,
        py: Python<'_>,
        proof_path: &str,
        input_path: &str,
    ) -> PyResult<bool> {
        let model = self.model_path.clone();
        let proof = proof_path.to_string();
        let input = input_path.to_string();
        py.allow_threads(move || {
            jstprove_remainder::runner::verify::run(
                Path::new(&model),
                Path::new(&proof),
                Path::new(&input),
            )
        })
        .map_err(anyhow_to_pyerr)?;
        Ok(true)
    }

    fn generate_witness_batch(
        &self,
        py: Python<'_>,
        manifest_path: &str,
    ) -> PyResult<BatchResult> {
        let model = self.model_path.clone();
        let manifest = manifest_path.to_string();
        let r = py
            .allow_threads(move || {
                jstprove_remainder::runner::batch::run_batch_witness(
                    Path::new(&model),
                    Path::new(&manifest),
                    true,
                )
            })
            .map_err(anyhow_to_pyerr)?;
        Ok(BatchResult {
            succeeded: r.succeeded,
            failed: r.failed,
            errors: r.errors,
        })
    }

    fn prove_batch(&self, py: Python<'_>, manifest_path: &str) -> PyResult<BatchResult> {
        let model = self.model_path.clone();
        let manifest = manifest_path.to_string();
        let r = py
            .allow_threads(move || {
                jstprove_remainder::runner::batch::run_batch_prove(
                    Path::new(&model),
                    Path::new(&manifest),
                    true,
                )
            })
            .map_err(anyhow_to_pyerr)?;
        Ok(BatchResult {
            succeeded: r.succeeded,
            failed: r.failed,
            errors: r.errors,
        })
    }

    fn verify_batch(&self, py: Python<'_>, manifest_path: &str) -> PyResult<BatchResult> {
        let model = self.model_path.clone();
        let manifest = manifest_path.to_string();
        let r = py
            .allow_threads(move || {
                jstprove_remainder::runner::batch::run_batch_verify(
                    Path::new(&model),
                    Path::new(&manifest),
                )
            })
            .map_err(anyhow_to_pyerr)?;
        Ok(BatchResult {
            succeeded: r.succeeded,
            failed: r.failed,
            errors: r.errors,
        })
    }

    #[staticmethod]
    fn is_compatible(py: Python<'_>, model_path: &str) -> PyResult<(bool, Vec<String>)> {
        let path = model_path.to_string();
        let result =
            py.allow_threads(move || jstprove_remainder::onnx::compat::is_compatible(Path::new(&path)));
        match result {
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
