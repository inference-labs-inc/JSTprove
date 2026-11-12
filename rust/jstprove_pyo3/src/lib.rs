use pyo3::prelude::*;

/// Run holographic GKR setup over a compiled circuit blob and
/// return the serialized verifying key bytes. The output is the
/// lightweight artifact validators need to verify proofs without
/// ever fetching the much larger circuit blob.
///
/// Currently only `proof_config = "goldilocks_ext4_whir"` is
/// wired (the post-quantum WHIR profile). Other proof configs
/// raise a Python `RuntimeError` mirroring the Rust dispatcher's
/// `Unsupported` variant.
///
/// On the validator side, the returned bytes can be passed to
/// `gkr::holographic::HolographicVerifyingKey::deserialize_from`
/// (or to a future Python-side helper) to recover the typed VK.
///
/// # Errors
/// Raises `RuntimeError` for unsupported proof configs, malformed
/// circuit blobs, wiring extraction failures (non-constant
/// coefficients, unsupported gate kinds), or serialization
/// failures.
#[pyfunction]
#[pyo3(signature = (proof_config, circuit_bytes))]
fn holographic_setup_vk(
    py: Python<'_>,
    proof_config: &str,
    circuit_bytes: &[u8],
) -> PyResult<Vec<u8>> {
    use jstprove_circuits::api::{setup_holographic_vk, ProofConfigType};

    let config: ProofConfigType = proof_config.parse().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown proof_config '{proof_config}': {e}"
        ))
    })?;

    let owned_bytes = circuit_bytes.to_vec();
    py.detach(move || setup_holographic_vk(config, &owned_bytes))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(holographic_setup_vk, m)?)?;
    Ok(())
}
