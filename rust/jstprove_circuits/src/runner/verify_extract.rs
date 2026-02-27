use std::io::{Cursor, Read};

use expander_compiler::circuit::layered::NormalInputType;
use expander_compiler::circuit::layered::witness::Witness;
use expander_compiler::expander_binary::executor;
use expander_compiler::frontend::{ChallengeField, Config};
use expander_compiler::gkr_engine::MPIConfig;
use expander_compiler::serdes::ExpSerde;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{ToPrimitive, Zero};
use serde::{Deserialize, Serialize};

use super::errors::RunError;
use super::main_runner::auto_decompress_bytes;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedOutput {
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>,
    pub scale_base: u64,
    pub scale_exponent: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedOutput {
    pub valid: bool,
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>,
    pub scale_base: u64,
    pub scale_exponent: u64,
}

fn read_u64_le(cursor: &mut Cursor<&[u8]>) -> Result<u64, RunError> {
    let mut buf = [0u8; 8];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| RunError::Deserialize(format!("witness header u64: {e}")))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_32_bytes(cursor: &mut Cursor<&[u8]>) -> Result<[u8; 32], RunError> {
    let mut buf = [0u8; 32];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| RunError::Deserialize(format!("witness field element: {e}")))?;
    Ok(buf)
}

fn biguint_from_le_bytes(bytes: &[u8; 32]) -> BigUint {
    BigUint::from_bytes_le(bytes)
}

fn biguint_to_u64(val: &BigUint) -> Result<u64, RunError> {
    val.to_u64()
        .ok_or_else(|| RunError::Deserialize("field element too large for u64".into()))
}

fn from_field_repr(value: &BigUint, modulus: &BigUint) -> BigInt {
    let half = modulus >> 1;
    if *value > half {
        let diff = modulus - value;
        BigInt::from_biguint(Sign::Minus, diff)
    } else {
        BigInt::from_biguint(Sign::Plus, value.clone())
    }
}

#[allow(clippy::cast_precision_loss)]
fn descale_outputs(
    values: &[BigInt],
    scale_base: u64,
    scale_exp: u64,
) -> Result<Vec<f64>, RunError> {
    let scale = (scale_base as f64).powi(
        i32::try_from(scale_exp)
            .map_err(|_| RunError::Deserialize("scale_exp overflows i32".into()))?,
    );
    if !scale.is_finite() || scale == 0.0 {
        return Err(RunError::Deserialize("invalid scale factor".into()));
    }
    values
        .iter()
        .map(|v| {
            let f = v.to_f64().unwrap_or(f64::INFINITY);
            if !f.is_finite() {
                return Err(RunError::Deserialize(
                    "non-finite value converting BigInt to f64".into(),
                ));
            }
            Ok(f / scale)
        })
        .collect()
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn scale_to_field(
    values: &[f64],
    scale_base: u64,
    scale_exp: u64,
    modulus: &BigUint,
) -> Result<Vec<BigUint>, RunError> {
    let scale_f64 = (scale_base as f64).powi(
        i32::try_from(scale_exp)
            .map_err(|_| RunError::Deserialize("scale_exp overflows i32".into()))?,
    );
    if !scale_f64.is_finite() || scale_f64 == 0.0 {
        return Err(RunError::Deserialize("invalid scale factor".into()));
    }
    values
        .iter()
        .map(|v| {
            if !v.is_finite() {
                return Err(RunError::Verify(format!(
                    "scale_to_field: non-finite input {v}"
                )));
            }
            let product = *v * scale_f64;
            if !product.is_finite() {
                return Err(RunError::Verify(format!(
                    "scale_to_field: non-finite scaled value {product}"
                )));
            }
            let rounded = product.round();
            if rounded.abs() > i128::MAX as f64 {
                return Err(RunError::Verify(
                    "scale_to_field: scaled value overflows i128".into(),
                ));
            }
            let scaled = rounded as i128;
            Ok(if scaled < 0 {
                let abs = BigUint::from(scaled.unsigned_abs());
                let reduced = &abs % modulus;
                if reduced.is_zero() {
                    BigUint::ZERO
                } else {
                    modulus - reduced
                }
            } else {
                BigUint::from(scaled as u128) % modulus
            })
        })
        .collect()
}

fn compare_field_values(
    expected: &[BigUint],
    actual: &[BigUint],
    modulus: &BigUint,
    tolerance: u64,
) -> bool {
    if expected.len() != actual.len() {
        return false;
    }
    let tol = BigUint::from(tolerance);
    let neg_tol = modulus - &tol;
    for (e, a) in expected.iter().zip(actual.iter()) {
        let diff = if e >= a {
            (e - a) % modulus
        } else {
            modulus - ((a - e) % modulus)
        };
        if diff > tol && diff < neg_tol {
            return false;
        }
    }
    true
}

fn parse_public_inputs_from_witness_bytes(
    data: &[u8],
) -> Result<(BigUint, Vec<BigUint>), RunError> {
    let mut cursor = Cursor::new(data);

    let _num_witnesses = read_u64_le(&mut cursor)?;
    let num_inputs = usize::try_from(read_u64_le(&mut cursor)?)
        .map_err(|_| RunError::Deserialize("num_inputs overflows usize".into()))?;
    let num_public_inputs = usize::try_from(read_u64_le(&mut cursor)?)
        .map_err(|_| RunError::Deserialize("num_public_inputs overflows usize".into()))?;
    let modulus_bytes = read_32_bytes(&mut cursor)?;
    let modulus = biguint_from_le_bytes(&modulus_bytes);
    if modulus.is_zero() {
        return Err(RunError::Deserialize("modulus is zero".into()));
    }

    let total_values = num_inputs.checked_add(num_public_inputs).ok_or_else(|| {
        RunError::Deserialize("num_inputs + num_public_inputs overflows usize".into())
    })?;
    let required_bytes = total_values
        .checked_mul(32)
        .ok_or_else(|| RunError::Deserialize("public input byte size overflows usize".into()))?;
    #[allow(clippy::cast_possible_truncation)]
    let remaining = data.len().saturating_sub(cursor.position() as usize);
    if remaining < required_bytes {
        return Err(RunError::Deserialize(format!(
            "witness too short: need {required_bytes} bytes for {total_values} field elements, have {remaining}"
        )));
    }
    let mut values = Vec::with_capacity(total_values);
    for _ in 0..total_values {
        let bytes = read_32_bytes(&mut cursor)?;
        values.push(biguint_from_le_bytes(&bytes));
    }

    let public_inputs = values[num_inputs..].to_vec();
    Ok((modulus, public_inputs))
}

fn extract_outputs_common(
    witness_data: &[u8],
    num_model_inputs: usize,
) -> Result<ExtractedOutput, RunError> {
    let (modulus, public_inputs) = parse_public_inputs_from_witness_bytes(witness_data)?;

    let min_public = num_model_inputs
        .checked_add(2)
        .ok_or_else(|| RunError::Deserialize("num_model_inputs overflows usize".into()))?;
    if public_inputs.len() < min_public {
        return Err(RunError::Deserialize(format!(
            "expected at least {} public inputs (num_model_inputs={} + 2 scale params), got {}",
            min_public,
            num_model_inputs,
            public_inputs.len()
        )));
    }

    let scale_base = biguint_to_u64(&public_inputs[public_inputs.len() - 2])?;
    let scale_exponent = biguint_to_u64(&public_inputs[public_inputs.len() - 1])?;

    let raw_model_inputs = &public_inputs[..num_model_inputs];
    let raw_model_outputs = &public_inputs[num_model_inputs..public_inputs.len() - 2];

    let signed_inputs: Vec<BigInt> = raw_model_inputs
        .iter()
        .map(|v| from_field_repr(v, &modulus))
        .collect();
    let signed_outputs: Vec<BigInt> = raw_model_outputs
        .iter()
        .map(|v| from_field_repr(v, &modulus))
        .collect();

    let inputs = descale_outputs(&signed_inputs, scale_base, scale_exponent)?;
    let outputs = descale_outputs(&signed_outputs, scale_base, scale_exponent)?;

    Ok(ExtractedOutput {
        inputs,
        outputs,
        scale_base,
        scale_exponent,
    })
}

/// # Errors
/// Returns `RunError` on deserialization or extraction failure.
pub fn extract_outputs_from_witness(
    witness_bytes: &[u8],
    num_model_inputs: usize,
) -> Result<ExtractedOutput, RunError> {
    let witness_data = auto_decompress_bytes(witness_bytes)?;
    extract_outputs_common(&witness_data, num_model_inputs)
}

/// # Errors
/// Returns `RunError` on deserialization, verification, or extraction failure.
pub fn verify_and_extract_from_bytes<C: Config>(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
    num_inputs: usize,
    expected_inputs: Option<&[f64]>,
) -> Result<VerifiedOutput, RunError> {
    let circuit_data = auto_decompress_bytes(circuit_bytes)?;
    let layered_circuit =
        expander_compiler::circuit::layered::Circuit::<C, NormalInputType>::deserialize_from(
            Cursor::new(&*circuit_data),
        )
        .map_err(|e| RunError::Deserialize(format!("circuit: {e:?}")))?;

    let mut expander_circuit = layered_circuit.export_to_expander_flatten();

    let witness_data = auto_decompress_bytes(witness_bytes)?;
    let witness = Witness::<C>::deserialize_from(Cursor::new(&*witness_data))
        .map_err(|e| RunError::Deserialize(format!("witness: {e:?}")))?;

    let (simd_input, simd_public_input) = witness.to_simd();
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input.clone_from(&simd_public_input);

    let proof_data = auto_decompress_bytes(proof_bytes)?;
    let (proof, claimed_v) = executor::load_proof_and_claimed_v::<ChallengeField<C>>(&proof_data)
        .map_err(|e| RunError::Deserialize(format!("proof: {e:?}")))?;

    let mpi_config = MPIConfig::verifier_new(1);
    let valid = executor::verify::<C>(&mut expander_circuit, mpi_config, &proof, &claimed_v);

    if !valid {
        return Ok(VerifiedOutput {
            valid: false,
            inputs: vec![],
            outputs: vec![],
            scale_base: 0,
            scale_exponent: 0,
        });
    }

    let extracted = extract_outputs_common(&witness_data, num_inputs)?;

    if let Some(expected) = expected_inputs {
        if expected.len() != num_inputs {
            return Err(RunError::Verify(format!(
                "expected_inputs length {} does not match num_inputs {}",
                expected.len(),
                num_inputs
            )));
        }
        let (modulus, public_inputs) = parse_public_inputs_from_witness_bytes(&witness_data)?;
        let raw_model_inputs = &public_inputs[..num_inputs];
        let expected_field = scale_to_field(
            expected,
            extracted.scale_base,
            extracted.scale_exponent,
            &modulus,
        )?;
        if !compare_field_values(&expected_field, raw_model_inputs, &modulus, 1) {
            return Err(RunError::Verify(
                "input verification failed: expected inputs do not match witness".into(),
            ));
        }
    }

    Ok(VerifiedOutput {
        valid,
        inputs: extracted.inputs,
        outputs: extracted.outputs,
        scale_base: extracted.scale_base,
        scale_exponent: extracted.scale_exponent,
    })
}
