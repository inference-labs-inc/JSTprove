/// Standard library imports
use std::{collections::HashMap, hash::BuildHasher, ops::Neg};

use ethnum::U256;
use ndarray::{ArrayD, IxDyn, Zip};
use rmpv::Value;

/// `ExpanderCompilerCollection` and proving framework imports
use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};
use expander_compiler::gkr_engine::{FieldEngine, GKREngine};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind},
    utils::{ArrayConversionError, UtilsError},
};

/// Load in circuit constant given i64, negative values are represented by p-x and positive values are x
pub fn load_circuit_constant<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: i64,
) -> Variable {
    if x < 0 {
        let y = api.constant(CircuitField::<C>::from_u256(U256::from(x.unsigned_abs())));
        api.neg(y)
    } else {
        api.constant(CircuitField::<C>::from_u256(U256::from(x.unsigned_abs())))
    }
}

/// Convert an `ArrayD` of i64 values to circuit constants (Variables)
pub fn load_array_constants<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input: &ArrayD<i64>,
) -> ArrayD<Variable> {
    let mut result = ArrayD::default(input.dim());
    Zip::from(&mut result)
        .and(input)
        .for_each(|out_elem, &val| {
            *out_elem = load_circuit_constant(api, val);
        });
    result
}

/// Load constants or retrieve inputs based on whether an initializer exists.
/// If the initializers exist, than use that, otherwise look for inputs in graph.
///
/// # Arguments
///
/// * `api` - The builder used to construct circuit constants and variables.
/// * `input` - A mapping from input names to input tensors.
/// * `input_name` - The name of the expected input tensor.
/// * `initializer` - Optional initializer tensor. If present, constants are loaded and returned.
/// * `layer_kind` - The type of layer requesting the tensor.
///
/// # Returns
/// Returns the loaded array or the input array associated with `input_name`.
///
/// # Errors
/// Returns a `CircuitError` if the requested input tensor does not exist in `input`.
pub fn load_array_constants_or_get_inputs<C: Config, Builder: RootAPI<C>, S: BuildHasher>(
    api: &mut Builder,
    input: &HashMap<String, ArrayD<Variable>, S>,
    input_name: &String,
    initializer: &Option<ArrayD<i64>>,
    layer_kind: LayerKind,
) -> Result<ArrayD<Variable>, CircuitError> {
    let a_input: ArrayD<Variable> = if let Some(init) = initializer {
        load_array_constants(api, init)
    } else {
        input
            .get(input_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: layer_kind,
                name: input_name.clone(),
            })?
            .clone()
    };
    Ok(a_input)
}

fn flatten_recursive(value: &Value, out: &mut Vec<i64>) -> Result<(), UtilsError> {
    match value {
        Value::Integer(n) => {
            if let Some(i) = n.as_i64() {
                out.push(i);
                Ok(())
            } else {
                Err(UtilsError::InvalidNumber {
                    value: value.clone(),
                })
            }
        }
        #[allow(clippy::cast_precision_loss)]
        Value::F64(f) => {
            if f.is_finite() && f.fract() == 0.0 && *f >= i64::MIN as f64 && *f <= i64::MAX as f64 {
                #[allow(clippy::cast_possible_truncation)]
                out.push(*f as i64);
                Ok(())
            } else {
                Err(UtilsError::InvalidNumber {
                    value: value.clone(),
                })
            }
        }
        #[allow(clippy::cast_precision_loss)]
        Value::F32(f) => {
            let d = f64::from(*f);
            if f.is_finite() && d.fract() == 0.0 && d >= i64::MIN as f64 && d <= i64::MAX as f64 {
                #[allow(clippy::cast_possible_truncation)]
                out.push(d as i64);
                Ok(())
            } else {
                Err(UtilsError::InvalidNumber {
                    value: value.clone(),
                })
            }
        }
        Value::Array(arr) => {
            for v in arr {
                flatten_recursive(v, out)?;
            }
            Ok(())
        }
        _ => Err(UtilsError::InvalidNumber {
            value: value.clone(),
        }),
    }
}

/// Converts an `rmpv::Value` into an n-dimensional array of circuit field elements.
///
/// The transformation proceeds in four steps:
/// 1. Flattening – The value tree is recursively flattened into a flat `Vec<i64>`.
/// 2. Shape preparation – The provided `input_shape` is used to define the
///    dimensions of the array. If the shape is empty, a dummy dimension of size 1 is added.
/// 3. Array construction – An [`ndarray::ArrayD<i64>`] is built from the flattened values
///    using the target shape.
/// 4. Field conversion – Each integer is mapped into the circuit’s native field type
///    via [`convert_val_to_field_element`].
///
/// # Errors
///
/// Returns a [`UtilsError`] if:
/// - The value cannot be flattened into a list of integers.
/// - The flattened input length does not match the expected shape.
/// - The shape itself is invalid for constructing an [`ndarray::ArrayD`].
///
pub fn get_nd_circuit_inputs<C: Config>(
    input: &Value,
    input_shape: &[usize],
) -> Result<ArrayD<CircuitField<C>>, UtilsError> {
    // 1) flatten JSON → Vec<i64>
    let mut flat = Vec::new();
    flatten_recursive(input, &mut flat)?;

    // 2) pad to at least 1 dimension
    let mut shape = input_shape.to_vec();
    if shape.is_empty() {
        shape.push(1);
    }

    // 3) build the i64 ndarray
    let a_i64: ArrayD<i64> =
        ArrayD::from_shape_vec(IxDyn(&shape), flat).map_err(ArrayConversionError::ShapeError)?;

    // 4) map to your circuit field type
    Ok(a_i64.mapv(|val| convert_val_to_field_element::<C>(val)))
}

// TODO change 64 bits to 128 across the board, or add checks. If more than 64 bits, fail
#[must_use]
pub fn convert_val_to_field_element<C: Config>(
    val: i64,
) -> <<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField {
    if val < 0 {
        CircuitField::<C>::from_u256(U256::from(val.unsigned_abs())).neg()
    } else {
        CircuitField::<C>::from_u256(U256::from(val.unsigned_abs()))
    }
}

fn determine_broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, CircuitError> {
    let rank = a.len().max(b.len());
    let mut result = vec![0; rank];

    for i in 0..rank {
        let a_dim = *a.get(a.len() - 1 - i).unwrap_or(&1);
        let b_dim = *b.get(b.len() - 1 - i).unwrap_or(&1);

        if a_dim == b_dim || a_dim == 1 || b_dim == 1 {
            result[rank - 1 - i] = usize::max(a_dim, b_dim);
        } else {
            return Err(CircuitError::Other(format!(
                "Cannot broadcast dimension {a_dim} and {b_dim}",
            )));
        }
    }

    Ok(result)
}

/// Broadcast two arrays using ONNX/NumPy multidirectional broadcasting.
///
/// # Errors
///
/// Returns a `CircuitError` if broadcasting is not possible.
/// This happens when the two shapes cannot be matched using multidirectional
/// broadcasting rules (i.e. neither dimension is equal or 1).
pub fn broadcast_two_arrays(
    a: &ArrayD<Variable>,
    b: &ArrayD<Variable>,
) -> Result<(ArrayD<Variable>, ArrayD<Variable>), CircuitError> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    // Determine output shape using numpy-like broadcasting rules
    let output_shape = determine_broadcast_shape(a_shape, b_shape)?;

    // Try broadcasting both arrays
    let a_bc = broadcast_array(a, &output_shape)?;
    let b_bc = broadcast_array(b, &output_shape)?;

    Ok((a_bc, b_bc))
}

/// Broadcast a single array into a specified broadcast shape.
///
/// # Errors
///
/// Returns a `CircuitError` if broadcasting fails.
/// This occurs when the array cannot expand to the target shape
/// following multidirectional broadcasting rules.
pub fn broadcast_array(
    a: &ArrayD<Variable>,
    broadcast_shape: &Vec<usize>,
) -> Result<ArrayD<Variable>, CircuitError> {
    let a_shape = a.shape();
    let out = a
        .broadcast(broadcast_shape.clone())
        .ok_or_else(|| {
            CircuitError::Other(format!(
                "Cannot broadcast A {a_shape:?} to {broadcast_shape:?}",
            ))
        })?
        .to_owned();
    Ok(out)
}

/// Reshape a 1D channel vector (mul/add) so it can broadcast
/// with an input tensor of shape [N, C, ...].
///
/// # Errors
///
/// Returns `CircuitError` if:
/// - `vec` is not 1-dimensional.
/// - The channel dimension of `vec` does not match `target`'s second dimension.
/// - Reshaping `vec` into shape `[C, 1, 1, ...]` fails.
pub fn reshape_channel_vector_for_broadcast(
    vec: &ArrayD<Variable>,
    target: &ArrayD<Variable>,
) -> Result<ArrayD<Variable>, CircuitError> {
    let target_shape = target.shape();
    let dims_x = target_shape.len();

    if vec.ndim() != 1 {
        return Err(CircuitError::Other(format!(
            "Expected 1D mul/add vector; got {:?}",
            vec.shape()
        )));
    }

    let channels = vec.len();

    if target_shape.len() < 2 || target_shape[1] != channels {
        return Err(CircuitError::Other(format!(
            "Channel mismatch: vector has {}, but input has {}",
            channels, target_shape[1]
        )));
    }

    // Build shape: [C, 1, 1, ...]
    let mut shape = Vec::with_capacity(dims_x);
    shape.push(channels);
    shape.extend(std::iter::repeat_n(1, dims_x - 2));

    // First reshape (adds new axes!)
    let reshaped = vec
        .to_shape(ndarray::IxDyn(&shape))
        .map_err(|_| CircuitError::Other(format!("Cannot reshape vector to {shape:?}")))?;

    Ok(reshaped.to_owned())
}
