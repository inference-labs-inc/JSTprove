/// Standard library imports
use std::ops::Neg;

/// External crate imports
use ethnum::U256;
use ndarray::{ArrayD, IxDyn, Zip};
use serde_json::Value;

/// `ExpanderCompilerCollection` and proving framework imports
use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};
use gkr_engine::{FieldEngine, GKREngine};

use crate::circuit_functions::utils::{ArrayConversionError, UtilsError};

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

fn flatten_recursive(value: &Value, out: &mut Vec<i64>) -> Result<(), UtilsError> {
    match value {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push(i);
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

/// Converts a JSON value into an n-dimensional array of circuit field elements.
///
/// This utility is used to transform structured JSON input (e.g., nested arrays)
/// into an [`ndarray::ArrayD`] of [`CircuitField`] values, suitable for use
/// in circuit assignments.
///
/// The transformation proceeds in four steps:
/// 1. Flattening – The JSON tree is recursively flattened into a flat `Vec<i64>`.
/// 2. Shape preparation – The provided `input_shape` is used to define the
///    dimensions of the array. If the shape is empty, a dummy dimension of size 1 is added.
/// 3. Array construction – An [`ndarray::ArrayD<i64>`] is built from the flattened values
///    using the target shape. Errors are raised if the length of the flat vector
///    does not match the product of the shape dimensions.
/// 4. Field conversion – Each integer is mapped into the circuit’s native field type
///    via [`convert_val_to_field_element`].
///
/// # Type Parameters
///
/// - `C` – A [`Config`] implementation that defines the underlying field configuration.
///
/// # Arguments
///
/// - `input` – A [`serde_json::Value`] containing the input data (expected to be an array
///   or nested arrays of integers).
/// - `input_shape` – The expected shape of the input tensor, given as a slice of dimension sizes.
///
/// # Returns
///
/// An [`ndarray::ArrayD`] of type [`CircuitField<C>`] matching the specified shape and
/// containing the converted field elements.
///
/// # Errors
///
/// Returns a [`UtilsError`] if:
/// - The JSON cannot be flattened into a list of integers.
/// - The flattened input length does not match the expected shape.
/// - The shape itself is invalid for constructing an [`ndarray::ArrayD`].
///
/// # Examples
///
/// ```ignore
/// use serde_json::json;
///
/// // JSON input: 2D array
/// let value = json!([[1, 2], [3, 4]]);
/// let shape = &[2, 2];
///
/// let arr = get_nd_circuit_inputs::<MyConfig>(&value, shape)?;
/// assert_eq!(arr.shape(), &[2, 2]);
/// ```
///
/// # Notes
///
/// - If `input_shape` is empty, a single dimension of length 1 will be used.
/// - All integers are first read as `i64` before conversion into the circuit field.
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
fn convert_val_to_field_element<C: Config>(
    val: i64,
) -> <<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField {
    if val < 0 {
        CircuitField::<C>::from_u256(U256::from(val.unsigned_abs())).neg()
    } else {
        CircuitField::<C>::from_u256(U256::from(val.unsigned_abs()))
    }
}
