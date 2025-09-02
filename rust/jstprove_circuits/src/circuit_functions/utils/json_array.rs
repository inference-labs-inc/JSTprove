/// External crate imports
use ndarray::{ArrayD, IxDyn};
use serde_json::Value;

use crate::circuit_functions::utils::errors::ArrayConversionError;

pub trait FromJsonNumber {
    fn from_json_number(n: &serde_json::Number) -> Option<Self>
    where
        Self: Sized;
}

impl FromJsonNumber for f64 {
    fn from_json_number(n: &serde_json::Number) -> Option<Self> {
        n.as_f64()
    }
}

impl FromJsonNumber for f32 {
    fn from_json_number(n: &serde_json::Number) -> Option<Self> {
        // TODO is this the best way to handle this?
        #[allow(clippy::cast_possible_truncation)]
        n.as_f64().map(|x| x as f32)
    }
}

impl FromJsonNumber for i32 {
    fn from_json_number(n: &serde_json::Number) -> Option<Self> {
        n.as_i64().and_then(|x| i32::try_from(x).ok())
    }
}

impl FromJsonNumber for i64 {
    fn from_json_number(n: &serde_json::Number) -> Option<Self> {
        n.as_i64()
    }
}

impl FromJsonNumber for u32 {
    fn from_json_number(n: &serde_json::Number) -> Option<Self> {
        n.as_u64().and_then(|x| u32::try_from(x).ok())
    }
}

impl FromJsonNumber for u64 {
    fn from_json_number(n: &serde_json::Number) -> Option<Self> {
        n.as_u64()
    }
}

fn get_array_shape(value: &Value) -> Result<Vec<usize>, ArrayConversionError> {
    let mut shape = Vec::new();
    let mut current = value;
    let mut path = vec![];

    loop {
        match current {
            Value::Array(arr) => {
                if arr.is_empty() {
                    shape.push(0);
                    break;
                }
                shape.push(arr.len());
                path.push(0); // always following index 0 for shape
                current = &arr[0];
            }
            Value::Number(_) => break,
            _ => {
                return Err(ArrayConversionError::InvalidArrayStructure {
                    expected: "array or number".to_string(),
                    found: format!("{:?}", current.clone()),
                });
            }
        }
    }
    Ok(shape)
}

fn flatten_to_typed_data<T>(value: &Value, data: &mut Vec<T>) -> Result<(), ArrayConversionError>
where
    T: Clone + FromJsonNumber,
{
    match value {
        Value::Number(n) => {
            let val = T::from_json_number(n).ok_or(ArrayConversionError::InvalidNumber)?;
            data.push(val);
            Ok(())
        }
        Value::Array(arr) => {
            for item in arr {
                flatten_to_typed_data(item, data)?;
            }
            Ok(())
        }
        _ => Err(ArrayConversionError::InvalidArrayStructure {
            expected: "array or number".to_string(),
            found: format!("{:?}", value.clone()),
        }),
    }
}

/// Converts a JSON `Value` into an `ndarray::ArrayD<T>`.
///
/// # Description
/// - If `value` is a number, returns a 0-dimensional array (scalar) containing that number.
/// - If `value` is a nested JSON array, recursively determines its shape, flattens the data,
///   and constructs an `ArrayD<T>` of the appropriate dimensions.
/// - Handles empty arrays by creating a 1-dimensional array of length 0.
///
/// # Type Parameters
/// - `T`: Target element type, must implement `Clone` and `FromJsonNumber`.
///
/// # Arguments
/// - `value`: JSON `Value` to convert into an array.
///
/// # Returns
/// An `ArrayD<T>` representing the input data.
///
/// # Errors
/// - [`ArrayConversionError::InvalidNumber`] if a JSON number cannot be converted to `T`.
/// - [`ArrayConversionError::ShapeError`] if the flattened data cannot be reshaped to the computed shape.
/// - [`ArrayConversionError::InvalidArrayStructure`] if the JSON value is not a number or array.
pub fn value_to_arrayd<T>(value: Value) -> Result<ArrayD<T>, ArrayConversionError>
where
    T: Clone + FromJsonNumber + 'static,
{
    match value {
        // Single number -> 0D array (scalar)
        Value::Number(n) => {
            let val = T::from_json_number(&n).ok_or(ArrayConversionError::InvalidNumber)?;
            Ok(ArrayD::from_elem(IxDyn(&[]), val))
        }

        // Array -> determine dimensions and convert
        Value::Array(arr) => {
            if arr.is_empty() {
                return ArrayD::from_shape_vec(IxDyn(&[0]), vec![])
                    .map_err(ArrayConversionError::ShapeError);
            }

            // Get the shape by walking through the nested structure
            let shape = get_array_shape(&Value::Array(arr.clone()))?;

            // Flatten all the data
            let mut data = Vec::new();
            flatten_to_typed_data::<T>(&Value::Array(arr), &mut data)?;

            // Create the ArrayD
            ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(ArrayConversionError::ShapeError)
        }

        other => Err(ArrayConversionError::InvalidArrayStructure {
            expected: "array or number".to_string(),
            found: format!("{:?}", other.clone()),
        }),
    }
}
