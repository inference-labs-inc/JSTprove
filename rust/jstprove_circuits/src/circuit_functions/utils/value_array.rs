use ndarray::{ArrayD, IxDyn};
use rmpv::Value;

use crate::circuit_functions::utils::errors::ArrayConversionError;

pub trait FromMsgpackValue {
    fn from_msgpack_value(v: &Value) -> Option<Self>
    where
        Self: Sized;
}

impl FromMsgpackValue for f64 {
    fn from_msgpack_value(v: &Value) -> Option<Self> {
        v.as_f64()
    }
}

impl FromMsgpackValue for f32 {
    #[allow(clippy::cast_possible_truncation)]
    fn from_msgpack_value(v: &Value) -> Option<Self> {
        v.as_f64().map(|x| x as f32)
    }
}

impl FromMsgpackValue for i32 {
    fn from_msgpack_value(v: &Value) -> Option<Self> {
        v.as_i64().and_then(|x| i32::try_from(x).ok())
    }
}

impl FromMsgpackValue for i64 {
    fn from_msgpack_value(v: &Value) -> Option<Self> {
        v.as_i64()
    }
}

impl FromMsgpackValue for u32 {
    fn from_msgpack_value(v: &Value) -> Option<Self> {
        v.as_u64().and_then(|x| u32::try_from(x).ok())
    }
}

impl FromMsgpackValue for u64 {
    fn from_msgpack_value(v: &Value) -> Option<Self> {
        v.as_u64()
    }
}

fn is_numeric(v: &Value) -> bool {
    matches!(v, Value::Integer(_) | Value::F32(_) | Value::F64(_))
}

fn validate_shape(
    value: &Value,
    depth: usize,
    shape: &mut Vec<usize>,
) -> Result<(), ArrayConversionError> {
    match value {
        Value::Array(arr) => {
            if depth == shape.len() {
                shape.push(arr.len());
            } else if arr.len() != shape[depth] {
                return Err(ArrayConversionError::InvalidArrayStructure {
                    expected: format!("length {} at depth {depth}", shape[depth]),
                    found: format!("length {} in {value:?}", arr.len()),
                });
            }
            if arr.is_empty() {
                return Ok(());
            }
            for item in arr {
                validate_shape(item, depth + 1, shape)?;
            }
            Ok(())
        }
        v if is_numeric(v) => {
            if depth != shape.len() {
                return Err(ArrayConversionError::InvalidArrayStructure {
                    expected: format!("array at depth {depth}"),
                    found: format!("{value:?}"),
                });
            }
            Ok(())
        }
        _ => Err(ArrayConversionError::InvalidArrayStructure {
            expected: "array or number".to_string(),
            found: format!("{value:?}"),
        }),
    }
}

fn get_array_shape(value: &Value) -> Result<Vec<usize>, ArrayConversionError> {
    let mut shape = Vec::new();
    validate_shape(value, 0, &mut shape)?;
    Ok(shape)
}

fn flatten_to_typed_data<T>(value: &Value, data: &mut Vec<T>) -> Result<(), ArrayConversionError>
where
    T: Clone + FromMsgpackValue,
{
    match value {
        v if is_numeric(v) => {
            let val = T::from_msgpack_value(v).ok_or(ArrayConversionError::InvalidNumber)?;
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
            found: format!("{value:?}"),
        }),
    }
}

/// # Errors
/// Returns `ArrayConversionError` if the value cannot be converted.
pub fn value_to_arrayd<T>(value: &Value) -> Result<ArrayD<T>, ArrayConversionError>
where
    T: Clone + FromMsgpackValue + 'static,
{
    match value {
        v if is_numeric(v) => {
            let val = T::from_msgpack_value(v).ok_or(ArrayConversionError::InvalidNumber)?;
            Ok(ArrayD::from_elem(IxDyn(&[]), val))
        }

        Value::Array(arr) => {
            if arr.is_empty() {
                return ArrayD::from_shape_vec(IxDyn(&[0]), vec![])
                    .map_err(ArrayConversionError::ShapeError);
            }

            let shape = get_array_shape(value)?;

            let mut data = Vec::new();
            flatten_to_typed_data::<T>(value, &mut data)?;

            ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(ArrayConversionError::ShapeError)
        }

        other => Err(ArrayConversionError::InvalidArrayStructure {
            expected: "array or number".to_string(),
            found: format!("{other:?}"),
        }),
    }
}

#[must_use]
pub fn map_get<'a>(map: &'a [(Value, Value)], key: &str) -> Option<&'a Value> {
    map.iter()
        .find(|(k, _)| k.as_str() == Some(key))
        .map(|(_, v)| v)
}
