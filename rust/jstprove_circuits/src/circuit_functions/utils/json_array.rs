/// External crate imports
use ndarray::{ArrayD, IxDyn};
use serde_json::Value;

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
        n.as_f64().map(|x| x as f32)
    }
}

impl FromJsonNumber for i32 {
    fn from_json_number(n: &serde_json::Number) -> Option<Self> {
        n.as_i64().map(|x| x as i32)
    }
}

impl FromJsonNumber for i64 {
    fn from_json_number(n: &serde_json::Number) -> Option<Self> {
        n.as_i64()
    }
}

impl FromJsonNumber for u32 {
    fn from_json_number(n: &serde_json::Number) -> Option<Self> {
        n.as_u64().map(|x| x as u32)
    }
}

impl FromJsonNumber for u64 {
    fn from_json_number(n: &serde_json::Number) -> Option<Self> {
        n.as_u64()
    }
}

fn get_array_shape(value: &Value) -> Result<Vec<usize>, String> {
    let mut shape = Vec::new();
    let mut current = value;

    loop {
        match current {
            Value::Array(arr) => {
                if arr.is_empty() {
                    shape.push(0);
                    break;
                }
                shape.push(arr.len());
                current = &arr[0];
            }
            Value::Number(_) => break,
            _ => return Err("Invalid array structure".to_string()),
        }
    }

    Ok(shape)
}

fn flatten_to_typed_data<T>(value: &Value, data: &mut Vec<T>) -> Result<(), String>
where
    T: Clone + FromJsonNumber,
{
    match value {
        Value::Number(n) => {
            let val = T::from_json_number(n).ok_or("Invalid number for target type")?;
            data.push(val);
            Ok(())
        }
        Value::Array(arr) => {
            for item in arr {
                flatten_to_typed_data(item, data)?;
            }
            Ok(())
        }
        _ => Err("Expected number or array".to_string()),
    }
}

pub fn value_to_arrayd<T>(value: Value) -> Result<ArrayD<T>, String>
where
    T: Clone + FromJsonNumber + 'static,
{
    match value {
        // Single number -> 0D array (scalar)
        Value::Number(n) => {
            let val = T::from_json_number(&n).ok_or("Invalid number for target type")?;
            Ok(ArrayD::from_elem(IxDyn(&[]), val))
        }

        // Array -> determine dimensions and convert
        Value::Array(arr) => {
            if arr.is_empty() {
                return Ok(ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap());
            }

            // Get the shape by walking through the nested structure
            let shape = get_array_shape(&Value::Array(arr.clone()))?;

            // Flatten all the data
            let mut data = Vec::new();
            flatten_to_typed_data::<T>(&Value::Array(arr), &mut data)?;

            // Create the ArrayD
            ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|e| format!("Shape error: {}", e))
        }

        _ => Err("Expected number or array".to_string()),
    }
}
