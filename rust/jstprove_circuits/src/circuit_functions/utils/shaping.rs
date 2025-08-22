/// Standard library imports
use std::collections::HashMap;

/// External crate imports
use ndarray::{Array2, ArrayD, IxDyn};

/// Internal crate imports
use crate::circuit_functions::{layers::{LayerError, LayerKind}, utils::{errors::ArrayConversionError, onnx_types::ONNXIO, UtilsError}};

/// Flattens an N-D array into shape `[prod(0..axis), prod(axis..)]`.
/// Returns `InvalidAxis` if `axis > ndim`.
pub fn onnx_flatten<T>(
    array: ArrayD<T>,
    axis: usize,
) -> Result<ArrayD<T>, ArrayConversionError> {
    let rank = array.ndim();
    if axis > rank {
        return Err(ArrayConversionError::InvalidAxis { axis, rank });
    }

    let shape = array.shape();
    let dim0 = shape[..axis].iter().product::<usize>();
    let dim1 = shape[axis..].iter().product::<usize>();

    // Uses #[from] on UtilsError::ShapeError
    let out = array.into_shape_with_order(IxDyn(&[dim0, dim1]))?;
    Ok(out)
}



pub fn get_inputs<T: Clone>(v: Vec<T>, inputs: Vec<ONNXIO>) -> Result<HashMap<String, ArrayD<T>>, UtilsError>{
    // Step 1: Compute total number of elements required
    let total_required: usize = inputs
        .iter()
        .map(|input| input.shape.iter().product::<usize>())
        .sum();

    // Step 2: Validate that v has exactly the required number of elements
    if v.len() != total_required {
        return Err(UtilsError::InputDataLengthMismatch {
            got: v.len(),
            required: total_required,
        });
    }

    // Step 3: Split and reshape
    let mut result = HashMap::new();
    let mut start = 0usize;

    for input_info in inputs {
        let num_elements: usize = input_info.shape.iter().product();
        let end = start + num_elements;

        let slice = v[start..end].to_vec(); // clone slice
        let arr = ArrayD::from_shape_vec(IxDyn(&input_info.shape), slice).map_err(|e| ArrayConversionError::ShapeError(e))?;

        result.insert(input_info.name.clone(), arr);
        start = end;
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: check_and_apply_transpose_array
// ─────────────────────────────────────────────────────────────────────────────

/// Applies a transpose to a 2D array if the transpose flag is set.
///
/// # Arguments
/// - `matrix`: A 2D array (`Array2<T>`) to conditionally transpose.
/// - `flag`: 0 means no transpose, 1 means transpose.
/// - `var_name`: Name of the transpose flag variable (for error messages).
/// - `layer_type`: Name of the layer type (for error messages).
/// - `layer_name`: Name of the layer instance (for error messages).
///
/// # Panics
/// Panics if `flag` is not 0 or 1.
        // other => Err(LayerError::InvalidParameterValue { layer: layer.try_into()?, param_name: var_name.to_string(), value: format!("{}", other) } ),


pub fn check_and_apply_transpose_array<T: Clone>(
    matrix: Array2<T>,
    flag: usize,
    var_name: &str,
    layer_type: &LayerKind,
    layer_name: &str,
) -> Result<Array2<T>, LayerError> {
    match flag {
        0 => Ok(matrix),
        1 => Ok(matrix.reversed_axes()),
        other => Err(LayerError::InvalidParameterValue{layer: layer_type.clone(), layer_name: layer_name.to_string(), param_name: var_name.to_string(), value: other.to_string() }
    ),
    }
}

pub fn infer_reshape_shape(input_size: usize, shape: &[isize]) -> Result<Vec<usize>, LayerError> {
    let mut minus_one_index: Option<usize> = None;

    let known_product: usize = shape.iter().enumerate().try_fold(1usize, |acc, (i, &d)| {
        match d {
            -1 => {
                if minus_one_index.is_some() {
                    return Err(LayerError::InvalidShape{layer: crate::circuit_functions::layers::LayerKind::Reshape, msg: "More than one -1 in reshape".to_string()});
                }
                minus_one_index = Some(i);
                Ok(acc) // -1 is a placeholder
            },
            d if d <= 0 => Err(LayerError::InvalidShape{layer: crate::circuit_functions::layers::LayerKind::Reshape, msg: format!("Invalid dimension {}", d)}),
            d => Ok(acc * (d as usize)),
        }
    })?;

    // Compute final shape
    let final_shape: Vec<usize> = shape.iter().enumerate().map(|(_, &d)| {
        if d == -1 {
            input_size / known_product
        } else {
            d as usize
        }
    }).collect();

    Ok(final_shape)
}
