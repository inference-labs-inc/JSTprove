/// Standard library imports
use std::collections::HashMap;

/// External crate imports
use ndarray::{Array2, ArrayD, IxDyn};

/// Internal crate imports
use crate::circuit_functions::utils::onnx_types::ONNXIO;

pub fn onnx_flatten<T>(array: ArrayD<T>, axis: usize) -> ArrayD<T> {
    let shape = array.shape();
    let dim0 = shape[..axis].iter().product::<usize>();
    let dim1 = shape[axis..].iter().product::<usize>();

    array.into_shape_with_order(IxDyn(&[dim0, dim1])).unwrap()
}

pub fn get_inputs<T: Clone>(v: Vec<T>, inputs: Vec<ONNXIO>) -> HashMap<String, ArrayD<T>> {
    // Step 1: Compute total number of elements required
    let total_required: usize = inputs
        .iter()
        .map(|input| input.shape.iter().product::<usize>())
        .sum();

    // Step 2: Validate that v has exactly the required number of elements
    if v.len() != total_required {
        panic!(
            "Input data length mismatch: got {}, but {} elements required by input shapes",
            v.len(),
            total_required
        );
    }

    // Step 3: Split and reshape
    let mut result = HashMap::new();
    let mut start = 0;

    for input_info in inputs {
        let num_elements: usize = input_info.shape.iter().product();
        let end = start + num_elements;

        let slice = v[start..end].to_vec(); // clone slice
        let arr = ArrayD::from_shape_vec(IxDyn(&input_info.shape), slice)
            .expect("Invalid shape for input data");

        result.insert(input_info.name.clone(), arr);
        start = end;
    }

    result
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
pub fn check_and_apply_transpose_array<T: Clone>(
    matrix: Array2<T>,
    flag: usize,
    var_name: &str,
    layer_type: &str,
    layer_name: &str,
) -> Array2<T> {
    match flag {
        0 => matrix,
        1 => matrix.reversed_axes(), // transpose
        other => panic!(
            "Unsupported {} value {} in {} layer: {}",
            var_name, other, layer_type, layer_name
        ),
    }
}
