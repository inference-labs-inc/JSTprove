/// Standard library imports
use std::collections::HashMap;

/// External crate imports
use ndarray::{Array2, ArrayD, IxDyn};

/// Internal crate imports
use crate::circuit_functions::{
    layers::{LayerError, LayerKind},
    utils::{UtilsError, errors::ArrayConversionError, onnx_types::ONNXIO},
};

/// Flattens an N-dimensional array into a 2D array along a specified axis.
///
/// This operation mimics the ONNX `Flatten` operator:
/// - Dimensions before `axis` are collapsed into the first dimension.
/// - Dimensions from `axis` onward are collapsed into the second dimension.
///
/// # Arguments
///
/// * `array` – Input array to flatten.
/// * `axis` – Split point for collapsing dimensions. Must be ≤ rank of `array`.
///
/// # Returns
///
/// A 2D array (`ArrayD<T>`) of shape `[dim0, dim1]` where:
/// - `dim0 = product of dimensions before axis`
/// - `dim1 = product of dimensions after and including axis`
///
/// # Errors
///
/// Returns [`ArrayConversionError::InvalidAxis`] if `axis > rank`.
/// Returns [`ArrayConversionError::ShapeError`] if the reshape operation fails.
///
/// # Examples
///
/// ```ignore
/// let arr = Array::from_shape_vec((2, 3, 4), (0..24).collect()).unwrap().into_dyn();
///
/// // Flatten with axis = 1 → shape [2, 12]
/// let flat = onnx_flatten(arr.clone(), 1)?;
/// assert_eq!(flat.shape(), &[2, 12]);
///
/// // Flatten with axis = 0 → shape [1, 24]
/// let flat_all = onnx_flatten(arr, 0)?;
/// assert_eq!(flat_all.shape(), &[1, 24]);
/// ```
pub fn onnx_flatten<T>(array: ArrayD<T>, axis: usize) -> Result<ArrayD<T>, ArrayConversionError> {
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

/// Splits a flat vector into named, shaped tensors based on ONNX input metadata.
///
/// This function takes a 1D vector `v` (e.g., deserialized inputs) and reshapes
/// its contents into a set of `ArrayD`s according to the shapes specified in
/// the provided `inputs`. Each resulting array is stored in a `HashMap` where the
/// key is the input’s name.
///
/// # Arguments
///
/// - `v` – A flat vector of elements representing all inputs concatenated.
/// - `inputs` – Metadata describing each input’s name and shape (`ONNXIO`).
///
/// # Returns
///
/// A map from input names to reshaped arrays (`ArrayD<T>`).
///
/// # Errors
///
/// Returns [`UtilsError::InputDataLengthMismatch`] if the length of `v` does
/// not exactly match the total number of elements required by all `inputs`.
/// Returns [`ArrayConversionError::ShapeError`] if reshaping into the requested
/// shape fails.
///
/// # Examples
///
/// ```ignore
/// let v = vec![1, 2, 3, 4, 5, 6];
/// let inputs = vec![
///     ONNXIO { name: "x".into(), shape: vec![2, 3] }
/// ];
///
/// let result = get_inputs(&v, &inputs)?;
/// assert_eq!(result["x"].shape(), &[2, 3]);
/// ```
///
pub fn get_inputs<T: Clone>(
    v: &[T],
    inputs: &[ONNXIO],
) -> Result<HashMap<String, ArrayD<T>>, UtilsError> {
    let total_required: usize = inputs
        .iter()
        .map(|input| input.shape.iter().product::<usize>())
        .sum();

    if v.len() != total_required {
        return Err(UtilsError::InputDataLengthMismatch {
            got: v.len(),
            required: total_required,
        });
    }

    let mut result = HashMap::new();
    let mut start = 0usize;

    for input_info in inputs {
        let num_elements: usize = input_info.shape.iter().product();
        let end = start + num_elements;

        let slice = v[start..end].to_vec(); // clone slice
        let arr = ArrayD::from_shape_vec(IxDyn(&input_info.shape), slice)
            .map_err(ArrayConversionError::ShapeError)?;

        result.insert(input_info.name.clone(), arr);
        start = end;
    }

    Ok(result)
}

/// Optionally transposes a 2D array (`Array2`) based on a control flag.
///
/// This is typically used for layers that support an optional
/// transpose parameter (e.g., matrix multiplications in ONNX models).
///
/// # Behavior
///
/// - `flag == 0` → returns the matrix unchanged.
/// - `flag == 1` → returns the matrix with axes reversed (i.e., transposed).
/// - Any other value results in an error.
///
/// # Arguments
///
/// - `matrix` – The 2D input array.
/// - `flag` – Control parameter (0 = no transpose, 1 = transpose).
/// - `var_name` – Name of the ONNX parameter controlling the flag, used for error messages.
/// - `layer_type` – The kind of layer this parameter belongs to, for contextual error reporting.
/// - `layer_name` – The name/identifier of the layer instance, for contextual error reporting.
///
/// # Returns
///
/// Either the original matrix, its transpose, or an error if the flag is invalid.
///
/// # Errors
///
/// Returns [`LayerError::InvalidParameterValue`] if `flag` is neither `0` nor `1`.
///
/// # Examples
///
/// ```ignore
/// let m = array![[1, 2], [3, 4]];
///
/// // flag = 0 → unchanged
/// assert_eq!(check_and_apply_transpose_array(m.clone(), 0, "transpose", &LayerKind::Gemm, "layer1")?, m);
///
/// // flag = 1 → transpose
/// let expected = array![[1, 3], [2, 4]];
/// assert_eq!(check_and_apply_transpose_array(m, 1, "transpose", &LayerKind::Gemm, "layer1")?, expected);
///
/// // invalid flag
/// assert!(check_and_apply_transpose_array(m, 2, "transpose", &LayerKind::Gemm, "layer1").is_err());
/// ```
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
        other => Err(LayerError::InvalidParameterValue {
            layer: layer_type.clone(),
            layer_name: layer_name.to_string(),
            param_name: var_name.to_string(),
            value: other.to_string(),
        }),
    }
}

/// Infers the target shape for a reshape operation, supporting a single inferred dimension (`-1`).
///
/// This function validates and computes the final shape of a tensor when
/// reshaping, ensuring that the total number of elements remains consistent
/// with the given `input_size`.
///
/// # Behavior
///
/// - Each element in `shape` specifies the size of a dimension:
///   - positive integers (`> 0`) are used as-is.
///   - `-1` indicates an inferred dimension. At most one `-1` is allowed,
///     and its size will be computed so that the product of all dimensions
///     equals `input_size`.
/// - The product of all dimensions (after resolving `-1`) must exactly match
///   `input_size`.
///
/// # Arguments
///
/// * `input_size` – The total number of elements in the input tensor.
/// * `shape` – A slice describing the target dimensions. May contain at most one `-1`.
///
/// # Returns
///
/// A vector of concrete dimension sizes (`Vec<usize>`) representing the resolved
/// shape of the reshaped tensor.
///
/// # Errors
///
/// Returns a [`LayerError::InvalidShape`] if:
/// - More than one `-1` is present in `shape`.
/// - Any dimension is `0` or negative (other than `-1`).
/// - A dimension cannot be converted to `usize`.
/// - The inferred shape does not evenly divide `input_size`.
///
/// # Examples
///
/// ```ignore
/// // Input tensor has 12 elements
/// let input_size = 12;
///
/// // Reshape to 3 × 4 (no inference)
/// assert_eq!(infer_reshape_shape(input_size, &[3, 4])?, vec![3, 4]);
///
/// // Reshape with inference (-1 → 6, because 2 × 6 = 12)
/// assert_eq!(infer_reshape_shape(input_size, &[2, -1])?, vec![2, 6]);
///
/// // Invalid: two -1s
/// assert!(infer_reshape_shape(input_size, &[-1, -1]).is_err());
///
/// // Invalid: product does not match input size
/// assert!(infer_reshape_shape(input_size, &[5, -1]).is_err());
/// ```
///
/// # Notes
///
/// This function mirrors the behavior of reshape operations in libraries like
/// ONNX, with the same rules around `-1`.
///
pub fn infer_reshape_shape(input_size: usize, shape: &[isize]) -> Result<Vec<usize>, LayerError> {
    let mut minus_one_index: Option<usize> = None;

    let known_product: usize = shape.iter().enumerate().try_fold(1usize, |acc, (i, &d)| {
        match d {
            -1 => {
                if minus_one_index.is_some() {
                    return Err(LayerError::InvalidShape {
                        layer: crate::circuit_functions::layers::LayerKind::Reshape,
                        msg: "More than one -1 in reshape".to_string(),
                    });
                }
                minus_one_index = Some(i);
                Ok(acc) // -1 is a placeholder
            }
            d if d <= 0 => Err(LayerError::InvalidShape {
                layer: crate::circuit_functions::layers::LayerKind::Reshape,
                msg: format!("Invalid dimension {d}"),
            }),
            d => usize::try_from(d)
                .map(|d_usize| acc * d_usize)
                .map_err(|_| LayerError::InvalidShape {
                    layer: LayerKind::Reshape,
                    msg: format!("Invalid dimension {d}"),
                }),
        }
    })?;

    if minus_one_index.is_some() && input_size % known_product != 0 {
        return Err(LayerError::InvalidShape {
            layer: LayerKind::Reshape,
            msg: format!(
                "Cannot infer dimension: input size {input_size} is not divisible by {known_product}"
            ),
        });
    }

    if minus_one_index.is_none() && known_product != input_size {
        return Err(LayerError::InvalidShape {
            layer: LayerKind::Reshape,
            msg: format!(
                "Product of dimensions {known_product} does not match input size {input_size}"
            ),
        });
    }

    let final_shape: Result<Vec<usize>, LayerError> = shape
        .iter()
        .map(|&d| {
            if d == -1 {
                Ok(input_size / known_product)
            } else {
                usize::try_from(d).map_err(|_| LayerError::InvalidShape {
                    layer: LayerKind::Reshape,
                    msg: format!("Invalid dimension {d}"),
                })
            }
        })
        .collect();

    final_shape
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, array};

    #[test]
    fn onnx_flatten_axis_1() {
        let arr = Array::from_shape_vec((2, 3, 4), (0..24).collect())
            .unwrap()
            .into_dyn();
        let flat = onnx_flatten(arr, 1).unwrap();
        assert_eq!(flat.shape(), &[2, 12]);
    }

    #[test]
    fn onnx_flatten_axis_0() {
        let arr = Array::from_shape_vec((2, 3, 4), (0..24).collect())
            .unwrap()
            .into_dyn();
        let flat = onnx_flatten(arr, 0).unwrap();
        assert_eq!(flat.shape(), &[1, 24]);
    }

    #[test]
    fn onnx_flatten_axis_eq_rank() {
        let arr = Array::from_shape_vec((2, 3, 4), (0..24).collect())
            .unwrap()
            .into_dyn();
        let flat = onnx_flatten(arr, 3).unwrap();
        assert_eq!(flat.shape(), &[24, 1]);
    }

    #[test]
    fn onnx_flatten_axis_out_of_range() {
        let arr = Array::from_shape_vec((2, 3), (0..6).collect())
            .unwrap()
            .into_dyn();
        assert!(onnx_flatten(arr, 3).is_err());
    }

    #[test]
    fn get_inputs_single_tensor() {
        let v = vec![1, 2, 3, 4, 5, 6];
        let inputs = vec![ONNXIO {
            name: "x".into(),
            elem_type: 1,
            shape: vec![2, 3],
        }];
        let result = get_inputs(&v, &inputs).unwrap();
        assert_eq!(result["x"].shape(), &[2, 3]);
    }

    #[test]
    fn get_inputs_multiple_tensors() {
        let v = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let inputs = vec![
            ONNXIO {
                name: "a".into(),
                elem_type: 1,
                shape: vec![2, 3],
            },
            ONNXIO {
                name: "b".into(),
                elem_type: 1,
                shape: vec![2],
            },
        ];
        let result = get_inputs(&v, &inputs).unwrap();
        assert_eq!(result["a"].shape(), &[2, 3]);
        assert_eq!(result["b"].shape(), &[2]);
    }

    #[test]
    fn get_inputs_length_mismatch() {
        let v = vec![1, 2, 3];
        let inputs = vec![ONNXIO {
            name: "x".into(),
            elem_type: 1,
            shape: vec![2, 3],
        }];
        assert!(get_inputs(&v, &inputs).is_err());
    }

    #[test]
    fn transpose_flag_zero_unchanged() {
        let m = array![[1, 2], [3, 4]];
        let result =
            check_and_apply_transpose_array(m.clone(), 0, "transpose", &LayerKind::Gemm, "l1")
                .unwrap();
        assert_eq!(result, m);
    }

    #[test]
    fn transpose_flag_one_transposes() {
        let m = array![[1, 2], [3, 4]];
        let result =
            check_and_apply_transpose_array(m, 1, "transpose", &LayerKind::Gemm, "l1").unwrap();
        assert_eq!(result, array![[1, 3], [2, 4]]);
    }

    #[test]
    fn transpose_invalid_flag() {
        let m = array![[1, 2], [3, 4]];
        assert!(
            check_and_apply_transpose_array(m, 2, "transpose", &LayerKind::Gemm, "l1").is_err()
        );
    }

    #[test]
    fn reshape_no_inference() {
        assert_eq!(infer_reshape_shape(12, &[3, 4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn reshape_with_inference() {
        assert_eq!(infer_reshape_shape(12, &[2, -1]).unwrap(), vec![2, 6]);
    }

    #[test]
    fn reshape_two_minus_ones() {
        assert!(infer_reshape_shape(12, &[-1, -1]).is_err());
    }

    #[test]
    fn reshape_indivisible() {
        assert!(infer_reshape_shape(12, &[5, -1]).is_err());
    }

    #[test]
    fn reshape_zero_dimension() {
        assert!(infer_reshape_shape(12, &[0, 4]).is_err());
    }

    #[test]
    fn reshape_explicit_product_mismatch() {
        assert!(infer_reshape_shape(12, &[3, 3]).is_err());
    }
}
