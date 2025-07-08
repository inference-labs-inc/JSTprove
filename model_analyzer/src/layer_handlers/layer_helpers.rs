use tract_onnx::prelude::*;
use std::collections::HashMap;

pub fn is_relu_like(
    node: &TypedNode,
    const_tensors: &HashMap<usize, Tensor>
) -> bool {
    // Must have exactly two inputs: Max(x, y)
    if node.inputs.len() != 2 {
        return false;
    }

    node.inputs.iter().any(|input| {
        const_tensors.get(&input.node).map_or(false, |tensor| {
            // Try to get float view
            if let Ok(array) = tensor.to_array_view::<f32>() {
                // Accept scalar or broadcasted scalar
                array.iter().all(|&v| v == 0.0) && (array.len() == 1 || array.shape().iter().all(|&d| d == 1))
            } else {
                false
            }
        })
    })
}
pub enum EinsumType {
    MatMul,
    TransposedRHSMatMul,
    TransposedLHSMatMul,
    TransposedRHSLHSMatMul,
    Unknown,
}
pub fn detect_einsum(equation: &str) -> EinsumType {
    let Some((inputs, output)) = equation.split_once("->") else { return EinsumType::Unknown };
    let mut inputs = inputs.split(',').map(str::trim);

    let Some(lhs) = inputs.next() else { return EinsumType::Unknown };
    let Some(rhs) = inputs.next() else { return EinsumType::Unknown };
    if inputs.next().is_some() { return EinsumType::Unknown; }

    if lhs.len() != 2 || rhs.len() != 2 || output.len() != 2 {
        return EinsumType::Unknown;
    }

    let [a1, a2] = lhs.chars().collect::<Vec<_>>()[..] else { return EinsumType::Unknown };
    let [b1, b2] = rhs.chars().collect::<Vec<_>>()[..] else { return EinsumType::Unknown };
    let output_chars: Vec<char> = output.chars().collect();

    let mut reduction_axis = None;

    for &c in &[a1, a2] {
        if c == b1 || c == b2 {
            reduction_axis = Some(c);
            break;
        }
    }

    let Some(r) = reduction_axis else { return EinsumType::Unknown };

    let mut non_reduced = vec![];
    for &c in &[a1, a2] {
        if c != r {
            non_reduced.push(c);
        }
    }
    for &c in &[b1, b2] {
        if c != r {
            non_reduced.push(c);
        }
    }

    if non_reduced.len() != 2 || !non_reduced.iter().all(|c| output_chars.contains(c)) {
        return EinsumType::Unknown;
    }

    if output_chars.contains(&r) {
        return EinsumType::Unknown;
    }

    let lhs_is_transposed = r == a1;
    let rhs_is_transposed = r == b2;

    match (lhs_is_transposed, rhs_is_transposed) {
        (false, false) => EinsumType::MatMul,
        (false, true) => EinsumType::TransposedRHSMatMul,
        (true, false) => EinsumType::TransposedLHSMatMul,
        (true, true) => EinsumType::TransposedRHSLHSMatMul,
    }
}
