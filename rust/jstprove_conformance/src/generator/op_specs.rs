use crate::onnx_builder::{AttrValue, Initializer, NodeAttr};
use crate::tolerance::Tolerance;

use super::shapes::ShapeSpec;
use super::values::ValueSpec;

/// Describes one tensor input slot for a single-op model.
#[derive(Debug, Clone)]
pub struct TensorSpec {
    /// Name used in the ONNX graph for this tensor.
    pub name: &'static str,
    pub shape: ShapeSpec,
    pub values: ValueSpec,
    /// If true, bake into the ONNX model as an initializer (constant weight/index tensor).
    pub is_initializer: bool,
}

/// Describes how to generate all inputs for one operator invocation.
#[derive(Debug, Clone)]
pub struct OpInputSpec {
    pub op_name: &'static str,
    /// Dynamic graph inputs (not initializers) — in declaration order.
    pub inputs: Vec<TensorSpec>,
    /// ONNX attributes required by this node.
    pub attrs: Vec<NodeAttr>,
    /// Whether this operator uses the rescale path (Gemm, Conv, Mul, etc.).
    /// Affects tolerance: rescaled ops have ±1 rounding error.
    pub needs_rescale: bool,
    /// Tolerance to use for element-wise output comparison.
    pub tolerance: Tolerance,
}

// ---------------------------------------------------------------------------
// Canonical op specs — populated here for Relu, Add, Gemm.
// Milestones 3–4 will add the remaining operators.
// ---------------------------------------------------------------------------

/// Relu: single input, single output, no attributes, no rescale.
/// Input type INT64 — bypasses alpha-quantization in JSTProve, keeping the test simple.
pub fn relu_spec() -> OpInputSpec {
    OpInputSpec {
        op_name: "Relu",
        inputs: vec![TensorSpec {
            name: "x",
            shape: ShapeSpec::VEC,
            values: ValueSpec::Mixed,
            is_initializer: false,
        }],
        attrs: vec![],
        needs_rescale: false,
        tolerance: Tolerance::EXACT,
    }
}

/// Add: two dynamic inputs with matching shapes, no rescale.
pub fn add_spec() -> OpInputSpec {
    OpInputSpec {
        op_name: "Add",
        inputs: vec![
            TensorSpec {
                name: "a",
                shape: ShapeSpec::VEC,
                values: ValueSpec::Mixed,
                is_initializer: false,
            },
            TensorSpec {
                name: "b",
                shape: ShapeSpec::VEC,
                values: ValueSpec::Mixed,
                is_initializer: false,
            },
        ],
        attrs: vec![],
        needs_rescale: false,
        tolerance: Tolerance::EXACT,
    }
}

/// Gemm: A [M, K] × B [K, N] + bias [N] → output [M, N].
/// B and bias are treated as initializers (constants baked into ONNX model).
/// Uses FLOAT-like rescale; tolerance is ±1 LSB.
pub fn gemm_spec(m: usize, k: usize, n: usize) -> OpInputSpec {
    let m = m as i64;
    let k = k as i64;
    let n = n as i64;

    OpInputSpec {
        op_name: "Gemm",
        inputs: vec![
            // Dynamic input A [M, K]
            TensorSpec {
                name: "A",
                shape: ShapeSpec {
                    min_rank: 2,
                    max_rank: 2,
                    min_dim: 1,
                    max_dim: k as usize,
                    max_elements: (m * k) as usize,
                    allow_singleton: false,
                },
                values: ValueSpec::Uniform {
                    lo: -(super::values::ALPHA * 4),
                    hi: super::values::ALPHA * 4,
                },
                is_initializer: false,
            },
            // Initializer B [K, N]
            TensorSpec {
                name: "B",
                shape: ShapeSpec {
                    min_rank: 2,
                    max_rank: 2,
                    min_dim: 1,
                    max_dim: n as usize,
                    max_elements: (k * n) as usize,
                    allow_singleton: false,
                },
                values: ValueSpec::Uniform {
                    lo: -(super::values::ALPHA),
                    hi: super::values::ALPHA,
                },
                is_initializer: true,
            },
            // Initializer bias [N]
            TensorSpec {
                name: "C",
                shape: ShapeSpec {
                    min_rank: 1,
                    max_rank: 1,
                    min_dim: n as usize,
                    max_dim: n as usize,
                    max_elements: n as usize,
                    allow_singleton: false,
                },
                values: ValueSpec::Uniform {
                    lo: -(super::values::ALPHA * super::values::ALPHA),
                    hi: super::values::ALPHA * super::values::ALPHA,
                },
                is_initializer: true,
            },
        ],
        attrs: vec![NodeAttr {
            name: "transB",
            value: AttrValue::Int(0),
        }],
        needs_rescale: true,
        tolerance: Tolerance {
            abs: 1,
            rel: 0.0,
            reason: "Gemm rescale: ±1 LSB rounding from integer division by alpha",
        },
    }
}

/// Build fixed-shape Initializer entries for all `is_initializer = true` TensorSpecs
/// in an `OpInputSpec`, using the provided pre-sampled flat data slices.
///
/// Returns `(initializers, dynamic_input_names)` where `dynamic_input_names` are the
/// names of non-initializer inputs (to be used as `input_shapes` in `build_single_op_model`).
pub fn split_initializers(
    spec: &OpInputSpec,
    sampled_data: &[Vec<i64>],
    sampled_shapes: &[Vec<usize>],
) -> (Vec<Initializer>, Vec<usize>) {
    // Indices of initializer slots
    let mut init_indices = Vec::new();
    let mut dynamic_indices = Vec::new();
    for (i, t) in spec.inputs.iter().enumerate() {
        if t.is_initializer {
            init_indices.push(i);
        } else {
            dynamic_indices.push(i);
        }
    }

    let initializers: Vec<Initializer> = init_indices
        .iter()
        .map(|&i| {
            let dims: Vec<i64> = sampled_shapes[i].iter().map(|&d| d as i64).collect();
            Initializer {
                name: spec.inputs[i].name,
                dims,
                data: sampled_data[i].clone(),
            }
        })
        .collect();

    // Return the index of the first (and usually only) dynamic input for shape lookup
    (initializers, dynamic_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_spec_is_valid() {
        let s = relu_spec();
        assert_eq!(s.op_name, "Relu");
        assert_eq!(s.inputs.len(), 1);
        assert!(!s.needs_rescale);
    }

    #[test]
    fn add_spec_is_valid() {
        let s = add_spec();
        assert_eq!(s.op_name, "Add");
        assert_eq!(s.inputs.len(), 2);
        assert!(!s.needs_rescale);
    }

    #[test]
    fn gemm_spec_is_valid() {
        let s = gemm_spec(4, 8, 4);
        assert_eq!(s.op_name, "Gemm");
        // A (dynamic) + B (init) + C (init)
        assert_eq!(s.inputs.len(), 3);
        assert!(s.needs_rescale);
        assert!(s.tolerance.abs >= 1);
    }
}
