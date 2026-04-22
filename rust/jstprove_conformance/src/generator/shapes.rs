use rand::Rng;

/// Describes the space of shapes to draw from for one tensor input.
#[derive(Debug, Clone)]
pub struct ShapeSpec {
    pub min_rank: usize,
    pub max_rank: usize,
    /// Minimum size of each dimension (0 = allow empty tensors).
    pub min_dim: usize,
    /// Maximum size of each dimension.
    pub max_dim: usize,
    /// Total element count ceiling.
    pub max_elements: usize,
    /// Allow dimensions of size 1 (for broadcast testing).
    pub allow_singleton: bool,
    /// When set, `sample()` always returns exactly this shape, ignoring all other fields.
    /// Use for ops where contraction dimensions must be pinned (e.g. Gemm A=[M,K], B=[K,N]).
    pub fixed_dims: Option<Vec<usize>>,
}

impl ShapeSpec {
    pub const SCALAR: ShapeSpec = ShapeSpec {
        min_rank: 0,
        max_rank: 0,
        min_dim: 1,
        max_dim: 1,
        max_elements: 1,
        allow_singleton: true,
        fixed_dims: None,
    };

    pub const VEC: ShapeSpec = ShapeSpec {
        min_rank: 1,
        max_rank: 1,
        min_dim: 1,
        max_dim: 512,
        max_elements: 512,
        allow_singleton: true,
        fixed_dims: None,
    };

    pub const MATRIX: ShapeSpec = ShapeSpec {
        min_rank: 2,
        max_rank: 2,
        min_dim: 1,
        max_dim: 128,
        max_elements: 16384,
        allow_singleton: true,
        fixed_dims: None,
    };

    pub const TENSOR: ShapeSpec = ShapeSpec {
        min_rank: 1,
        max_rank: 5,
        min_dim: 1,
        max_dim: 64,
        max_elements: 65536,
        allow_singleton: true,
        fixed_dims: None,
    };

    /// Generate a concrete shape from this spec using the given RNG.
    pub fn sample(&self, rng: &mut impl Rng) -> Vec<usize> {
        if let Some(ref dims) = self.fixed_dims {
            return dims.clone();
        }
        let rank = if self.min_rank == self.max_rank {
            self.min_rank
        } else {
            rng.gen_range(self.min_rank..=self.max_rank)
        };

        if rank == 0 {
            return vec![];
        }

        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            let lo = self.min_dim;
            let hi = self.max_dim;
            let d = if lo == hi { lo } else { rng.gen_range(lo..=hi) };
            // Occasionally produce singleton dimensions when allowed
            let d = if self.allow_singleton && rng.gen_bool(0.15) {
                1
            } else {
                d
            };
            shape.push(d.max(lo));
        }

        // Clamp total elements
        self.clamp_to_max_elements(shape)
    }

    fn clamp_to_max_elements(&self, mut shape: Vec<usize>) -> Vec<usize> {
        loop {
            let total: usize = shape.iter().product::<usize>().max(1);
            if total <= self.max_elements {
                break;
            }
            // Halve the largest dimension
            let max_idx = shape
                .iter()
                .enumerate()
                .max_by_key(|(_, &d)| d)
                .map(|(i, _)| i)
                .unwrap();
            let new_val = (shape[max_idx] / 2).max(1);
            if new_val == shape[max_idx] {
                break; // already at minimum
            }
            shape[max_idx] = new_val;
        }
        shape
    }
}

/// Generate a pair of shapes (a, b) that are broadcast-compatible per ONNX numpy rules.
///
/// Shapes are right-aligned; each dimension must be equal, or one of them must be 1.
pub fn broadcast_pair(
    rng: &mut impl Rng,
    max_rank: usize,
    max_dim: usize,
) -> (Vec<usize>, Vec<usize>) {
    let rank = rng.gen_range(1..=max_rank.max(1));
    let mut a = Vec::with_capacity(rank);
    let mut b = Vec::with_capacity(rank);

    for _ in 0..rank {
        let d = rng.gen_range(1..=max_dim.max(1));
        match rng.gen_range(0u8..3) {
            0 => {
                // a broadcasts into b (a dim = 1)
                a.push(1);
                b.push(d);
            }
            1 => {
                // b broadcasts into a (b dim = 1)
                a.push(d);
                b.push(1);
            }
            _ => {
                // equal dims
                a.push(d);
                b.push(d);
            }
        }
    }
    (a, b)
}

/// Generate a pair of shapes that are broadcast-INCOMPATIBLE (for negative testing).
pub fn incompatible_pair(rng: &mut impl Rng) -> (Vec<usize>, Vec<usize>) {
    // Two different sizes, neither of which is 1
    let m = rng.gen_range(2usize..=8);
    let p = loop {
        let v = rng.gen_range(2usize..=8);
        if v != m {
            break v;
        }
    };
    let n = rng.gen_range(2usize..=8);
    (vec![m, n], vec![p, n])
}

/// Returns a fixed set of shapes that exercise known edge cases.
pub fn edge_case_shapes() -> Vec<Vec<usize>> {
    vec![
        vec![],           // scalar
        vec![1],          // singleton vector
        vec![4],          // simple vector
        vec![1, 4],       // row vector
        vec![4, 1],       // column vector
        vec![1, 1],       // scalar-like matrix
        vec![3, 4],       // plain matrix
        vec![2, 3, 4],    // 3-D tensor
        vec![1, 1, 4],    // broadcast-friendly 3-D
        vec![2, 3, 4, 5], // 4-D tensor
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn scalar_has_rank_zero() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let shape = ShapeSpec::SCALAR.sample(&mut rng);
        assert_eq!(
            shape,
            Vec::<usize>::new(),
            "SCALAR must produce rank-0 shape"
        );
    }

    #[test]
    fn vec_spec_produces_rank_one() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for _ in 0..20 {
            let shape = ShapeSpec::VEC.sample(&mut rng);
            assert_eq!(shape.len(), 1, "VEC must produce rank-1 shape");
            assert!(shape[0] >= 1 && shape[0] <= 512);
        }
    }

    #[test]
    fn matrix_spec_produces_rank_two() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..20 {
            let shape = ShapeSpec::MATRIX.sample(&mut rng);
            assert_eq!(shape.len(), 2, "MATRIX must produce rank-2 shape");
        }
    }

    #[test]
    fn tensor_spec_respects_max_elements() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for _ in 0..50 {
            let shape = ShapeSpec::TENSOR.sample(&mut rng);
            let n: usize = shape.iter().product::<usize>().max(1);
            assert!(
                n <= ShapeSpec::TENSOR.max_elements,
                "shape {shape:?} has {n} elements, exceeds max {}",
                ShapeSpec::TENSOR.max_elements
            );
        }
    }

    #[test]
    fn sample_is_deterministic() {
        let shape_a = {
            let mut rng = ChaCha8Rng::seed_from_u64(99);
            ShapeSpec::MATRIX.sample(&mut rng)
        };
        let shape_b = {
            let mut rng = ChaCha8Rng::seed_from_u64(99);
            ShapeSpec::MATRIX.sample(&mut rng)
        };
        assert_eq!(
            shape_a, shape_b,
            "sample must be deterministic for same seed"
        );
    }

    #[test]
    fn broadcast_pair_is_compatible() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for _ in 0..50 {
            let (a, b) = broadcast_pair(&mut rng, 4, 8);
            assert!(
                are_broadcast_compatible(&a, &b),
                "broadcast_pair produced incompatible shapes: {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn incompatible_pair_is_incompatible() {
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        for _ in 0..20 {
            let (a, b) = incompatible_pair(&mut rng);
            assert!(
                !are_broadcast_compatible(&a, &b),
                "incompatible_pair produced compatible shapes: {a:?} vs {b:?}"
            );
        }
    }

    /// ONNX numpy-style broadcast compatibility check.
    fn are_broadcast_compatible(a: &[usize], b: &[usize]) -> bool {
        let la = a.len();
        let lb = b.len();
        let rank = la.max(lb);
        for i in 0..rank {
            let da = if i < rank - la { 1 } else { a[i - (rank - la)] };
            let db = if i < rank - lb { 1 } else { b[i - (rank - lb)] };
            if da != db && da != 1 && db != 1 {
                return false;
            }
        }
        true
    }
}
