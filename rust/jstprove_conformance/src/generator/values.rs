use rand::Rng;

/// α = 2^18 = 262144: the fixed-point scale factor used throughout JSTProve.
pub const ALPHA: i64 = 262144; // 2^18

/// Safe quantized value range: roughly ±2^31. Values outside this range may overflow
/// in subsequent multiply-rescale passes (Gemm, Conv produce α²-scaled intermediates).
pub const SAFE_RANGE: (i64, i64) = (-ALPHA * 8192, ALPHA * 8192); // ≈ ±2^31

/// Describes the distribution of values in a generated tensor.
#[derive(Debug, Clone)]
pub enum ValueSpec {
    /// Uniform random i64 in [lo, hi].
    Uniform { lo: i64, hi: i64 },
    /// Values near the quantization boundary ±(2^31) — exercises overflow paths.
    QuantizationBoundary,
    /// Values near zero (for divisor tensors). Set `include_zero` to test divide-by-zero.
    NearZero { include_zero: bool },
    /// Non-negative only (for Log, Sqrt inputs). `max` is the upper bound.
    NonNegative { max: i64 },
    /// Binary {0, 1} (for boolean/mask ops).
    Binary,
    /// Indices in [0, max) for Gather/TopK index inputs.
    Indices { max: i64 },
    /// Fractional exponents encoded as fixed-point (e.g. ALPHA/2 = 0.5) for Pow.
    FractionalExponent,
    /// Mix of positive, negative, and zero — broadest general-purpose distribution.
    Mixed,
}

impl ValueSpec {
    /// Sample a flat tensor of `n_elements` values using this spec.
    pub fn sample_tensor(&self, n_elements: usize, rng: &mut impl Rng) -> Vec<i64> {
        (0..n_elements).map(|_| self.sample_one(rng)).collect()
    }

    fn sample_one(&self, rng: &mut impl Rng) -> i64 {
        match self {
            ValueSpec::Uniform { lo, hi } => rng.gen_range(*lo..=*hi),

            ValueSpec::QuantizationBoundary => {
                let boundary = SAFE_RANGE.1; // 2^31
                let delta: i64 = rng.gen_range(-4..=4);
                if rng.gen_bool(0.5) {
                    boundary + delta
                } else {
                    -boundary + delta
                }
            }

            ValueSpec::NearZero { include_zero } => {
                let v: i64 = rng.gen_range(-8..=8);
                if !include_zero && v == 0 {
                    1
                } else {
                    v
                }
            }

            ValueSpec::NonNegative { max } => rng.gen_range(0..=*max),

            ValueSpec::Binary => rng.gen_range(0..=1),

            ValueSpec::Indices { max } => {
                if *max <= 0 {
                    0
                } else {
                    rng.gen_range(0..*max)
                }
            }

            ValueSpec::FractionalExponent => {
                // Common fractional exponents in fixed-point: 1/4, 1/2, 3/4, 1, 2
                let choices: &[i64] = &[ALPHA / 4, ALPHA / 2, (3 * ALPHA) / 4, ALPHA, 2 * ALPHA];
                choices[rng.gen_range(0..choices.len())]
            }

            ValueSpec::Mixed => {
                // 1/3 positive, 1/3 negative, 1/6 near-zero, 1/6 near-boundary
                let (lo, hi) = SAFE_RANGE;
                match rng.gen_range(0u8..6) {
                    0 | 1 => rng.gen_range(1..=hi),  // positive
                    2 | 3 => rng.gen_range(lo..=-1), // negative
                    4 => rng.gen_range(-8..=8),      // near zero
                    _ => {
                        let boundary = hi;
                        let delta: i64 = rng.gen_range(-4..=4);
                        if rng.gen_bool(0.5) {
                            boundary + delta
                        } else {
                            -boundary + delta
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn uniform_stays_in_range() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let spec = ValueSpec::Uniform { lo: -100, hi: 100 };
        for v in spec.sample_tensor(500, &mut rng) {
            assert!(v >= -100 && v <= 100, "uniform out of range: {v}");
        }
    }

    #[test]
    fn non_negative_no_negatives() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let spec = ValueSpec::NonNegative { max: ALPHA * 100 };
        for v in spec.sample_tensor(200, &mut rng) {
            assert!(v >= 0, "NonNegative produced negative: {v}");
        }
    }

    #[test]
    fn binary_only_zero_or_one() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let spec = ValueSpec::Binary;
        for v in spec.sample_tensor(200, &mut rng) {
            assert!(v == 0 || v == 1, "Binary produced {v}");
        }
    }

    #[test]
    fn near_zero_excludes_zero_when_requested() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let spec = ValueSpec::NearZero {
            include_zero: false,
        };
        for v in spec.sample_tensor(200, &mut rng) {
            assert!(v != 0, "NearZero(include_zero=false) produced 0");
        }
    }

    #[test]
    fn near_zero_range_is_small() {
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let spec = ValueSpec::NearZero { include_zero: true };
        for v in spec.sample_tensor(200, &mut rng) {
            assert!(v.abs() <= 8, "NearZero out of range: {v}");
        }
    }

    #[test]
    fn indices_in_range() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let spec = ValueSpec::Indices { max: 10 };
        for v in spec.sample_tensor(200, &mut rng) {
            assert!(v >= 0 && v < 10, "Index out of range: {v}");
        }
    }

    #[test]
    fn quantization_boundary_near_limit() {
        let mut rng = ChaCha8Rng::seed_from_u64(6);
        let spec = ValueSpec::QuantizationBoundary;
        let boundary = SAFE_RANGE.1;
        for v in spec.sample_tensor(100, &mut rng) {
            let dist = (v.abs() - boundary).abs();
            assert!(
                dist <= 4,
                "QuantizationBoundary too far from limit: {v}, boundary={boundary}"
            );
        }
    }

    #[test]
    fn sample_tensor_is_deterministic() {
        let a = {
            let mut rng = ChaCha8Rng::seed_from_u64(77);
            ValueSpec::Mixed.sample_tensor(64, &mut rng)
        };
        let b = {
            let mut rng = ChaCha8Rng::seed_from_u64(77);
            ValueSpec::Mixed.sample_tensor(64, &mut rng)
        };
        assert_eq!(a, b, "sample_tensor must be deterministic for same seed");
    }
}
