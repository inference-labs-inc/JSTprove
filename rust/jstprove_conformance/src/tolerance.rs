/// How outputs from JSTProve and the reference backend are compared.
#[derive(Debug, Clone)]
pub struct Tolerance {
    /// Maximum absolute difference in quantized i64 units (0 = exact match).
    pub abs: i64,
    /// Maximum relative difference as a fraction of the reference value (0.0 = exact).
    pub rel: f64,
    /// Human-readable justification for this bound — required, used in failure messages.
    pub reason: &'static str,
}

impl Tolerance {
    pub const EXACT: Tolerance = Tolerance {
        abs: 0,
        rel: 0.0,
        reason: "structural op — no rounding",
    };

    /// Return true if `actual` is within tolerance of `expected`.
    pub fn check(&self, expected: i64, actual: i64) -> bool {
        // abs_diff returns u64 and is free of signed overflow even when expected
        // and actual have opposite signs of large magnitude (e.g. i64::MIN vs i64::MAX).
        let delta_u = expected.abs_diff(actual);
        if delta_u <= self.abs as u64 {
            return true;
        }
        if self.rel > 0.0 && expected != 0 {
            return (delta_u as f64 / expected.unsigned_abs() as f64) <= self.rel;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_passes_zero_delta() {
        assert!(Tolerance::EXACT.check(5, 5));
    }

    #[test]
    fn exact_fails_nonzero_delta() {
        assert!(!Tolerance::EXACT.check(5, 6));
    }

    #[test]
    fn abs_tolerance() {
        let t = Tolerance {
            abs: 2,
            rel: 0.0,
            reason: "test",
        };
        assert!(t.check(10, 11));
        assert!(t.check(10, 12));
        assert!(!t.check(10, 13));
    }

    #[test]
    fn rel_tolerance() {
        let t = Tolerance {
            abs: 0,
            rel: 0.1,
            reason: "test",
        };
        assert!(t.check(100, 109)); // 9% — within 10%
        assert!(!t.check(100, 115)); // 15% — outside 10%
    }

    #[test]
    fn extreme_value_wraparound() {
        // Opposite-sign extremes: distance between i64::MIN and i64::MAX is u64::MAX
        // (≈1.8×10¹⁹).  A naive signed subtraction wraps to 0, making EXACT pass
        // incorrectly.  Verify it correctly fails.
        assert!(!Tolerance::EXACT.check(i64::MIN, i64::MAX));
        assert!(!Tolerance::EXACT.check(i64::MAX, i64::MIN));

        // A large abs tolerance that genuinely spans the chosen pair.
        // Use ±(i64::MAX/2): distance = i64::MAX - 1, which fits in both i64 and u64.
        let half = i64::MAX / 2;
        let huge = Tolerance {
            abs: i64::MAX,
            rel: 0.0,
            reason: "extreme test",
        };
        assert!(huge.check(half, -half));
        assert!(huge.check(-half, half));
    }
}
