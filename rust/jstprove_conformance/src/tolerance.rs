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
        let delta = (expected - actual).unsigned_abs() as i64;
        if delta <= self.abs {
            return true;
        }
        if self.rel > 0.0 && expected != 0 {
            return (delta as f64 / expected.unsigned_abs() as f64) <= self.rel;
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
        let t = Tolerance { abs: 2, rel: 0.0, reason: "test" };
        assert!(t.check(10, 11));
        assert!(t.check(10, 12));
        assert!(!t.check(10, 13));
    }

    #[test]
    fn rel_tolerance() {
        let t = Tolerance { abs: 0, rel: 0.1, reason: "test" };
        assert!(t.check(100, 109)); // 9% — within 10%
        assert!(!t.check(100, 115)); // 15% — outside 10%
    }
}
