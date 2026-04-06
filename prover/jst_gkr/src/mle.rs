use arith::Field;

pub fn eq_eval<F: Field>(r: &[F], x: &[F]) -> F {
    assert_eq!(r.len(), x.len());
    let mut result = F::ONE;
    for i in 0..r.len() {
        result *= r[i] * x[i] + (F::ONE - r[i]) * (F::ONE - x[i]);
    }
    result
}

pub fn build_eq_table<F: Field>(r: &[F]) -> Vec<F> {
    let n = r.len();
    let size = 1 << n;
    let mut table = vec![F::ONE; size];

    for i in 0..n {
        let bit = 1 << i;
        for idx in 0..size {
            if idx & bit != 0 {
                table[idx] *= r[i];
            } else {
                table[idx] *= F::ONE - r[i];
            }
        }
    }
    table
}

pub fn evaluate_mle<F: Field>(evals: &[F], point: &[F]) -> F {
    let n = point.len();
    assert_eq!(evals.len(), 1 << n);

    let mut buf = evals.to_vec();
    for i in 0..n {
        let half = 1 << (n - 1 - i);
        for j in 0..half {
            buf[j] = buf[2 * j] * (F::ONE - point[i]) + buf[2 * j + 1] * point[i];
        }
    }
    buf[0]
}
