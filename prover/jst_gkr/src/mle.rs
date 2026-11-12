use arith::Field;

pub fn eq_eval<F: Field>(r: &[F], x: &[F]) -> F {
    assert_eq!(r.len(), x.len());
    let mut result = F::ONE;
    for (ri, xi) in r.iter().zip(x.iter()) {
        result *= *ri * *xi + (F::ONE - *ri) * (F::ONE - *xi);
    }
    result
}

pub fn build_eq_table<F: Field>(r: &[F]) -> Vec<F> {
    let n = r.len();
    let size = 1 << n;
    let mut table = vec![F::ONE; size];

    for (i, ri) in r.iter().enumerate() {
        let bit = 1 << i;
        for (idx, entry) in table.iter_mut().enumerate() {
            if idx & bit != 0 {
                *entry *= *ri;
            } else {
                *entry *= F::ONE - *ri;
            }
        }
    }
    table
}

pub fn evaluate_mle<F: Field>(evals: &[F], point: &[F]) -> F {
    let n = point.len();
    assert_eq!(evals.len(), 1 << n);

    let mut buf = evals.to_vec();
    for (i, pi) in point.iter().enumerate() {
        let half = 1 << (n - 1 - i);
        for j in 0..half {
            buf[j] = buf[2 * j] * (F::ONE - *pi) + buf[2 * j + 1] * *pi;
        }
    }
    buf[0]
}
