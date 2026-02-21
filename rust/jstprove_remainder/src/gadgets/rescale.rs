pub fn compute_rescale(value: i64, alpha: i64, offset: i64) -> (i64, i64) {
    let shifted_dividend = alpha * offset + value;
    let q_shifted = shifted_dividend / alpha;
    let remainder = shifted_dividend - alpha * q_shifted;
    let quotient = q_shifted - offset;
    (quotient, remainder)
}

pub fn compute_rescale_array(
    values: &[i64],
    alpha: i64,
    offset: i64,
) -> (Vec<i64>, Vec<i64>) {
    let mut quotients = Vec::with_capacity(values.len());
    let mut remainders = Vec::with_capacity(values.len());
    for &v in values {
        let (q, r) = compute_rescale(v, alpha, offset);
        quotients.push(q);
        remainders.push(r);
    }
    (quotients, remainders)
}
