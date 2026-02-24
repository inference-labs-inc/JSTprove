use anyhow::{ensure, Result};

pub fn compute_rescale(value: i64, alpha: i64, offset: i64) -> Result<(i64, i64)> {
    ensure!(alpha != 0, "compute_rescale: alpha must be non-zero");
    let shifted_dividend = (alpha as i128) * (offset as i128) + (value as i128);
    let q_shifted = shifted_dividend / (alpha as i128);
    let remainder = shifted_dividend - (alpha as i128) * q_shifted;
    let quotient = q_shifted - (offset as i128);
    Ok((quotient as i64, remainder as i64))
}

pub fn compute_rescale_array(
    values: &[i64],
    alpha: i64,
    offset: i64,
) -> Result<(Vec<i64>, Vec<i64>)> {
    let mut quotients = Vec::with_capacity(values.len());
    let mut remainders = Vec::with_capacity(values.len());
    for &v in values {
        let (q, r) = compute_rescale(v, alpha, offset)?;
        quotients.push(q);
        remainders.push(r);
    }
    Ok((quotients, remainders))
}
