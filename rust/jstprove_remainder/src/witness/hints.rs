use crate::gadgets::range_check;
use crate::gadgets::rescale;

pub fn compute_rescale_hints(
    values: &[i64],
    alpha: i64,
    offset: i64,
) -> (Vec<i64>, Vec<i64>) {
    rescale::compute_rescale_array(values, alpha, offset)
}

pub fn compute_digit_decomposition(
    values: &[i64],
    chunk_bits: usize,
    total_bits: usize,
) -> (Vec<Vec<u64>>, Vec<u64>) {
    let n_digits = range_check::num_digits_for_bits(total_bits, chunk_bits);
    let mut all_digits = Vec::new();
    let mut decompositions = Vec::with_capacity(values.len());

    for &v in values {
        let digits = range_check::decompose_to_digits(v, chunk_bits, n_digits);
        all_digits.extend_from_slice(&digits);
        decompositions.push(digits);
    }

    let multiplicities = range_check::compute_multiplicities(&all_digits, chunk_bits);
    (decompositions, multiplicities)
}

pub fn compute_max_hints(values: &[i64], max_val: i64) -> Vec<i64> {
    values.iter().map(|v| max_val - v).collect()
}

pub fn compute_min_hints(values: &[i64], min_val: i64) -> Vec<i64> {
    values.iter().map(|v| v - min_val).collect()
}
