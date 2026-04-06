use arith::{ExtensionField, FFTField, Field};

use super::types::RATE_LOG;

pub fn reverse_bits(x: usize, bits: usize) -> usize {
    if bits == 0 {
        return 0;
    }
    x.reverse_bits() >> (usize::BITS as usize - bits)
}

pub fn bit_reverse_slice<T: Copy>(v: &mut [T]) {
    let n = v.len();
    if n <= 1 {
        return;
    }
    let bits = n.ilog2() as usize;
    for i in 0..n {
        let j = reverse_bits(i, bits);
        if i < j {
            v.swap(i, j);
        }
    }
}

pub fn rs_encode<F: FFTField>(evals: &[F]) -> Vec<F> {
    rs_encode_with_rate(evals, RATE_LOG)
}

pub fn rs_encode_with_rate<F: FFTField>(evals: &[F], rate_log: usize) -> Vec<F> {
    let n = evals.len();
    assert!(n.is_power_of_two());

    let codeword_len = n << rate_log;
    let mut padded = vec![F::ZERO; codeword_len];
    padded[..n].copy_from_slice(evals);

    F::fft_in_place(&mut padded);

    bit_reverse_slice(&mut padded);
    padded
}

pub fn verifier_folding_coeff<F: FFTField>(level: usize, index: usize) -> F {
    let g_inv = F::two_adic_generator(level + 1).inv().unwrap();
    let idx_bit_rev = reverse_bits(index, level);
    g_inv.exp(idx_bit_rev as u128) * F::INV_2
}

pub fn compute_twiddle_coeffs<F: FFTField>(level: usize) -> Vec<F> {
    let n = 1usize << level;
    let g_inv = F::two_adic_generator(level + 1).inv().unwrap();
    let inv_2 = F::INV_2;
    (0..n)
        .map(|i| {
            let idx_br = reverse_bits(i, level);
            g_inv.exp(idx_br as u128) * inv_2
        })
        .collect()
}

pub fn fold_codeword_first_round<F, EvalF>(
    codeword: &[F],
    challenge: EvalF,
    twiddle_coeffs: &[F],
) -> Vec<EvalF>
where
    F: FFTField,
    EvalF: ExtensionField<BaseField = F>,
{
    let half = codeword.len() / 2;
    assert_eq!(twiddle_coeffs.len(), half);
    let mut folded = Vec::with_capacity(half);
    for i in 0..half {
        let left = EvalF::from(codeword[2 * i]);
        let right = EvalF::from(codeword[2 * i + 1]);
        let lo = (left + right).mul_by_base_field(&F::INV_2);
        let hi = (left - right).mul_by_base_field(&twiddle_coeffs[i]);
        folded.push(lo + challenge * (hi - lo));
    }
    folded
}

pub fn fold_codeword<F, EvalF>(
    codeword: &[EvalF],
    challenge: EvalF,
    twiddle_coeffs: &[F],
) -> Vec<EvalF>
where
    F: FFTField,
    EvalF: ExtensionField<BaseField = F>,
{
    let half = codeword.len() / 2;
    assert_eq!(twiddle_coeffs.len(), half);
    let mut folded = Vec::with_capacity(half);
    for i in 0..half {
        let left = codeword[2 * i];
        let right = codeword[2 * i + 1];
        let lo = (left + right).mul_by_base_field(&F::INV_2);
        let hi = (left - right).mul_by_base_field(&twiddle_coeffs[i]);
        folded.push(lo + challenge * (hi - lo));
    }
    folded
}

pub fn codeword_fold_single<F, EvalF>(
    left: EvalF,
    right: EvalF,
    challenge: EvalF,
    twiddle_coeff: F,
) -> EvalF
where
    F: Field,
    EvalF: ExtensionField<BaseField = F>,
{
    let lo = (left + right).mul_by_base_field(&F::INV_2);
    let hi = (left - right).mul_by_base_field(&twiddle_coeff);
    lo + challenge * (hi - lo)
}
