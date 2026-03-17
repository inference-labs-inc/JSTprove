use arith::Field;

pub fn fold_evals<F: Field>(evals: &[F], challenge: F) -> Vec<F> {
    let half = evals.len() / 2;
    let one_minus_r = F::ONE - challenge;
    let mut folded = Vec::with_capacity(half);
    for i in 0..half {
        folded.push(evals[i] * one_minus_r + evals[i + half] * challenge);
    }
    folded
}
