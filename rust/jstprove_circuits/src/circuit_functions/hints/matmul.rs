//! Hint for unconstrained matrix multiplication.
//!
//! # ZK design
//!
//! Computing `C = A * B` inside the circuit via nested calls to
//! `api.unconstrained_mul` / `api.unconstrained_add` allocates O(ell * m * n)
//! Variables and Instructions in the expander_compiler builder: every
//! `unconstrained_mul(a, b)` actually emits a constrained `Mul`
//! instruction plus an `UnconstrainedIdentity` wrapper (see
//! `expander_compiler::frontend::builder::{unconstrained_add, unconstrained_mul}`).
//! For `[300, 256] @ [256, 256]` that is ~80M variables and ~80M
//! instructions — hundreds of GB of compiler bookkeeping at build time.
//!
//! This hint evaluates the product in plain field arithmetic outside
//! the circuit and returns the flat `ell * n` output entries as witness
//! values. Callers pair it with `freivalds_verify_matrix_product` to
//! constrain `A @ B = C` soundly; the probabilistic verifier is
//! O(reps * (ell*m + m*n + ell*n)), linear in the matrix sizes rather
//! than cubic.
//!
//! # Layout
//!
//! Inputs, flattened in row-major order:
//!
//!   [a_{0,0}, a_{0,1}, ..., a_{0,m-1}, a_{1,0}, ..., a_{ell-1,m-1},
//!    b_{0,0}, ..., b_{m-1,n-1}]
//!
//! followed by three constant u64-backed field values encoding the dimensions:
//!
//!   [ell, m, n]
//!
//! Outputs: `ell * n` field elements, the row-major entries of C = A @ B.

use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

/// Hint key used to register and look up this function.
pub const MATMUL_HINT_KEY: &str = "jstprove.matmul_hint";

/// # Errors
///
/// Returns `Error::UserError` when the input layout does not match
/// the declared `(ell, m, n)` dimensions or the output slice length
/// does not equal `ell * n`.
pub fn matmul_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.len() < 3 {
        return Err(Error::UserError(format!(
            "matmul_hint: expected at least 3 inputs (ell, m, n trailing), got {}",
            inputs.len()
        )));
    }
    let tail_len = inputs.len();
    let to_usize = |x: &F| -> Result<usize, Error> {
        let v = x.to_u256();
        if v > U256::from(usize::MAX as u128) {
            return Err(Error::UserError(
                "matmul_hint: dimension does not fit in usize on this target".to_string(),
            ));
        }
        let (hi, lo) = v.into_words();
        if hi != 0 {
            return Err(Error::UserError(
                "matmul_hint: dimension does not fit in usize on this target".to_string(),
            ));
        }
        usize::try_from(lo).map_err(|_| {
            Error::UserError(
                "matmul_hint: dimension does not fit in usize on this target".to_string(),
            )
        })
    };
    let ell = to_usize(&inputs[tail_len - 3])?;
    let m = to_usize(&inputs[tail_len - 2])?;
    let n = to_usize(&inputs[tail_len - 1])?;

    let overflow = || {
        Error::UserError(format!(
            "matmul_hint: dimension product overflows usize for ell={ell} m={m} n={n}"
        ))
    };
    let ell_m = ell.checked_mul(m).ok_or_else(overflow)?;
    let m_n = m.checked_mul(n).ok_or_else(overflow)?;
    let ell_n = ell.checked_mul(n).ok_or_else(overflow)?;
    let expected_input_len = ell_m
        .checked_add(m_n)
        .and_then(|s| s.checked_add(3))
        .ok_or_else(overflow)?;
    if inputs.len() != expected_input_len {
        return Err(Error::UserError(format!(
            "matmul_hint: expected {expected_input_len} inputs for ell={ell} m={m} n={n} (ell*m + m*n + 3), got {}",
            inputs.len()
        )));
    }
    if outputs.len() != ell_n {
        return Err(Error::UserError(format!(
            "matmul_hint: expected {ell_n} outputs (ell*n), got {}",
            outputs.len()
        )));
    }

    let a = &inputs[..ell_m];
    let b = &inputs[ell_m..ell_m + m_n];

    for i in 0..ell {
        let a_row = &a[i * m..(i + 1) * m];
        let out_row = &mut outputs[i * n..(i + 1) * n];
        for (j, out) in out_row.iter_mut().enumerate() {
            let mut acc = F::from_u256(U256::ZERO);
            for k in 0..m {
                let prod = a_row[k].mul(&b[k * n + j]);
                acc = acc.add(&prod);
            }
            *out = acc;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use expander_compiler::field::BN254Fr;

    type F = BN254Fr;

    fn f(n: u64) -> F {
        F::from_u256(U256::from(n))
    }

    #[test]
    fn matmul_hint_2x3_times_3x2() {
        // A = [[1, 2, 3],
        //      [4, 5, 6]]       (ell=2, m=3)
        // B = [[7,  8],
        //      [9, 10],
        //      [11, 12]]        (m=3, n=2)
        // C = A @ B =
        //     [[58,  64],
        //      [139, 154]]
        let mut inputs: Vec<F> = Vec::new();
        inputs.extend([1, 2, 3, 4, 5, 6].iter().map(|&v| f(v)));
        inputs.extend([7, 8, 9, 10, 11, 12].iter().map(|&v| f(v)));
        inputs.push(f(2));
        inputs.push(f(3));
        inputs.push(f(2));

        let mut outputs = vec![F::zero(); 4];
        matmul_hint::<F>(&inputs, &mut outputs).unwrap();

        let expected = [58u64, 64, 139, 154];
        for (o, &e) in outputs.iter().zip(expected.iter()) {
            assert_eq!(*o, f(e));
        }
    }

    #[test]
    fn matmul_hint_identity() {
        // A = I_3, B = random, C = B
        let mut inputs: Vec<F> = Vec::new();
        inputs.extend([1, 0, 0, 0, 1, 0, 0, 0, 1].iter().map(|&v| f(v)));
        let b_vals = [7, 8, 5, 9, 10, 6, 11, 12, 4];
        inputs.extend(b_vals.iter().map(|&v| f(v)));
        inputs.push(f(3));
        inputs.push(f(3));
        inputs.push(f(3));

        let mut outputs = vec![F::zero(); 9];
        matmul_hint::<F>(&inputs, &mut outputs).unwrap();

        for (o, &e) in outputs.iter().zip(b_vals.iter()) {
            assert_eq!(*o, f(e));
        }
    }

    #[test]
    fn matmul_hint_rejects_wrong_input_count() {
        let mut outputs = [F::zero(); 4];
        // too few inputs for ell=2 m=3 n=2
        let inputs = vec![f(1), f(2), f(3)];
        assert!(matmul_hint::<F>(&inputs, &mut outputs).is_err());
    }

    #[test]
    fn matmul_hint_rejects_wrong_output_count() {
        let mut inputs: Vec<F> = Vec::new();
        inputs.extend((0..6).map(f)); // A[2][3]
        inputs.extend((0..6).map(f)); // B[3][2]
        inputs.push(f(2));
        inputs.push(f(3));
        inputs.push(f(2));
        let mut outputs = [F::zero(); 5]; // should be 4
        assert!(matmul_hint::<F>(&inputs, &mut outputs).is_err());
    }

    #[test]
    fn matmul_hint_rejects_too_few_dim_tail() {
        let inputs: Vec<F> = vec![f(1), f(2)];
        let mut outputs = [F::zero(); 1];
        assert!(matmul_hint::<F>(&inputs, &mut outputs).is_err());
    }
}
