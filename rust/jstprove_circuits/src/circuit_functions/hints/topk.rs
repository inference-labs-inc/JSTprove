// Hint function for the ONNX `TopK` operator.
//
// # ZK design
// TopK selects the K largest (or smallest) values from a 1-D lane of quantised
// integers. The selection and sort cannot be expressed as polynomials over a
// finite field, so the computation is performed outside the circuit (native
// Rust on i64 decoded values) and the K output values are injected as
// unconstrained witnesses.
//
// # Input layout
// `inputs[0..n-1]` : quantised input elements for one lane (n values).
// `inputs[n]`      : the scaling factor `scale = 2^scale_exponent`.
// `outputs.len()` == 2*K — K values followed by K indices.
//
// # Output layout
// `outputs[0..K-1]` : the K largest input values in descending order (when
//                     largest=1), encoded as field elements using the same
//                     two's-complement-mod-p convention as all other quantised
//                     tensors in this project.
// `outputs[K..2K-1]`: the original indices of the K selected values (as
//                     non-negative field elements).
//
// # Verification strategy
// The circuit constrains: (a) each output equals some input element via
// multiplexer, (b) outputs are sorted descending via pairwise range checks,
// (c) the K-th value is >= all non-selected inputs via range checks, and
// (d) indices are pairwise distinct.
//
// # Largest / sorted
// This hint always selects the K largest values in descending order, matching
// the ONNX defaults (`largest=1, sorted=1`). Smallest/unsorted are not yet
// supported.

use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

/// Hint key used to register and look up this function.
pub const TOPK_HINT_KEY: &str = "jstprove.topk_hint";

/// Hint function for `TopK` — selects K largest values from a fixed-point lane.
///
/// # Inputs
/// - `inputs[0..n-1]`: quantised lane values as field elements. Values above
///   `p/2` are treated as negative (two's-complement mod p).
/// - `inputs[n]`: scaling factor `scale = 2^scale_exponent` (positive i64).
///
/// # Outputs
/// - `outputs[i]`: the (i+1)-th largest value, encoded as a field element
///   using two's-complement mod p for negatives.
///
/// # Errors
/// Returns [`Error::UserError`] when `inputs.len() < outputs.len() + 1` (i.e.
/// fewer data values than K, or missing scale).
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
pub fn topk_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if outputs.len() % 2 != 0 {
        return Err(Error::UserError(format!(
            "topk_hint: outputs.len() must be even (2*K), got {}",
            outputs.len()
        )));
    }
    let k = outputs.len() / 2;
    if inputs.len() < k + 1 {
        return Err(Error::UserError(format!(
            "topk_hint: expected at least {} inputs (k={k} data + 1 scale), got {}",
            k + 1,
            inputs.len()
        )));
    }

    let n = inputs.len() - 1;

    let mut indexed: Vec<(i64, usize)> = inputs[..n]
        .iter()
        .enumerate()
        .map(|(i, &x)| (field_to_i64(x), i))
        .collect();

    indexed.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let encode = |val: i64| -> F {
        if val >= 0 {
            F::from_u256(U256::from(val as u64))
        } else {
            let mag = U256::from(val.unsigned_abs());
            F::from_u256(F::MODULUS - mag)
        }
    };

    for i in 0..k {
        outputs[i] = encode(indexed[i].0);
        outputs[k + i] = F::from_u256(U256::from(indexed[i].1 as u64));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ethnum::U256;
    use expander_compiler::field::BN254Fr;

    type F = BN254Fr;

    fn field(n: i64) -> F {
        if n >= 0 {
            F::from_u256(U256::from(n as u64))
        } else {
            let mag = U256::from(n.unsigned_abs());
            F::from_u256(F::MODULUS - mag)
        }
    }

    fn run_topk(xs: &[i64], k: usize) -> (Vec<i64>, Vec<usize>) {
        let scale: u64 = 1 << 18;
        let mut inputs: Vec<F> = xs.iter().map(|&x| field(x)).collect();
        inputs.push(F::from_u256(U256::from(scale)));
        let mut outputs = vec![F::zero(); 2 * k];
        topk_hint::<F>(&inputs, &mut outputs).unwrap();
        let values: Vec<i64> = outputs[..k]
            .iter()
            .map(|o| super::super::field_to_i64(*o))
            .collect();
        let indices: Vec<usize> = outputs[k..]
            .iter()
            .map(|o| o.to_u256().as_u64() as usize)
            .collect();
        (values, indices)
    }

    #[test]
    fn topk_k1_selects_largest() {
        let xs = [3i64, 1, 4, 1, 5, 9, 2, 6];
        let (values, indices) = run_topk(&xs, 1);
        assert_eq!(values, vec![9]);
        assert_eq!(indices, vec![5]);
    }

    #[test]
    fn topk_k3_sorted_descending() {
        let xs = [3i64, 1, 4, 1, 5, 9, 2, 6];
        let (values, indices) = run_topk(&xs, 3);
        assert_eq!(values, vec![9, 6, 5]);
        assert_eq!(indices, vec![5, 7, 4]);
    }

    #[test]
    fn topk_with_negatives() {
        let xs = [-5i64, 3, -1, 7, 0];
        let (values, indices) = run_topk(&xs, 2);
        assert_eq!(values, vec![7, 3]);
        assert_eq!(indices, vec![3, 1]);
    }

    #[test]
    fn topk_k_equals_n_is_full_sort() {
        let xs = [2i64, 4, 1, 3];
        let (values, _indices) = run_topk(&xs, 4);
        assert_eq!(values, vec![4, 3, 2, 1]);
    }

    #[test]
    fn topk_too_few_inputs_returns_error() {
        let mut outputs = [F::zero(); 6]; // 2*3 = 6
        let inputs = vec![field(1), field(2), F::from_u256(U256::from(1u64 << 18))];
        assert!(topk_hint::<F>(&inputs, &mut outputs).is_err());
    }
}
