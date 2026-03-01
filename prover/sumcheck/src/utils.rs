use arith::{ExtensionField, SimdField};
use gkr_engine::{MPIEngine, Transcript};

#[inline(always)]
/// Input
/// - a SIMD field, denoted by p := [p0, ... pn]
/// - a vector of coefficients, denoted by coef := [c0, ... cn]
///
/// Output
/// - p0 * c0 + ... + pn * cn
pub fn unpack_and_combine<F: SimdField>(p: &F, coef: &[F::Scalar]) -> F::Scalar {
    if coef.len() >= F::PACK_SIZE {
        let coef_packed = F::pack(&coef[..F::PACK_SIZE]);
        return (coef_packed * p).horizontal_sum();
    }

    let p_unpacked = p.unpack();
    p_unpacked
        .into_iter()
        .zip(coef)
        .map(|(p_i, coef_i)| p_i * coef_i)
        .sum()
}

/// Transcript IO between sumcheck steps
#[inline]
pub fn transcript_io<F, T>(mpi_config: &impl MPIEngine, ps: &[F], transcript: &mut T) -> F
where
    F: ExtensionField,
    T: Transcript,
{
    // 3 for x, y; 4 for simd var; 7 for pow5, 9 for pow7
    assert!(
        ps.len() == 3 || ps.len() == 4 || ps.len() == 7 || ps.len() == 9,
        "Unexpected polynomial size"
    );
    for p in ps {
        transcript.append_field_element(p);
    }
    let mut r = transcript.generate_field_element::<F>();
    mpi_config.root_broadcast_f(&mut r);
    r
}
