use arith::{ExtensionField, Field, SimdField};
use gf2::GF2;
use gkr_engine::Transcript;
use polynomials::{EqPolynomial, MultilinearExtension, RefMultiLinearPoly};

use crate::{
    orion::{
        utils::{
            commit_encoded, lut_open_linear_combine, orion_mt_openings, simd_open_linear_combine,
        },
        OrionCommitment, OrionProof, OrionResult, OrionSRS, OrionScratchPad,
    },
    traits::TensorCodeIOPPCS,
    PCS_SOUNDNESS_BITS,
};

#[inline(always)]
pub fn orion_commit_simd_field<F, SimdF, ComPackF>(
    pk: &OrionSRS,
    poly: &impl MultilinearExtension<SimdF>,
    scratch_pad: &mut OrionScratchPad,
) -> OrionResult<OrionCommitment>
where
    F: Field,
    SimdF: SimdField<Scalar = F>,
    ComPackF: SimdField<Scalar = F>,
{
    let packed_evals_ref = unsafe {
        let relative_pack_size = ComPackF::PACK_SIZE / SimdF::PACK_SIZE;
        assert_eq!(ComPackF::PACK_SIZE % SimdF::PACK_SIZE, 0);

        let ptr = poly.hypercube_basis_ref().as_ptr();
        let len = poly.hypercube_size() / relative_pack_size;
        assert_eq!(len * relative_pack_size, poly.hypercube_size());

        std::slice::from_raw_parts(ptr as *const ComPackF, len)
    };

    commit_encoded(pk, packed_evals_ref, scratch_pad)
}

#[inline(always)]
pub fn orion_open_simd_field<F, SimdF, EvalF, ComPackF>(
    pk: &OrionSRS,
    poly: &impl MultilinearExtension<SimdF>,
    point: &[EvalF],
    transcript: &mut impl Transcript,
    scratch_pad: &OrionScratchPad,
) -> (EvalF, OrionProof<EvalF>)
where
    F: Field,
    SimdF: SimdField<Scalar = F>,
    EvalF: ExtensionField<BaseField = F>,
    ComPackF: SimdField<Scalar = F>,
{
    let msg_size = pk.message_len();

    let num_vars_in_com_simd = ComPackF::PACK_SIZE.ilog2() as usize;
    let num_vars_in_msg = msg_size.ilog2() as usize;

    // NOTE: pre-compute the eq linear combine coeffs for linear combination
    let eq_col_coeffs = {
        let mut eq_vars = point[..num_vars_in_com_simd].to_vec();
        eq_vars.extend_from_slice(&point[num_vars_in_com_simd + num_vars_in_msg..]);
        EqPolynomial::build_eq_x_r(&eq_vars)
    };

    // NOTE: pre-declare the spaces for returning evaluation and proximity queries
    let mut eval_row = vec![EvalF::ZERO; msg_size];

    let proximity_test_num = pk.proximity_repetitions::<EvalF>(PCS_SOUNDNESS_BITS);
    let mut proximity_rows = vec![vec![EvalF::ZERO; msg_size]; proximity_test_num];

    let random_col_coeffs: Vec<_> = (0..proximity_test_num)
        .map(|_| {
            let rand = transcript.generate_field_elements::<EvalF>(point.len() - num_vars_in_msg);
            EqPolynomial::build_eq_x_r(&rand)
        })
        .collect();

    match F::NAME {
        GF2::NAME => lut_open_linear_combine(
            ComPackF::PACK_SIZE,
            poly.hypercube_basis_ref(),
            &eq_col_coeffs,
            &mut eval_row,
            &random_col_coeffs,
            &mut proximity_rows,
        ),
        _ => simd_open_linear_combine(
            ComPackF::PACK_SIZE,
            poly.hypercube_basis_ref(),
            &eq_col_coeffs,
            &mut eval_row,
            &random_col_coeffs,
            &mut proximity_rows,
        ),
    }

    // NOTE: working on evaluation response, evaluate the rest of the response
    let mut scratch = vec![EvalF::ZERO; msg_size];
    let eval = RefMultiLinearPoly::from_ref(&eval_row).evaluate_with_buffer(
        &point[num_vars_in_com_simd..num_vars_in_com_simd + num_vars_in_msg],
        &mut scratch,
    );
    drop(scratch);

    // NOTE: MT opening for point queries
    let query_openings = orion_mt_openings(pk, transcript, scratch_pad);

    (
        eval,
        OrionProof {
            eval_row,
            proximity_rows,
            query_openings,
            merkle_cap: scratch_pad.merkle_cap.clone(),
        },
    )
}
