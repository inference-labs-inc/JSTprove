use arith::{ExtensionField, Field, SimdField};
use gf2::GF2;
use gkr_engine::{MPIEngine, Transcript};
use polynomials::{EqPolynomial, MultilinearExtension};

use crate::{
    orion::{
        mpi_utils::{mpi_commit_encoded, orion_mpi_mt_openings},
        utils::{lut_open_linear_combine, simd_open_linear_combine},
        OrionCommitment, OrionProof, OrionResult, OrionSRS, OrionScratchPad,
    },
    traits::TensorCodeIOPPCS,
    PCS_SOUNDNESS_BITS,
};

#[inline(always)]
pub fn orion_mpi_commit_simd_field<F, SimdF, ComPackF>(
    mpi_engine: &impl MPIEngine,
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

    mpi_commit_encoded(mpi_engine, pk, packed_evals_ref, scratch_pad)
}

#[inline(always)]
pub fn orion_mpi_open_simd_field<F, SimdF, EvalF, ComPackF>(
    mpi_engine: &impl MPIEngine,
    pk: &OrionSRS,
    poly: &impl MultilinearExtension<SimdF>,
    point: &[EvalF],
    mpi_point: &[EvalF],
    transcript: &mut impl Transcript,
    scratch_pad: &OrionScratchPad,
) -> Option<OrionProof<EvalF>>
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
        let mut coeffs = EqPolynomial::build_eq_x_r(&eq_vars);
        let mpi_weight = EqPolynomial::ith_eq_vec_elem(mpi_point, mpi_engine.world_rank());
        coeffs.iter_mut().for_each(|c| *c *= mpi_weight);
        coeffs
    };

    // NOTE: pre-declare the spaces for returning evaluation and proximity queries
    let mut eval_row = vec![EvalF::ZERO; msg_size];

    let proximity_test_num = pk.proximity_repetitions::<EvalF>(PCS_SOUNDNESS_BITS);
    let mut proximity_rows = vec![vec![EvalF::ZERO; msg_size]; proximity_test_num];

    // NOTE: draw randomness from transcript with log random complexity
    let num_of_local_random_vars = point.len() - num_vars_in_msg;
    let local_random_coeffs: Vec<_> = (0..proximity_test_num)
        .map(|_| {
            let local_rand = transcript.generate_field_elements::<EvalF>(num_of_local_random_vars);
            let mpi_rand = transcript.generate_field_elements::<EvalF>(mpi_point.len());
            let mut coeffs = EqPolynomial::build_eq_x_r(&local_rand);
            let mpi_weight = EqPolynomial::ith_eq_vec_elem(&mpi_rand, mpi_engine.world_rank());
            coeffs.iter_mut().for_each(|c| *c *= mpi_weight);
            coeffs
        })
        .collect();

    match F::NAME {
        GF2::NAME => lut_open_linear_combine(
            ComPackF::PACK_SIZE,
            poly.hypercube_basis_ref(),
            &eq_col_coeffs,
            &mut eval_row,
            &local_random_coeffs,
            &mut proximity_rows,
        ),
        _ => simd_open_linear_combine(
            ComPackF::PACK_SIZE,
            poly.hypercube_basis_ref(),
            &eq_col_coeffs,
            &mut eval_row,
            &local_random_coeffs,
            &mut proximity_rows,
        ),
    }

    // NOTE: MPI sum up local weighed rows
    eval_row = mpi_engine.sum_vec(&eval_row);
    proximity_rows = proximity_rows
        .iter()
        .map(|r| mpi_engine.sum_vec(r))
        .collect();

    // NOTE: MT opening for point queries
    let query_openings = orion_mpi_mt_openings(mpi_engine, pk, scratch_pad, transcript);

    if !mpi_engine.is_root() {
        return None;
    }

    OrionProof {
        eval_row,
        proximity_rows,
        query_openings: query_openings.unwrap(),
        merkle_cap: scratch_pad.merkle_cap.clone(),
    }
    .into()
}
