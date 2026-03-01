use std::iter;

use arith::{ExtensionField, Field, SimdField};
use gf2::GF2;
use gkr_engine::Transcript;
use itertools::{chain, izip};
use polynomials::{EqPolynomial, MultilinearExtension, RefMultiLinearPoly};
use tree::LEAF_BYTES;

use crate::{
    orion::{
        utils::{lut_verify_alphabet_check, orion_mt_verify, simd_verify_alphabet_check},
        OrionCommitment, OrionProof, OrionSRS,
    },
    traits::TensorCodeIOPPCS,
    PCS_SOUNDNESS_BITS,
};

#[inline(always)]
pub fn orion_verify<F, SimdF, EvalF, ComPackF>(
    vk: &OrionSRS,
    commitment: &OrionCommitment,
    point: &[EvalF],
    mpi_point: &[EvalF],
    evaluation: EvalF,
    transcript: &mut impl Transcript,
    proof: &OrionProof<EvalF>,
) -> bool
where
    F: Field,
    SimdF: SimdField<Scalar = F>,
    EvalF: ExtensionField<BaseField = F>,
    ComPackF: SimdField<Scalar = F>,
{
    let world_size = 1 << mpi_point.len();
    let msg_size = vk.message_len();

    let num_vars_in_com_simd = ComPackF::PACK_SIZE.ilog2() as usize;
    let num_vars_in_msg = msg_size.ilog2() as usize;

    // NOTE: working on evaluation response, evaluate the rest of the response
    let mut scratch = vec![EvalF::ZERO; msg_size];
    let final_eval = RefMultiLinearPoly::from_ref(&proof.eval_row).evaluate_with_buffer(
        &point[num_vars_in_com_simd..num_vars_in_com_simd + num_vars_in_msg],
        &mut scratch,
    );

    if final_eval != evaluation {
        return false;
    }

    // NOTE: working on proximity responses, draw random linear combinations
    // then draw query points from fiat shamir transcripts
    let proximity_reps = vk.proximity_repetitions::<EvalF>(PCS_SOUNDNESS_BITS);
    let random_linear_combinations: Vec<_> = (0..proximity_reps)
        .map(|_| {
            let num_vars = point.len() - num_vars_in_msg + mpi_point.len();
            let rand = transcript.generate_field_elements::<EvalF>(num_vars);
            EqPolynomial::build_eq_x_r(&rand)
        })
        .collect();

    let query_num = vk.query_complexity(PCS_SOUNDNESS_BITS);
    let query_indices = transcript.generate_usize_vector(query_num);

    // NOTE: check consistency in MT in the opening trees and against the commitment tree
    {
        if proof.merkle_cap.len() != world_size {
            return false;
        }

        let actual_commitment = if world_size > 1 {
            let height = 1 + mpi_point.len();
            let internal = tree::Tree::new_with_leaf_nodes(&proof.merkle_cap, height as u32);
            internal[0]
        } else {
            proof.merkle_cap[0]
        };

        if actual_commitment != *commitment {
            return false;
        }
    }

    if !orion_mt_verify(vk, &query_indices, &proof.query_openings, &proof.merkle_cap) {
        return false;
    }

    // NOTE: prepare the interleaved alphabets from the MT paths
    let num_simd_elems_per_leaf = vk.num_leaves_per_mt_query() * LEAF_BYTES / SimdF::SIZE;
    let packed_interleaved_alphabets: Vec<Vec<SimdF>> = proof
        .query_openings
        .iter()
        .map(|c| unsafe {
            let ptr = c.leaves.as_ptr();
            std::slice::from_raw_parts(ptr as *const SimdF, num_simd_elems_per_leaf).to_vec()
        })
        .collect();

    let eq_col_coeffs = {
        let mut eq_vars = point[..num_vars_in_com_simd].to_vec();
        eq_vars.extend_from_slice(&point[num_vars_in_com_simd + num_vars_in_msg..]);
        eq_vars.extend_from_slice(mpi_point);
        EqPolynomial::build_eq_x_r(&eq_vars)
    };

    chain!(
        izip!(&random_linear_combinations, &proof.proximity_rows),
        iter::once((&eq_col_coeffs, &proof.eval_row))
    )
    .all(|(rl, msg)| {
        let codeword = match vk.code_instance.encode(msg) {
            Ok(c) => c,
            _ => return false,
        };

        match F::NAME {
            GF2::NAME => lut_verify_alphabet_check(
                &codeword,
                rl,
                &query_indices,
                &packed_interleaved_alphabets,
            ),
            _ => simd_verify_alphabet_check(
                &codeword,
                rl,
                &query_indices,
                &packed_interleaved_alphabets,
            ),
        }
    })
}
