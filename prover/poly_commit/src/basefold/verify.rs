use arith::{ExtensionField, FFTField, Field};
use gkr_engine::Transcript;
use tree::LEAF_BYTES;

use super::encoding::fold_evals;
use super::types::{BasefoldCommitment, BasefoldOpening, BASEFOLD_NUM_QUERIES};

fn min_tree_elems<EvalF: Field>() -> usize {
    let elems_per_leaf = LEAF_BYTES / EvalF::SIZE;
    (2 * elems_per_leaf).max(4)
}

pub fn basefold_verify<F, EvalF>(
    commitment: &BasefoldCommitment,
    eval_point: &[EvalF],
    claimed_eval: EvalF,
    opening: &BasefoldOpening<EvalF>,
    transcript: &mut impl Transcript,
) -> bool
where
    F: FFTField,
    EvalF: ExtensionField<BaseField = F> + FFTField,
{
    let num_vars = commitment.num_vars;
    if eval_point.len() != num_vars {
        return false;
    }

    let min_elems = min_tree_elems::<EvalF>();
    let committed_rounds = opening.round_commitments.len();

    let remaining_rounds = num_vars - committed_rounds;
    let final_poly_len = 1usize << (num_vars - committed_rounds);
    let expected_final_poly_len = final_poly_len / 2;

    if opening.final_poly.len() != expected_final_poly_len {
        return false;
    }

    let mut commit_idx = 0;
    let mut saw_final = false;
    for round in 0..num_vars {
        let challenge = eval_point[num_vars - 1 - round];
        transcript.append_field_element(&challenge);

        let folded_size = 1usize << (num_vars - 1 - round);
        if folded_size >= min_elems && commit_idx < committed_rounds {
            transcript.append_commitment(opening.round_commitments[commit_idx].as_bytes());
            commit_idx += 1;
        } else if !saw_final {
            for f in &opening.final_poly {
                transcript.append_field_element(f);
            }
            saw_final = true;
        }
    }

    if commit_idx != committed_rounds {
        return false;
    }

    let mut remaining_poly = opening.final_poly.clone();
    for r in 1..remaining_rounds {
        let challenge = eval_point[num_vars - 1 - (committed_rounds + r)];
        remaining_poly = fold_evals(&remaining_poly, challenge);
    }

    if remaining_poly.len() != 1 || remaining_poly[0] != claimed_eval {
        return false;
    }

    let initial_len = 1 << num_vars;
    let query_indices = generate_query_indices(transcript, initial_len);

    if opening.query_proofs.len() != query_indices.len() {
        return false;
    }

    let base_elems_per_leaf = LEAF_BYTES / F::SIZE;
    let ext_elems_per_leaf = LEAF_BYTES / EvalF::SIZE;

    for (qi, &query_idx) in query_indices.iter().enumerate() {
        let qp = &opening.query_proofs[qi];

        if !qp.initial_leaf_proof.verify(&commitment.root) {
            return false;
        }
        if !qp.initial_sibling_proof.verify(&commitment.root) {
            return false;
        }

        let half = initial_len / 2;
        let lo_idx = query_idx % half;
        let hi_idx = lo_idx + half;

        let lo_leaf_idx = lo_idx / base_elems_per_leaf;
        let hi_leaf_idx = hi_idx / base_elems_per_leaf;

        if qp.initial_leaf_proof.index != lo_leaf_idx {
            return false;
        }
        if qp.initial_sibling_proof.index != hi_leaf_idx {
            return false;
        }

        let lo_offset = lo_idx % base_elems_per_leaf;
        let hi_offset = hi_idx % base_elems_per_leaf;

        let lo_val: EvalF =
            extract_base_from_leaf::<F, EvalF>(&qp.initial_leaf_proof.leaf.data, lo_offset);
        let hi_val: EvalF =
            extract_base_from_leaf::<F, EvalF>(&qp.initial_sibling_proof.leaf.data, hi_offset);

        let mut current_lo_val = lo_val;
        let mut current_hi_val = hi_val;
        let mut result_idx = lo_idx;
        let mut current_size = initial_len;

        for round in 0..committed_rounds {
            let challenge = eval_point[num_vars - 1 - round];
            let one_minus_r = EvalF::ONE - challenge;

            let expected_folded = current_lo_val * one_minus_r + current_hi_val * challenge;

            current_size /= 2;

            let rp = &qp.round_proofs[round];

            if !rp.leaf_proof.verify(&opening.round_commitments[round]) {
                return false;
            }

            let result_leaf = result_idx / ext_elems_per_leaf;
            let result_offset = result_idx % ext_elems_per_leaf;

            if rp.leaf_proof.index != result_leaf {
                return false;
            }

            if !verify_leaf_values::<EvalF>(&rp.leaf_proof.leaf.data, &rp.leaf_values) {
                return false;
            }

            if result_offset >= rp.leaf_values.len() {
                return false;
            }

            let actual_folded = rp.leaf_values[result_offset];
            if actual_folded != expected_folded {
                return false;
            }

            if !rp.sibling_proof.verify(&opening.round_commitments[round]) {
                return false;
            }

            if !verify_leaf_values::<EvalF>(&rp.sibling_proof.leaf.data, &rp.sibling_values) {
                return false;
            }

            let next_half = current_size / 2;
            let next_lo = result_idx % next_half;
            let next_hi = next_lo + next_half;
            let partner_idx = if result_idx < next_half {
                next_hi
            } else {
                next_lo
            };

            let partner_leaf = partner_idx / ext_elems_per_leaf;
            if rp.sibling_proof.index != partner_leaf {
                return false;
            }

            let next_lo_offset = next_lo % ext_elems_per_leaf;
            let next_hi_offset = next_hi % ext_elems_per_leaf;

            if result_idx < next_half {
                current_lo_val = rp.leaf_values[next_lo_offset];
                current_hi_val = rp.sibling_values[next_hi_offset];
            } else {
                current_lo_val = rp.sibling_values[next_lo_offset];
                current_hi_val = rp.leaf_values[next_hi_offset];
            }

            result_idx = next_lo;
        }

        let challenge = eval_point[num_vars - 1 - committed_rounds];
        let one_minus_r = EvalF::ONE - challenge;
        let expected_first_non_committed =
            current_lo_val * one_minus_r + current_hi_val * challenge;

        if result_idx >= opening.final_poly.len() {
            return false;
        }

        if opening.final_poly[result_idx] != expected_first_non_committed {
            return false;
        }
    }

    true
}

fn extract_base_from_leaf<F: Field, EvalF: ExtensionField<BaseField = F>>(
    leaf_data: &[u8],
    offset: usize,
) -> EvalF {
    let start = offset * F::SIZE;
    let end = start + F::SIZE;
    let base_val = F::deserialize_from(&leaf_data[start..end]).unwrap();
    EvalF::from(base_val)
}

fn verify_leaf_values<EvalF: Field>(leaf_data: &[u8], values: &[EvalF]) -> bool {
    let elems_per_leaf = LEAF_BYTES / EvalF::SIZE;
    if values.len() != elems_per_leaf {
        return false;
    }
    for (i, v) in values.iter().enumerate() {
        let start = i * EvalF::SIZE;
        let end = start + EvalF::SIZE;
        let deserialized = EvalF::deserialize_from(&leaf_data[start..end]).unwrap();
        if deserialized != *v {
            return false;
        }
    }
    true
}

fn generate_query_indices(transcript: &mut impl Transcript, len: usize) -> Vec<usize> {
    let raw = transcript.generate_usize_vector(BASEFOLD_NUM_QUERIES);
    raw.into_iter().map(|x| x % len).collect()
}
