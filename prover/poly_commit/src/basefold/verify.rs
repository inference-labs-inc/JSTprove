use arith::{ExtensionField, FFTField, Field};
use gkr_engine::Transcript;
use tree::LEAF_BYTES;

use super::encoding::{codeword_fold_single, verifier_folding_coeff};
use super::types::{BasefoldCommitment, BasefoldOpening, BASEFOLD_NUM_QUERIES, RATE_LOG};

fn min_tree_elems<EvalF: Field>() -> usize {
    let elems_per_leaf = LEAF_BYTES / EvalF::SIZE;
    (2 * elems_per_leaf).max(4)
}

fn extrapolate_degree2<EvalF: Field>(
    eval_at_0: EvalF,
    eval_at_1: EvalF,
    eval_at_2: EvalF,
    x: EvalF,
) -> EvalF {
    let two_inv = EvalF::INV_2;
    let a = (eval_at_2 - eval_at_1 - eval_at_1 + eval_at_0) * two_inv;
    let b = eval_at_1 - eval_at_0 - a;
    eval_at_0 + x * (b + x * a)
}

fn compute_eq_eval<EvalF: Field>(challenges: &[EvalF], eval_point: &[EvalF]) -> EvalF {
    assert_eq!(challenges.len(), eval_point.len());
    let mut result = EvalF::ONE;
    for (&alpha, &u) in challenges.iter().zip(eval_point.iter()) {
        result = result * ((EvalF::ONE - alpha) * (EvalF::ONE - u) + alpha * u);
    }
    result
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
    if eval_point.len() != num_vars || num_vars == 0 {
        return false;
    }

    let codeword_log = match num_vars.checked_add(RATE_LOG) {
        Some(cl) if cl < usize::BITS as usize => cl,
        _ => return false,
    };

    if opening.sumcheck_messages.len() != num_vars {
        return false;
    }

    let committed_rounds = opening.round_commitments.len();
    if committed_rounds >= num_vars {
        return false;
    }

    let min_elems = min_tree_elems::<EvalF>();

    if committed_rounds > 0 {
        let last_folded_size = 1usize << (codeword_log - committed_rounds);
        if last_folded_size < min_elems {
            return false;
        }
    }
    let next_folded_size = 1usize << (codeword_log - 1 - committed_rounds);
    if next_folded_size >= min_elems {
        return false;
    }

    let expected_final_poly_len = 1usize << (codeword_log - 1 - committed_rounds);
    if opening.final_poly.len() != expected_final_poly_len {
        return false;
    }

    let mut challenges = Vec::with_capacity(num_vars);
    let mut running_claim = claimed_eval;

    for round in 0..num_vars {
        let sc_msg = &opening.sumcheck_messages[round];

        transcript.append_field_element(&sc_msg.eval_at_1);
        transcript.append_field_element(&sc_msg.eval_at_2);

        let eval_at_0 = running_claim - sc_msg.eval_at_1;

        let challenge: EvalF = transcript.generate_field_element();
        challenges.push(challenge);

        running_claim =
            extrapolate_degree2(eval_at_0, sc_msg.eval_at_1, sc_msg.eval_at_2, challenge);

        let folded_size = 1usize << (codeword_log - 1 - round);
        if folded_size >= min_elems && round < committed_rounds {
            transcript.append_commitment(opening.round_commitments[round].as_bytes());
        } else if round == committed_rounds {
            for f in &opening.final_poly {
                transcript.append_field_element(f);
            }
        }
    }

    let mut remaining_codeword = opening.final_poly.clone();
    for r in (committed_rounds + 1)..num_vars {
        let level = codeword_log - 1 - r;
        let challenge = challenges[r];
        let half = remaining_codeword.len() / 2;
        if half == 0 {
            break;
        }
        let mut new_cw = Vec::with_capacity(half);
        for i in 0..half {
            let twiddle = verifier_folding_coeff::<F>(level, i);
            new_cw.push(codeword_fold_single(
                remaining_codeword[2 * i],
                remaining_codeword[2 * i + 1],
                challenge,
                twiddle,
            ));
        }
        remaining_codeword = new_cw;
    }

    if remaining_codeword.is_empty() {
        return false;
    }

    let fri_constant = remaining_codeword[0];
    for val in &remaining_codeword[1..] {
        if *val != fri_constant {
            return false;
        }
    }

    let eq_at_challenges = compute_eq_eval(&challenges, eval_point);

    if fri_constant * eq_at_challenges != running_claim {
        return false;
    }

    let initial_len = 1 << codeword_log;
    let query_indices = generate_query_indices(transcript, initial_len);

    if opening.query_proofs.len() != query_indices.len() {
        return false;
    }

    let base_elems_per_leaf = LEAF_BYTES / F::SIZE;
    let ext_elems_per_leaf = LEAF_BYTES / EvalF::SIZE;

    for (qi, &query_idx) in query_indices.iter().enumerate() {
        let qp = &opening.query_proofs[qi];

        if qp.round_proofs.len() != committed_rounds {
            return false;
        }

        if !qp.initial_leaf_proof.verify(&commitment.root) {
            return false;
        }
        if !qp.initial_sibling_proof.verify(&commitment.root) {
            return false;
        }

        let even_idx = (query_idx / 2) * 2;
        let odd_idx = even_idx + 1;

        let even_leaf_idx = even_idx / base_elems_per_leaf;
        let odd_leaf_idx = odd_idx / base_elems_per_leaf;

        if qp.initial_leaf_proof.index != even_leaf_idx {
            return false;
        }
        if qp.initial_sibling_proof.index != odd_leaf_idx {
            return false;
        }

        let even_offset = even_idx % base_elems_per_leaf;
        let odd_offset = odd_idx % base_elems_per_leaf;

        let left: EvalF =
            match extract_base_from_leaf::<F>(&qp.initial_leaf_proof.leaf.data, even_offset) {
                Some(v) => EvalF::from(v),
                None => return false,
            };
        let right: EvalF = if odd_leaf_idx == even_leaf_idx {
            match extract_base_from_leaf::<F>(&qp.initial_leaf_proof.leaf.data, odd_offset) {
                Some(v) => EvalF::from(v),
                None => return false,
            }
        } else {
            match extract_base_from_leaf::<F>(&qp.initial_sibling_proof.leaf.data, odd_offset) {
                Some(v) => EvalF::from(v),
                None => return false,
            }
        };

        let level = codeword_log - 1;
        let pair_idx = even_idx / 2;
        let twiddle = verifier_folding_coeff::<F>(level, pair_idx);
        let mut current_val = codeword_fold_single(left, right, challenges[0], twiddle);
        let mut result_idx = query_idx / 2;

        if committed_rounds == 0 {
            if opening.final_poly[result_idx] != current_val {
                return false;
            }
        }

        for round in 0..committed_rounds {
            let rp = &qp.round_proofs[round];

            if !rp.leaf_proof.verify(&opening.round_commitments[round]) {
                return false;
            }

            let result_leaf = result_idx / ext_elems_per_leaf;

            if rp.leaf_proof.index != result_leaf {
                return false;
            }

            if !verify_leaf_values::<EvalF>(&rp.leaf_proof.leaf.data, &rp.leaf_values) {
                return false;
            }

            let result_offset = result_idx % ext_elems_per_leaf;
            if result_offset >= rp.leaf_values.len() {
                return false;
            }
            let actual_folded = rp.leaf_values[result_offset];
            if actual_folded != current_val {
                return false;
            }

            let pair_even = (result_idx / 2) * 2;
            let pair_odd = pair_even + 1;

            let partner_leaf = pair_odd / ext_elems_per_leaf;

            if !rp.sibling_proof.verify(&opening.round_commitments[round]) {
                return false;
            }

            if rp.sibling_proof.index != partner_leaf {
                return false;
            }

            if !verify_leaf_values::<EvalF>(&rp.sibling_proof.leaf.data, &rp.sibling_values) {
                return false;
            }

            let lo_val = if result_idx == pair_even {
                rp.leaf_values[pair_even % ext_elems_per_leaf]
            } else {
                rp.sibling_values[pair_even % ext_elems_per_leaf]
            };
            let hi_val = if result_idx == pair_even {
                if partner_leaf == result_leaf {
                    rp.leaf_values[pair_odd % ext_elems_per_leaf]
                } else {
                    rp.sibling_values[pair_odd % ext_elems_per_leaf]
                }
            } else {
                rp.leaf_values[pair_odd % ext_elems_per_leaf]
            };

            let next_level = codeword_log - 2 - round;
            let next_pair_idx = pair_even / 2;
            let next_twiddle = verifier_folding_coeff::<F>(next_level, next_pair_idx);
            current_val = codeword_fold_single(lo_val, hi_val, challenges[round + 1], next_twiddle);

            result_idx /= 2;

            if round + 1 == committed_rounds {
                if opening.final_poly[result_idx] != current_val {
                    return false;
                }
            }
        }
    }

    true
}

fn extract_base_from_leaf<F: Field>(leaf_data: &[u8], offset: usize) -> Option<F> {
    let start = offset * F::SIZE;
    let end = start + F::SIZE;
    if end > leaf_data.len() {
        return None;
    }
    F::deserialize_from(&leaf_data[start..end]).ok()
}

fn verify_leaf_values<EvalF: Field>(leaf_data: &[u8], values: &[EvalF]) -> bool {
    let elems_per_leaf = LEAF_BYTES / EvalF::SIZE;
    if leaf_data.len() < LEAF_BYTES {
        return false;
    }
    if values.len() != elems_per_leaf {
        return false;
    }
    for (i, v) in values.iter().enumerate() {
        let start = i * EvalF::SIZE;
        let end = start + EvalF::SIZE;
        if end > leaf_data.len() {
            return false;
        }
        let deserialized = match EvalF::deserialize_from(&leaf_data[start..end]) {
            Ok(val) => val,
            Err(_) => return false,
        };
        if deserialized != *v {
            return false;
        }
    }
    true
}

fn generate_query_indices(transcript: &mut impl Transcript, len: usize) -> Vec<usize> {
    assert!(len > 0, "generate_query_indices requires len > 0");
    let raw = transcript.generate_usize_vector(BASEFOLD_NUM_QUERIES);
    let mut seen = std::collections::HashSet::with_capacity(raw.len());
    raw.into_iter()
        .map(|x| x % len)
        .filter(|idx| seen.insert(*idx))
        .collect()
}
