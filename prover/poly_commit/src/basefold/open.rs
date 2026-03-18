use arith::{ExtensionField, FFTField, Field, SimdField};
use gkr_engine::Transcript;
use tree::{Tree, LEAF_BYTES};

use super::encoding::{compute_twiddle_coeffs, fold_codeword, fold_codeword_first_round};
use super::types::{
    BasefoldOpening, FriQueryProof, FriRoundProof, SumcheckRoundMessage, BASEFOLD_NUM_QUERIES,
    RATE_LOG,
};

fn min_tree_elems<EvalF: Field>() -> usize {
    let elems_per_leaf = LEAF_BYTES / EvalF::SIZE;
    (2 * elems_per_leaf).max(4)
}

fn compute_sumcheck_round<F, EvalF>(
    f_table: &[EvalF],
    eq_table: &[EvalF],
) -> SumcheckRoundMessage<EvalF>
where
    F: Field,
    EvalF: ExtensionField<BaseField = F>,
{
    let half = f_table.len() / 2;
    let mut eval_at_1 = EvalF::ZERO;
    let mut eval_at_2 = EvalF::ZERO;

    for i in 0..half {
        let f0 = f_table[2 * i];
        let f1 = f_table[2 * i + 1];
        let eq0 = eq_table[2 * i];
        let eq1 = eq_table[2 * i + 1];

        eval_at_1 += f1 * eq1;

        let f_at_2 = f1 + f1 - f0;
        let eq_at_2 = eq1 + eq1 - eq0;
        eval_at_2 += f_at_2 * eq_at_2;
    }

    SumcheckRoundMessage {
        eval_at_1,
        eval_at_2,
    }
}

fn fold_table<EvalF: Field>(table: &[EvalF], challenge: EvalF) -> Vec<EvalF> {
    let half = table.len() / 2;
    let one_minus_r = EvalF::ONE - challenge;
    (0..half)
        .map(|i| table[2 * i] * one_minus_r + table[2 * i + 1] * challenge)
        .collect()
}

fn build_eq_table<EvalF: Field>(eval_point: &[EvalF]) -> Vec<EvalF> {
    let n = eval_point.len();
    let size = 1usize << n;
    let mut eq = vec![EvalF::ONE; size];
    for (k, &u_k) in eval_point.iter().enumerate() {
        let stride = 1 << k;
        for i in (0..size).rev() {
            let bit = (i >> k) & 1;
            if bit == 1 {
                eq[i] = eq[i - stride] * u_k;
            } else {
                eq[i] = eq[i] * (EvalF::ONE - u_k);
            }
        }
    }
    eq
}

#[allow(clippy::too_many_lines)]
pub fn basefold_open<F, EvalF>(
    evals: &[F],
    initial_codeword: &[F],
    initial_tree: &Tree,
    num_vars: usize,
    eval_point: &[EvalF],
    transcript: &mut impl Transcript,
) -> BasefoldOpening<EvalF>
where
    F: FFTField + SimdField<Scalar = F>,
    EvalF: ExtensionField<BaseField = F> + FFTField + SimdField<Scalar = EvalF>,
{
    assert_eq!(eval_point.len(), num_vars);
    assert_eq!(evals.len(), 1 << num_vars);

    let min_elems = min_tree_elems::<EvalF>();
    let codeword_log = num_vars + RATE_LOG;

    let ext_evals: Vec<EvalF> = evals.iter().map(|&x| EvalF::from(x)).collect();
    let mut f_table = ext_evals;
    let mut eq_table = build_eq_table(eval_point);

    let mut round_commitments = Vec::with_capacity(num_vars);
    let mut round_trees: Vec<Tree> = Vec::with_capacity(num_vars);
    let mut round_codewords: Vec<Vec<EvalF>> = Vec::with_capacity(num_vars);
    let mut sumcheck_messages = Vec::with_capacity(num_vars);

    let mut current_codeword_ext: Option<Vec<EvalF>> = None;
    let mut committed_rounds = 0;
    let mut final_poly_captured = false;
    let mut final_poly: Vec<EvalF> = Vec::new();

    for round in 0..num_vars {
        let sc_msg = compute_sumcheck_round::<F, EvalF>(&f_table, &eq_table);

        transcript.append_field_element(&sc_msg.eval_at_1);
        transcript.append_field_element(&sc_msg.eval_at_2);
        sumcheck_messages.push(sc_msg);

        let challenge: EvalF = transcript.generate_field_element();

        let level = codeword_log - 1 - round;
        let twiddle_coeffs = compute_twiddle_coeffs::<F>(level);

        let folded_codeword = if round == 0 {
            fold_codeword_first_round(initial_codeword, challenge, &twiddle_coeffs)
        } else {
            fold_codeword(
                current_codeword_ext.as_ref().unwrap(),
                challenge,
                &twiddle_coeffs,
            )
        };

        f_table = fold_table(&f_table, challenge);
        eq_table = fold_table(&eq_table, challenge);

        if folded_codeword.len() >= min_elems {
            let tree = Tree::compact_new_with_field_elems::<EvalF, EvalF>(folded_codeword.clone());
            let root = tree.root();
            transcript.append_commitment(root.as_bytes());
            round_commitments.push(root);
            round_trees.push(tree);
            round_codewords.push(folded_codeword.clone());
            committed_rounds = round + 1;
        } else if !final_poly_captured {
            final_poly = folded_codeword.clone();
            final_poly_captured = true;
            for f in &folded_codeword {
                transcript.append_field_element(f);
            }
        }

        current_codeword_ext = Some(folded_codeword);
    }

    if !final_poly_captured {
        final_poly = current_codeword_ext.unwrap_or_default();
    }

    let initial_len = 1 << codeword_log;
    let query_indices = generate_query_indices(transcript, initial_len);

    let base_elems_per_leaf = LEAF_BYTES / F::SIZE;
    let ext_elems_per_leaf = LEAF_BYTES / EvalF::SIZE;

    let mut query_proofs = Vec::with_capacity(query_indices.len());

    for &query_idx in &query_indices {
        let even_idx = (query_idx / 2) * 2;
        let odd_idx = even_idx + 1;

        let even_leaf_idx = even_idx / base_elems_per_leaf;
        let odd_leaf_idx = odd_idx / base_elems_per_leaf;

        let initial_leaf_proof = initial_tree.index_query(even_leaf_idx);
        let initial_sibling_proof = if odd_leaf_idx != even_leaf_idx {
            initial_tree.index_query(odd_leaf_idx)
        } else {
            initial_leaf_proof.clone()
        };

        let mut round_proofs = Vec::with_capacity(committed_rounds);
        let mut result_idx = query_idx / 2;

        for round in 0..committed_rounds {
            let re = &round_codewords[round];
            let re_len = re.len();

            let pair_even = (result_idx / 2) * 2;
            let pair_odd = pair_even + 1;

            let result_leaf = pair_even / ext_elems_per_leaf;
            let result_proof = round_trees[round].index_query(result_leaf);
            let result_start = result_leaf * ext_elems_per_leaf;
            let result_end = (result_start + ext_elems_per_leaf).min(re_len);
            let result_values = re[result_start..result_end].to_vec();

            let partner_leaf = pair_odd / ext_elems_per_leaf;
            let (partner_proof, partner_values) = if partner_leaf != result_leaf {
                let proof = round_trees[round].index_query(partner_leaf);
                let partner_start = partner_leaf * ext_elems_per_leaf;
                let partner_end = (partner_start + ext_elems_per_leaf).min(re_len);
                (proof, re[partner_start..partner_end].to_vec())
            } else {
                (result_proof.clone(), result_values.clone())
            };

            round_proofs.push(FriRoundProof {
                leaf_proof: result_proof,
                sibling_proof: partner_proof,
                leaf_values: result_values,
                sibling_values: partner_values,
            });

            result_idx /= 2;
        }

        query_proofs.push(FriQueryProof {
            initial_leaf_proof,
            initial_sibling_proof,
            round_proofs,
        });
    }

    BasefoldOpening {
        round_commitments,
        sumcheck_messages,
        final_poly,
        query_proofs,
    }
}

fn generate_query_indices(transcript: &mut impl Transcript, len: usize) -> Vec<usize> {
    assert!(len > 0, "generate_query_indices: len must be > 0");
    let target = BASEFOLD_NUM_QUERIES.min(len);
    let mut seen = std::collections::HashSet::with_capacity(target);
    let mut indices = Vec::with_capacity(target);
    while indices.len() < target {
        let batch = transcript.generate_usize_vector(target - indices.len());
        for x in batch {
            let idx = x % len;
            if seen.insert(idx) {
                indices.push(idx);
                if indices.len() == target {
                    break;
                }
            }
        }
    }
    indices
}
