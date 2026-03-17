use arith::{ExtensionField, FFTField, SimdField};
use gkr_engine::Transcript;
use tree::{Tree, LEAF_BYTES};

use super::encoding::fold_evals;
use super::types::{BasefoldOpening, FriQueryProof, FriRoundProof, BASEFOLD_NUM_QUERIES};

fn min_tree_elems<EvalF: arith::Field>() -> usize {
    let elems_per_leaf = LEAF_BYTES / EvalF::SIZE;
    (2 * elems_per_leaf).max(4)
}

pub fn basefold_open<F, EvalF>(
    evals: &[F],
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

    let mut round_commitments = Vec::with_capacity(num_vars);
    let mut round_trees: Vec<Tree> = Vec::with_capacity(num_vars);
    let mut round_evals: Vec<Vec<EvalF>> = Vec::with_capacity(num_vars);

    let mut current: Vec<EvalF> = evals.iter().map(|&x| EvalF::from(x)).collect();
    let mut committed_rounds = 0;
    let mut final_poly_captured = false;
    let mut final_poly: Vec<EvalF> = Vec::new();

    for round in 0..num_vars {
        let challenge = eval_point[num_vars - 1 - round];
        transcript.append_field_element(&challenge);

        let folded = fold_evals(&current, challenge);

        if folded.len() >= min_elems {
            let tree = Tree::compact_new_with_field_elems::<EvalF, EvalF>(folded.clone());
            let root = tree.root();
            transcript.append_commitment(root.as_bytes());
            round_commitments.push(root);
            round_trees.push(tree);
            round_evals.push(folded.clone());
            committed_rounds = round + 1;
        } else if !final_poly_captured {
            final_poly = folded.clone();
            final_poly_captured = true;
            for f in &folded {
                transcript.append_field_element(f);
            }
        }

        current = folded;
    }

    if !final_poly_captured {
        final_poly = current;
    }

    let initial_len = 1 << num_vars;
    let query_indices = generate_query_indices(transcript, initial_len);

    let base_elems_per_leaf = LEAF_BYTES / F::SIZE;
    let ext_elems_per_leaf = LEAF_BYTES / EvalF::SIZE;

    let mut query_proofs = Vec::with_capacity(query_indices.len());

    for &query_idx in &query_indices {
        let half = initial_len / 2;
        let lo_idx = query_idx % half;
        let hi_idx = lo_idx + half;

        let lo_leaf_idx = lo_idx / base_elems_per_leaf;
        let hi_leaf_idx = hi_idx / base_elems_per_leaf;

        let initial_leaf_proof = initial_tree.index_query(lo_leaf_idx);
        let initial_sibling_proof = initial_tree.index_query(hi_leaf_idx);

        let mut round_proofs = Vec::with_capacity(committed_rounds);
        let mut result_idx = lo_idx;

        for round in 0..committed_rounds {
            let re = &round_evals[round];
            let re_len = re.len();

            let result_leaf = result_idx / ext_elems_per_leaf;
            let result_proof = round_trees[round].index_query(result_leaf);
            let result_start = result_leaf * ext_elems_per_leaf;
            let result_end = result_start + ext_elems_per_leaf;
            let result_values = re[result_start..result_end].to_vec();

            let next_half = re_len / 2;
            let next_lo = result_idx % next_half;
            let next_hi = next_lo + next_half;
            let partner_idx = if result_idx < next_half {
                next_hi
            } else {
                next_lo
            };

            let partner_leaf = partner_idx / ext_elems_per_leaf;
            let partner_proof = round_trees[round].index_query(partner_leaf);
            let partner_start = partner_leaf * ext_elems_per_leaf;
            let partner_end = partner_start + ext_elems_per_leaf;
            let partner_values = re[partner_start..partner_end].to_vec();

            round_proofs.push(FriRoundProof {
                leaf_proof: result_proof,
                sibling_proof: partner_proof,
                leaf_values: result_values,
                sibling_values: partner_values,
            });

            result_idx = next_lo;
        }

        query_proofs.push(FriQueryProof {
            initial_leaf_proof,
            initial_sibling_proof,
            round_proofs,
        });
    }

    BasefoldOpening {
        round_commitments,
        final_poly,
        query_proofs,
    }
}

fn generate_query_indices(transcript: &mut impl Transcript, len: usize) -> Vec<usize> {
    let raw = transcript.generate_usize_vector(BASEFOLD_NUM_QUERIES);
    raw.into_iter().map(|x| x % len).collect()
}
