use arith::{FFTField, SimdField};
use tree::Tree;

use super::encoding::rs_encode;
use super::types::BasefoldCommitment;

pub fn basefold_commit<F: FFTField + SimdField<Scalar = F>>(
    evals: &[F],
) -> (BasefoldCommitment, Tree, Vec<F>) {
    let num_vars = evals.len().ilog2() as usize;
    assert_eq!(evals.len(), 1 << num_vars);

    let codeword = rs_encode(evals);
    let tree = Tree::compact_new_with_field_elems::<F, F>(codeword.clone());
    let root = tree.root();
    (BasefoldCommitment { root, num_vars }, tree, codeword)
}
