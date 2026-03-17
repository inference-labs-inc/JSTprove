use arith::{FFTField, SimdField};
use tree::Tree;

use super::types::BasefoldCommitment;

pub fn basefold_commit<F: FFTField + SimdField<Scalar = F>>(
    evals: &[F],
) -> (BasefoldCommitment, Tree) {
    let num_vars = evals.len().ilog2() as usize;
    assert_eq!(evals.len(), 1 << num_vars);
    let tree = Tree::compact_new_with_field_elems::<F, F>(evals.to_vec());
    let root = tree.root();
    (BasefoldCommitment { root, num_vars }, tree)
}
