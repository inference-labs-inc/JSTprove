use std::marker::PhantomData;

use arith::{ExtensionField, FFTField, Field, SimdField};
use gkr_engine::{
    ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, MPIEngine, PolynomialCommitmentType,
    StructuredReferenceString, Transcript,
};
use polynomials::{MultiLinearPoly, MultilinearExtension};
use sha2::{Digest, Sha256};
use tree::{Tree, LEAF_BYTES};

use super::parameters::{
    effective_rate_log, num_committed_rounds, whir_queries_for_committed_round,
    WHIR_FOLDING_FACTOR, WHIR_OOD_SAMPLES, WHIR_POW_BITS, WHIR_RATE_LOG,
};
use super::types::{WhirCommitment, WhirOpening, WhirRoundQueryProof, WhirSRS, WhirScratchPad};
use crate::basefold::encoding::{
    bit_reverse_slice, compute_twiddle_coeffs, fold_codeword, fold_codeword_first_round,
    rs_encode_with_rate,
};
use crate::basefold::SumcheckRoundMessage;
use crate::utils::{lift_expander_challenge_to_n_vars, lift_poly_to_n_vars};

pub struct WhirPCSForGKR<C: FieldEngine> {
    _phantom: PhantomData<C>,
}

fn prepare_base_evals<C: FieldEngine>(
    poly: &impl MultilinearExtension<C::SimdCircuitField>,
    params: usize,
) -> Vec<C::CircuitField>
where
    C::CircuitField: FFTField + SimdField<Scalar = C::CircuitField>,
{
    let local_poly = if poly.num_vars() < params {
        lift_poly_to_n_vars(poly, params)
    } else {
        MultiLinearPoly::new(poly.hypercube_basis())
    };
    local_poly
        .hypercube_basis()
        .iter()
        .flat_map(|simd| simd.unpack())
        .collect()
}

fn _min_tree_elems<EvalF: Field>() -> usize {
    let epl = elems_per_leaf::<EvalF>();
    (2 * epl).max(8)
}

fn _base_elems_per_leaf<F: Field>() -> usize {
    assert!(
        LEAF_BYTES % F::SIZE == 0,
        "LEAF_BYTES ({LEAF_BYTES}) must be divisible by base F::SIZE ({})",
        F::SIZE,
    );
    LEAF_BYTES / F::SIZE
}

fn elems_per_leaf<EvalF: Field>() -> usize {
    assert!(
        EvalF::SIZE <= LEAF_BYTES,
        "WHIR PCS requires EvalF::SIZE ({}) <= LEAF_BYTES ({LEAF_BYTES})",
        EvalF::SIZE,
    );
    if LEAF_BYTES % EvalF::SIZE == 0 {
        LEAF_BYTES / EvalF::SIZE
    } else {
        LEAF_BYTES / EvalF::SIZE.next_power_of_two()
    }
}

fn build_tree_from_ext_codeword<EvalF: Field + SimdField<Scalar = EvalF>>(
    codeword: &[EvalF],
) -> Tree {
    if EvalF::SIZE > 0 && LEAF_BYTES % EvalF::SIZE == 0 {
        Tree::compact_new_with_field_elems::<EvalF, EvalF>(codeword.to_vec())
    } else {
        let epl = elems_per_leaf::<EvalF>();
        let padded_elem_size = EvalF::SIZE.next_power_of_two();
        let num_leaves = (codeword.len() + epl - 1) / epl;
        assert!(num_leaves.is_power_of_two());
        let mut leaves = Vec::with_capacity(num_leaves);
        for chunk in codeword.chunks(epl) {
            let mut data = [0u8; LEAF_BYTES];
            for (i, elem) in chunk.iter().enumerate() {
                let start = i * padded_elem_size;
                let mut buf = vec![0u8; EvalF::SIZE];
                elem.serialize_into(&mut buf[..]).unwrap();
                data[start..start + EvalF::SIZE].copy_from_slice(&buf);
            }
            leaves.push(tree::Leaf::new(data));
        }
        Tree::new_with_leaves(leaves)
    }
}

fn _extract_ext_from_padded_leaf<EvalF: Field>(leaf_data: &[u8], offset: usize) -> Option<EvalF> {
    let padded_elem_size = if LEAF_BYTES % EvalF::SIZE == 0 {
        EvalF::SIZE
    } else {
        EvalF::SIZE.next_power_of_two()
    };
    let start = offset * padded_elem_size;
    let end = start + EvalF::SIZE;
    if end > leaf_data.len() {
        return None;
    }
    EvalF::deserialize_from(&leaf_data[start..end]).ok()
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

fn generate_query_indices(
    transcript: &mut impl Transcript,
    len: usize,
    count: usize,
) -> Vec<usize> {
    assert!(len > 0);
    let target = count.min(len);
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

fn evaluate_ood<EvalF: FFTField>(codeword: &[EvalF], z: EvalF) -> EvalF {
    let mut coeffs = codeword.to_vec();
    bit_reverse_slice(&mut coeffs);
    let coeffs = EvalF::ifft(&coeffs);
    let mut result = EvalF::ZERO;
    let mut power = EvalF::ONE;
    for c in &coeffs {
        result += *c * power;
        power *= z;
    }
    result
}

fn re_encode<EvalF: FFTField>(
    folded_codeword: &[EvalF],
    old_rate_log: usize,
    new_rate_log: usize,
) -> Vec<EvalF> {
    let mut coeffs = folded_codeword.to_vec();
    bit_reverse_slice(&mut coeffs);
    let coeffs = EvalF::ifft(&coeffs);

    let degree = coeffs.len() >> old_rate_log;
    let new_cw_len = degree << new_rate_log;
    let mut padded = vec![EvalF::ZERO; new_cw_len];
    padded[..degree].copy_from_slice(&coeffs[..degree]);

    EvalF::fft_in_place(&mut padded);
    bit_reverse_slice(&mut padded);
    padded
}

fn pow_digest(transcript: &mut impl Transcript) -> [u8; 32] {
    let state = transcript.hash_and_return_state();
    Sha256::digest(&state).into()
}

fn grind_pow(transcript: &mut impl Transcript, pow_bits: usize) -> u64 {
    if pow_bits == 0 {
        return 0;
    }
    let digest = pow_digest(transcript);
    let threshold = u64::MAX >> pow_bits;
    for nonce in 0u64.. {
        let mut hasher = Sha256::new();
        hasher.update(digest);
        hasher.update(nonce.to_le_bytes());
        let result = hasher.finalize();
        let val = u64::from_be_bytes(result[..8].try_into().unwrap());
        if val <= threshold {
            transcript.append_u8_slice(&nonce.to_le_bytes());
            return nonce;
        }
    }
    unreachable!()
}

fn verify_pow(transcript: &mut impl Transcript, pow_bits: usize, nonce: u64) -> bool {
    if pow_bits == 0 {
        return true;
    }
    let digest = pow_digest(transcript);
    let threshold = u64::MAX >> pow_bits;
    let mut hasher = Sha256::new();
    hasher.update(digest);
    hasher.update(nonce.to_le_bytes());
    let result = hasher.finalize();
    let val = u64::from_be_bytes(result[..8].try_into().unwrap());
    if val > threshold {
        return false;
    }
    transcript.append_u8_slice(&nonce.to_le_bytes());
    true
}

#[allow(clippy::too_many_lines)]
pub(crate) fn whir_open<F, EvalF>(
    evals: &[F],
    initial_codeword: &[F],
    initial_tree: &Tree,
    num_vars: usize,
    eval_point: &[EvalF],
    transcript: &mut impl Transcript,
) -> WhirOpening<EvalF>
where
    F: FFTField + SimdField<Scalar = F>,
    EvalF: ExtensionField<BaseField = F> + FFTField + SimdField<Scalar = EvalF>,
{
    assert_eq!(eval_point.len(), num_vars);
    assert_eq!(evals.len(), 1 << num_vars);

    transcript.append_commitment(initial_tree.root().as_bytes());

    let n_committed = num_committed_rounds(num_vars);

    let ext_evals: Vec<EvalF> = evals.iter().map(|&x| EvalF::from(x)).collect();
    let mut f_table = ext_evals;
    let mut eq_table = build_eq_table(eval_point);

    let mut round_commitments = Vec::new();
    let mut round_trees: Vec<Tree> = Vec::new();
    let mut sumcheck_messages = Vec::with_capacity(num_vars);
    let mut ood_evaluations = Vec::new();
    let mut pow_nonces = Vec::new();

    let mut current_codeword_ext: Option<Vec<EvalF>> = None;
    let mut total_folds_done = 0;
    let mut current_cw_log = num_vars + WHIR_RATE_LOG;
    let mut current_rate_log = WHIR_RATE_LOG;

    for cr in 0..=n_committed {
        let folds_this_round = if cr < n_committed {
            WHIR_FOLDING_FACTOR
        } else {
            num_vars - total_folds_done
        };

        for sub in 0..folds_this_round {
            let sc_msg = compute_sumcheck_round::<F, EvalF>(&f_table, &eq_table);
            transcript.append_field_element(&sc_msg.eval_at_1);
            transcript.append_field_element(&sc_msg.eval_at_2);
            sumcheck_messages.push(sc_msg);

            let challenge: EvalF = transcript.generate_field_element();
            let level = current_cw_log - 1;
            let twiddle_coeffs = compute_twiddle_coeffs::<F>(level);
            current_cw_log -= 1;

            let is_first_fold = total_folds_done + sub == 0;
            let folded_codeword = if is_first_fold {
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
            current_codeword_ext = Some(folded_codeword);
        }

        total_folds_done += folds_this_round;

        if cr < n_committed {
            let folded_cw = current_codeword_ext.as_ref().unwrap();
            let new_rate_log = effective_rate_log(cr);
            let re_encoded = re_encode(folded_cw, current_rate_log, new_rate_log);

            let tree = build_tree_from_ext_codeword(&re_encoded);
            let root = tree.root();
            transcript.append_commitment(root.as_bytes());
            round_commitments.push(root);
            round_trees.push(tree);

            let mut round_ood = Vec::with_capacity(WHIR_OOD_SAMPLES);
            for _ in 0..WHIR_OOD_SAMPLES {
                let z: EvalF = transcript.generate_field_element();
                let v = evaluate_ood(&re_encoded, z);
                transcript.append_field_element(&v);
                round_ood.push(v);
            }
            ood_evaluations.push(round_ood);

            let nonce = grind_pow(transcript, WHIR_POW_BITS);
            pow_nonces.push(nonce);

            current_cw_log = (num_vars - total_folds_done) + new_rate_log;
            current_rate_log = new_rate_log;
            current_codeword_ext = Some(re_encoded);
        } else {
            let fp = current_codeword_ext.clone().unwrap_or_default();
            for f in &fp {
                transcript.append_field_element(f);
            }
        }
    }

    let fallback_no_queries = round_commitments.is_empty();
    if fallback_no_queries {
        transcript.append_commitment(initial_tree.root().as_bytes());
    }

    let final_poly = current_codeword_ext.unwrap_or_default();
    let committed_rounds = round_commitments.len();

    let mut round_query_proofs = Vec::with_capacity(committed_rounds);

    for cr in 0..committed_rounds {
        let q_count = whir_queries_for_committed_round(cr);
        let tree = &round_trees[cr];
        let num_leaves = tree.leaves.len();
        let query_indices = generate_query_indices(transcript, num_leaves, q_count);

        let mut proofs = Vec::with_capacity(query_indices.len());
        for &leaf_idx in &query_indices {
            proofs.push(WhirRoundQueryProof {
                leaf_proof: tree.index_query(leaf_idx),
            });
        }
        round_query_proofs.push(proofs);
    }

    WhirOpening {
        round_commitments,
        sumcheck_messages,
        ood_evaluations,
        pow_nonces,
        final_poly,
        round_query_proofs,
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn whir_verify<F, EvalF>(
    commitment: &WhirCommitment,
    eval_point: &[EvalF],
    claimed_eval: EvalF,
    opening: &WhirOpening<EvalF>,
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

    match num_vars.checked_add(WHIR_RATE_LOG) {
        Some(cl) if cl < usize::BITS as usize => {}
        _ => return false,
    };

    transcript.append_commitment(commitment.root.as_bytes());

    if opening.sumcheck_messages.len() != num_vars {
        return false;
    }

    let committed_rounds = opening.round_commitments.len();
    let n_committed_expected = num_committed_rounds(num_vars);

    if committed_rounds != n_committed_expected {
        return false;
    }
    if opening.round_query_proofs.len() != committed_rounds
        || opening.ood_evaluations.len() != committed_rounds
        || opening.pow_nonces.len() != committed_rounds
    {
        return false;
    }

    let final_rate_log = if n_committed_expected > 0 {
        effective_rate_log(n_committed_expected - 1)
    } else {
        WHIR_RATE_LOG
    };
    let expected_final_len = 1usize << final_rate_log;
    if opening.final_poly.len() != expected_final_len {
        return false;
    }

    let mut challenges = Vec::with_capacity(num_vars);
    let mut running_claim = claimed_eval;
    let mut total_folds_done = 0;

    for cr in 0..=n_committed_expected {
        let folds_this_round = if cr < n_committed_expected {
            WHIR_FOLDING_FACTOR
        } else {
            num_vars - total_folds_done
        };

        for sub in 0..folds_this_round {
            let global_round = total_folds_done + sub;
            let sc_msg = &opening.sumcheck_messages[global_round];
            transcript.append_field_element(&sc_msg.eval_at_1);
            transcript.append_field_element(&sc_msg.eval_at_2);

            let eval_at_0 = running_claim - sc_msg.eval_at_1;
            let challenge: EvalF = transcript.generate_field_element();
            challenges.push(challenge);
            running_claim =
                extrapolate_degree2(eval_at_0, sc_msg.eval_at_1, sc_msg.eval_at_2, challenge);
        }

        total_folds_done += folds_this_round;

        if cr < n_committed_expected {
            transcript.append_commitment(opening.round_commitments[cr].as_bytes());

            let ood_evals = &opening.ood_evaluations[cr];
            if ood_evals.len() != WHIR_OOD_SAMPLES {
                return false;
            }
            for ood_val in ood_evals {
                let _z: EvalF = transcript.generate_field_element();
                transcript.append_field_element(ood_val);
            }

            if !verify_pow(transcript, WHIR_POW_BITS, opening.pow_nonces[cr]) {
                return false;
            }
        } else {
            for f in &opening.final_poly {
                transcript.append_field_element(f);
            }
        }
    }

    let fallback_no_queries = committed_rounds == 0;
    if fallback_no_queries {
        transcript.append_commitment(commitment.root.as_bytes());
    }

    let fri_constant = opening.final_poly[0];
    for val in &opening.final_poly[1..] {
        if *val != fri_constant {
            return false;
        }
    }

    let eq_at_challenges = compute_eq_eval(&challenges, eval_point);
    if fri_constant * eq_at_challenges != running_claim {
        return false;
    }

    for cr in 0..committed_rounds {
        let q_count = whir_queries_for_committed_round(cr);
        let new_rate_log = effective_rate_log(cr);
        let degree_log = num_vars - (cr + 1) * WHIR_FOLDING_FACTOR;
        let re_encoded_cw_log = degree_log + new_rate_log;
        let re_encoded_len = 1usize << re_encoded_cw_log;
        let epl = elems_per_leaf::<EvalF>();
        let num_leaves = (re_encoded_len + epl - 1) / epl;

        let query_indices = generate_query_indices(transcript, num_leaves, q_count);

        let round_proofs = &opening.round_query_proofs[cr];
        if round_proofs.len() != query_indices.len() {
            return false;
        }

        let target_commitment = &opening.round_commitments[cr];

        for (qi, &leaf_idx) in query_indices.iter().enumerate() {
            let qp = &round_proofs[qi];
            if !qp.leaf_proof.verify(target_commitment) {
                return false;
            }
            if qp.leaf_proof.index != leaf_idx {
                return false;
            }
        }
    }

    true
}

pub(crate) fn whir_commit<F: FFTField + SimdField<Scalar = F>>(
    evals: &[F],
) -> (WhirCommitment, Tree, Vec<F>) {
    assert!(!evals.is_empty(), "whir_commit: evals must not be empty");
    let num_vars = evals.len().ilog2() as usize;
    assert_eq!(evals.len(), 1 << num_vars);
    let codeword = rs_encode_with_rate(evals, WHIR_RATE_LOG);
    // Mirror the open-side `build_tree_from_ext_codeword` branching:
    // when the field element size divides the leaf width we use the
    // compact packed encoding; when it doesn't we fall back to a
    // padded leaf where each element occupies its next-power-of-two
    // byte slot. This is exactly what the open / verify halves
    // already expect for non-divisible sizes (e.g. GoldilocksExt3
    // with `SIZE = 24` against `LEAF_BYTES = 64`), so the commit
    // path is now consistent with them and the previously-rejected
    // Ext3 path goes through.
    let tree = if F::SIZE > 0 && LEAF_BYTES % F::SIZE == 0 {
        Tree::compact_new_with_field_elems::<F, F>(codeword.clone())
    } else {
        build_padded_tree_from_codeword::<F>(&codeword)
    };
    let root = tree.root();
    (WhirCommitment { root, num_vars }, tree, codeword)
}

/// Pack a codeword whose element size does not divide [`LEAF_BYTES`]
/// into a tree of zero-padded leaves. Each element occupies a
/// `next_power_of_two(F::SIZE)`-byte slot inside the leaf so the
/// open-side query path can index into it deterministically via
/// `_extract_ext_from_padded_leaf`.
fn build_padded_tree_from_codeword<F: Field>(codeword: &[F]) -> Tree {
    let elems_per_leaf = elems_per_leaf::<F>();
    assert!(
        elems_per_leaf > 0,
        "build_padded_tree_from_codeword: elems_per_leaf must be positive"
    );
    let padded_elem_size = F::SIZE.next_power_of_two();
    let num_leaves = codeword.len().div_ceil(elems_per_leaf);
    assert!(
        num_leaves.is_power_of_two(),
        "build_padded_tree_from_codeword: leaf count {num_leaves} must be a power of two"
    );
    let mut leaves = Vec::with_capacity(num_leaves);
    for chunk in codeword.chunks(elems_per_leaf) {
        let mut data = [0u8; LEAF_BYTES];
        for (i, elem) in chunk.iter().enumerate() {
            let start = i * padded_elem_size;
            let mut buf = vec![0u8; F::SIZE];
            elem.serialize_into(&mut buf[..]).unwrap();
            data[start..start + F::SIZE].copy_from_slice(&buf);
        }
        leaves.push(tree::Leaf::new(data));
    }
    Tree::new_with_leaves(leaves)
}

#[cfg(test)]
pub fn whir_open_for_test(
    evals: &[goldilocks::Goldilocks],
    initial_codeword: &[goldilocks::Goldilocks],
    initial_tree: &Tree,
    num_vars: usize,
    eval_point: &[goldilocks::GoldilocksExt2],
    transcript: &mut impl Transcript,
) -> WhirOpening<goldilocks::GoldilocksExt2> {
    whir_open::<goldilocks::Goldilocks, goldilocks::GoldilocksExt2>(
        evals,
        initial_codeword,
        initial_tree,
        num_vars,
        eval_point,
        transcript,
    )
}

#[cfg(test)]
pub fn whir_verify_for_test(
    commitment: &WhirCommitment,
    eval_point: &[goldilocks::GoldilocksExt2],
    claimed_eval: goldilocks::GoldilocksExt2,
    opening: &WhirOpening<goldilocks::GoldilocksExt2>,
    transcript: &mut impl Transcript,
) -> bool {
    whir_verify::<goldilocks::Goldilocks, goldilocks::GoldilocksExt2>(
        commitment,
        eval_point,
        claimed_eval,
        opening,
        transcript,
    )
}

#[cfg(test)]
pub fn whir_open_for_test_ext3(
    evals: &[goldilocks::Goldilocks],
    initial_codeword: &[goldilocks::Goldilocks],
    initial_tree: &Tree,
    num_vars: usize,
    eval_point: &[goldilocks::GoldilocksExt3],
    transcript: &mut impl Transcript,
) -> WhirOpening<goldilocks::GoldilocksExt3> {
    whir_open::<goldilocks::Goldilocks, goldilocks::GoldilocksExt3>(
        evals,
        initial_codeword,
        initial_tree,
        num_vars,
        eval_point,
        transcript,
    )
}

#[cfg(test)]
pub fn whir_verify_for_test_ext3(
    commitment: &WhirCommitment,
    eval_point: &[goldilocks::GoldilocksExt3],
    claimed_eval: goldilocks::GoldilocksExt3,
    opening: &WhirOpening<goldilocks::GoldilocksExt3>,
    transcript: &mut impl Transcript,
) -> bool {
    whir_verify::<goldilocks::Goldilocks, goldilocks::GoldilocksExt3>(
        commitment,
        eval_point,
        claimed_eval,
        opening,
        transcript,
    )
}

#[cfg(test)]
pub fn whir_open_for_test_ext4(
    evals: &[goldilocks::Goldilocks],
    initial_codeword: &[goldilocks::Goldilocks],
    initial_tree: &Tree,
    num_vars: usize,
    eval_point: &[goldilocks::GoldilocksExt4],
    transcript: &mut impl Transcript,
) -> WhirOpening<goldilocks::GoldilocksExt4> {
    whir_open::<goldilocks::Goldilocks, goldilocks::GoldilocksExt4>(
        evals,
        initial_codeword,
        initial_tree,
        num_vars,
        eval_point,
        transcript,
    )
}

#[cfg(test)]
pub fn whir_verify_for_test_ext4(
    commitment: &WhirCommitment,
    eval_point: &[goldilocks::GoldilocksExt4],
    claimed_eval: goldilocks::GoldilocksExt4,
    opening: &WhirOpening<goldilocks::GoldilocksExt4>,
    transcript: &mut impl Transcript,
) -> bool {
    whir_verify::<goldilocks::Goldilocks, goldilocks::GoldilocksExt4>(
        commitment,
        eval_point,
        claimed_eval,
        opening,
        transcript,
    )
}

impl<C> ExpanderPCS<C> for WhirPCSForGKR<C>
where
    C: FieldEngine,
    C::CircuitField: FFTField + SimdField<Scalar = C::CircuitField>,
    C::ChallengeField: FFTField + SimdField<Scalar = C::ChallengeField>,
{
    const NAME: &'static str = "WhirPCS";
    const PCS_TYPE: PolynomialCommitmentType = PolynomialCommitmentType::Whir;

    type Params = usize;
    type ScratchPad = WhirScratchPad;
    type SRS = WhirSRS;
    type Commitment = WhirCommitment;
    type Opening = WhirOpening<C::ChallengeField>;
    type BatchOpening = ();

    fn gen_params(n_input_vars: usize, _world_size: usize) -> Self::Params {
        // The Merkle tree leaf packing requires:
        //   codeword_len * F::SIZE >= LEAF_BYTES  AND  codeword_len * F::SIZE % LEAF_BYTES == 0
        // where codeword_len = (1 << n_vars) << WHIR_RATE_LOG.
        // For small base fields (Goldilocks, 8 bytes) with LEAF_BYTES=64 this
        // means n_vars >= log2(LEAF_BYTES / F::SIZE) - WHIR_RATE_LOG.
        let elems_per_leaf = _base_elems_per_leaf::<C::CircuitField>();
        let min_evals = (elems_per_leaf >> WHIR_RATE_LOG).max(1).next_power_of_two();
        let min_vars = min_evals.ilog2() as usize;
        n_input_vars.max(min_vars)
    }

    fn gen_srs(
        _params: &Self::Params,
        _mpi_engine: &impl MPIEngine,
        _rng: impl rand::RngCore,
    ) -> Self::SRS {
        WhirSRS
    }

    fn init_scratch_pad(_params: &Self::Params, _mpi_engine: &impl MPIEngine) -> Self::ScratchPad {
        WhirScratchPad::default()
    }

    fn commit(
        params: &Self::Params,
        mpi_engine: &impl MPIEngine,
        _proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &impl MultilinearExtension<C::SimdCircuitField>,
        scratch_pad: &mut Self::ScratchPad,
    ) -> Option<Self::Commitment> {
        if !mpi_engine.is_single_process() {
            return None;
        }
        if poly.num_vars() > *params {
            return None;
        }
        let base_evals = prepare_base_evals::<C>(poly, *params);
        let num_evals = base_evals.len();
        let (commitment, tree, codeword) = whir_commit(&base_evals);
        scratch_pad.store_commit(tree, codeword, num_evals);
        Some(commitment)
    }

    fn open(
        params: &Self::Params,
        mpi_engine: &impl MPIEngine,
        _proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &impl MultilinearExtension<C::SimdCircuitField>,
        eval_point: &ExpanderSingleVarChallenge<C>,
        transcript: &mut impl Transcript,
        scratch_pad: &Self::ScratchPad,
    ) -> Option<Self::Opening> {
        if !mpi_engine.is_single_process() {
            return None;
        }
        if poly.num_vars() > *params || eval_point.num_vars() > *params {
            return None;
        }

        let effective_point = if eval_point.num_vars() < *params {
            lift_expander_challenge_to_n_vars(eval_point, *params)
        } else {
            eval_point.clone()
        };

        let base_evals = prepare_base_evals::<C>(poly, *params);
        let xs = effective_point.local_xs();

        let opening = if let Some((tree, codeword)) =
            scratch_pad.get_commit::<C::CircuitField>(base_evals.len())
        {
            whir_open::<C::CircuitField, C::ChallengeField>(
                &base_evals,
                codeword,
                tree,
                *params,
                &xs,
                transcript,
            )
        } else {
            let (_commitment, tree, codeword) = whir_commit(&base_evals);
            whir_open::<C::CircuitField, C::ChallengeField>(
                &base_evals,
                &codeword,
                &tree,
                *params,
                &xs,
                transcript,
            )
        };

        Some(opening)
    }

    fn verify(
        params: &Self::Params,
        _verifying_key: &<Self::SRS as StructuredReferenceString>::VKey,
        commitment: &Self::Commitment,
        eval_point: &ExpanderSingleVarChallenge<C>,
        claimed_eval: C::ChallengeField,
        transcript: &mut impl Transcript,
        opening: &Self::Opening,
    ) -> bool {
        if eval_point.num_vars() > *params {
            return false;
        }

        let effective_point = if eval_point.num_vars() < *params {
            lift_expander_challenge_to_n_vars(eval_point, *params)
        } else {
            eval_point.clone()
        };

        let xs = effective_point.local_xs();

        whir_verify::<C::CircuitField, C::ChallengeField>(
            commitment,
            &xs,
            claimed_eval,
            opening,
            transcript,
        )
    }
}
