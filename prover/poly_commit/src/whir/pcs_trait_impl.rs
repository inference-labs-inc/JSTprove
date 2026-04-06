use std::marker::PhantomData;

use arith::{ExtensionField, FFTField, Field, SimdField};
use gkr_engine::{
    ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, MPIEngine, PolynomialCommitmentType,
    StructuredReferenceString, Transcript,
};
use polynomials::{MultiLinearPoly, MultilinearExtension};
use tree::{Tree, LEAF_BYTES};

use sha2::{Digest, Sha256};

use super::parameters::{
    num_committed_rounds, whir_queries_for_committed_round, WHIR_FOLDING_FACTOR, WHIR_OOD_SAMPLES,
    WHIR_POW_BITS, WHIR_RATE_LOG,
};
use super::types::{WhirCommitment, WhirOpening, WhirRoundQueryProof, WhirSRS, WhirScratchPad};
use crate::basefold::encoding::{
    codeword_fold_single, compute_twiddle_coeffs, fold_codeword, fold_codeword_first_round,
    rs_encode_with_rate, verifier_folding_coeff,
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
    crate::basefold::encoding::bit_reverse_slice(&mut coeffs);
    let coeffs = EvalF::ifft(&coeffs);
    let mut result = EvalF::ZERO;
    let mut power = EvalF::ONE;
    for c in &coeffs {
        result += *c * power;
        power *= z;
    }
    result
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

fn replay_folds<F, EvalF>(
    source_values: &[EvalF],
    challenges: &[EvalF],
    codeword_log: usize,
    fold_start_round: usize,
    num_folds: usize,
    source_base_pair_idx: usize,
) -> EvalF
where
    F: FFTField,
    EvalF: ExtensionField<BaseField = F>,
{
    let mut current = source_values.to_vec();
    for k in 0..num_folds {
        let global_round = fold_start_round + k;
        let level = codeword_log - 1 - global_round;
        let half = current.len() / 2;
        let pair_base = source_base_pair_idx >> k;
        let mut next = Vec::with_capacity(half);
        for i in 0..half {
            let global_pair = pair_base + i;
            let twiddle = verifier_folding_coeff::<F>(level, global_pair);
            next.push(codeword_fold_single(
                current[2 * i],
                current[2 * i + 1],
                challenges[global_round],
                twiddle,
            ));
        }
        current = next;
    }
    assert_eq!(current.len(), 1);
    current[0]
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

    let min_elems = min_tree_elems::<EvalF>();
    let codeword_log = num_vars + WHIR_RATE_LOG;
    let n_committed = num_committed_rounds(num_vars);

    let ext_evals: Vec<EvalF> = evals.iter().map(|&x| EvalF::from(x)).collect();
    let mut f_table = ext_evals;
    let mut eq_table = build_eq_table(eval_point);

    let mut round_commitments = Vec::new();
    let mut round_trees: Vec<Tree> = Vec::new();
    let mut round_codewords: Vec<Vec<EvalF>> = Vec::new();
    let mut sumcheck_messages = Vec::with_capacity(num_vars);
    let mut ood_evaluations = Vec::new();
    let mut pow_nonces = Vec::new();

    let mut current_codeword_ext: Option<Vec<EvalF>> = None;
    let mut total_folds_done = 0;

    for cr in 0..=n_committed {
        let folds_this_round = if cr < n_committed {
            WHIR_FOLDING_FACTOR
        } else {
            num_vars - total_folds_done
        };

        for sub in 0..folds_this_round {
            let global_round = total_folds_done + sub;
            let sc_msg = compute_sumcheck_round::<F, EvalF>(&f_table, &eq_table);
            transcript.append_field_element(&sc_msg.eval_at_1);
            transcript.append_field_element(&sc_msg.eval_at_2);
            sumcheck_messages.push(sc_msg);

            let challenge: EvalF = transcript.generate_field_element();
            let level = codeword_log - 1 - global_round;
            let twiddle_coeffs = compute_twiddle_coeffs::<F>(level);

            let folded_codeword = if global_round == 0 {
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
            let cw = current_codeword_ext.as_ref().unwrap();
            if cw.len() >= min_elems {
                let tree = Tree::compact_new_with_field_elems::<EvalF, EvalF>(cw.clone());
                let root = tree.root();
                transcript.append_commitment(root.as_bytes());
                round_commitments.push(root);
                round_trees.push(tree);
                round_codewords.push(cw.clone());

                let mut round_ood = Vec::with_capacity(WHIR_OOD_SAMPLES);
                for _ in 0..WHIR_OOD_SAMPLES {
                    let z: EvalF = transcript.generate_field_element();
                    let v = evaluate_ood(cw, z);
                    transcript.append_field_element(&v);
                    round_ood.push(v);
                }
                ood_evaluations.push(round_ood);

                let nonce = grind_pow(transcript, WHIR_POW_BITS);
                pow_nonces.push(nonce);
            }
        } else {
            let fp = current_codeword_ext.clone().unwrap_or_default();
            for f in &fp {
                transcript.append_field_element(f);
            }
        }
    }

    let final_poly = current_codeword_ext.unwrap_or_default();
    let committed_rounds = round_commitments.len();
    let source_to_target = 1usize << WHIR_FOLDING_FACTOR;
    let ext_elems_per_leaf = LEAF_BYTES / EvalF::SIZE;

    let mut round_query_proofs = Vec::with_capacity(committed_rounds);

    for cr in 0..committed_rounds {
        let q_count = whir_queries_for_committed_round(cr);
        let target_cw = &round_codewords[cr];
        let target_len = target_cw.len();
        let query_indices = generate_query_indices(transcript, target_len, q_count);

        let source_tree = if cr == 0 {
            initial_tree
        } else {
            &round_trees[cr - 1]
        };
        let target_tree = &round_trees[cr];

        let source_elems_per_leaf = if cr == 0 {
            LEAF_BYTES / F::SIZE
        } else {
            ext_elems_per_leaf
        };

        let mut proofs = Vec::with_capacity(query_indices.len());

        for &target_idx in &query_indices {
            let source_start = target_idx * source_to_target;
            let source_end = source_start + source_to_target;
            let first_leaf = source_start / source_elems_per_leaf;
            let last_leaf = (source_end - 1) / source_elems_per_leaf;
            let mut source_leaf_proofs = Vec::new();
            for leaf_idx in first_leaf..=last_leaf {
                source_leaf_proofs.push(source_tree.index_query(leaf_idx));
            }

            let target_leaf_idx = target_idx / ext_elems_per_leaf;
            let target_leaf_proof = target_tree.index_query(target_leaf_idx);
            let t_start = target_leaf_idx * ext_elems_per_leaf;
            let t_end = (t_start + ext_elems_per_leaf).min(target_len);
            let target_leaf_values = target_cw[t_start..t_end].to_vec();

            proofs.push(WhirRoundQueryProof {
                source_leaf_proofs,
                target_leaf_proof,
                target_leaf_values,
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
    if leaf_data.len() < LEAF_BYTES || values.len() != elems_per_leaf {
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

    let codeword_log = match num_vars.checked_add(WHIR_RATE_LOG) {
        Some(cl) if cl < usize::BITS as usize => cl,
        _ => return false,
    };

    if opening.sumcheck_messages.len() != num_vars {
        return false;
    }

    let n_committed = num_committed_rounds(num_vars);
    let committed_rounds = opening.round_commitments.len();
    if committed_rounds != n_committed {
        return false;
    }
    if opening.round_query_proofs.len() != committed_rounds
        || opening.ood_evaluations.len() != committed_rounds
        || opening.pow_nonces.len() != committed_rounds
    {
        return false;
    }

    let mut challenges = Vec::with_capacity(num_vars);
    let mut running_claim = claimed_eval;
    let mut total_folds_done = 0;

    for cr in 0..=n_committed {
        let folds_this_round = if cr < n_committed {
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

        if cr < n_committed {
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

    if opening.final_poly.is_empty() {
        return false;
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

    let ext_elems_per_leaf = LEAF_BYTES / EvalF::SIZE;
    let source_to_target = 1usize << WHIR_FOLDING_FACTOR;

    for cr in 0..committed_rounds {
        let q_count = whir_queries_for_committed_round(cr);
        let folds_before = (cr + 1) * WHIR_FOLDING_FACTOR;
        let target_codeword_log = codeword_log - folds_before;
        let target_len = 1usize << target_codeword_log;
        let query_indices = generate_query_indices(transcript, target_len, q_count);

        let round_proofs = &opening.round_query_proofs[cr];
        if round_proofs.len() != query_indices.len() {
            return false;
        }

        let source_commitment = if cr == 0 {
            &commitment.root
        } else {
            &opening.round_commitments[cr - 1]
        };
        let target_commitment = &opening.round_commitments[cr];

        let source_elems_per_leaf = if cr == 0 {
            LEAF_BYTES / F::SIZE
        } else {
            ext_elems_per_leaf
        };

        let fold_start_round = cr * WHIR_FOLDING_FACTOR;

        for (qi, &target_idx) in query_indices.iter().enumerate() {
            let qp = &round_proofs[qi];

            for sp in &qp.source_leaf_proofs {
                if !sp.verify(source_commitment) {
                    return false;
                }
            }
            if !qp.target_leaf_proof.verify(target_commitment) {
                return false;
            }

            let source_start = target_idx * source_to_target;
            let source_end = source_start + source_to_target;
            let first_leaf = source_start / source_elems_per_leaf;
            let last_leaf = (source_end - 1) / source_elems_per_leaf;
            let expected_leaves = last_leaf - first_leaf + 1;

            if qp.source_leaf_proofs.len() != expected_leaves {
                return false;
            }

            for (li, sp) in qp.source_leaf_proofs.iter().enumerate() {
                if sp.index != first_leaf + li {
                    return false;
                }
            }

            let mut source_values = Vec::with_capacity(source_to_target);

            for elem_i in 0..source_to_target {
                let global_idx = source_start + elem_i;
                let leaf_of_elem = global_idx / source_elems_per_leaf;
                let offset_in_leaf = global_idx % source_elems_per_leaf;
                let proof_idx = leaf_of_elem - first_leaf;
                let leaf_data = &qp.source_leaf_proofs[proof_idx].leaf.data;

                let val: EvalF = if cr == 0 {
                    match extract_base_from_leaf::<F>(leaf_data, offset_in_leaf) {
                        Some(v) => EvalF::from(v),
                        None => return false,
                    }
                } else {
                    match extract_base_from_leaf::<EvalF>(leaf_data, offset_in_leaf) {
                        Some(v) => v,
                        None => return false,
                    }
                };
                source_values.push(val);
            }

            let source_base_pair = source_start / 2;
            let expected = replay_folds::<F, EvalF>(
                &source_values,
                &challenges,
                codeword_log,
                fold_start_round,
                WHIR_FOLDING_FACTOR,
                source_base_pair,
            );

            if !verify_leaf_values::<EvalF>(&qp.target_leaf_proof.leaf.data, &qp.target_leaf_values)
            {
                return false;
            }

            let target_leaf_idx_check = target_idx / ext_elems_per_leaf;
            if qp.target_leaf_proof.index != target_leaf_idx_check {
                return false;
            }

            let target_offset = target_idx % ext_elems_per_leaf;
            if target_offset >= qp.target_leaf_values.len() {
                return false;
            }
            if qp.target_leaf_values[target_offset] != expected {
                return false;
            }
        }
    }

    true
}

fn whir_commit<F: FFTField + SimdField<Scalar = F>>(evals: &[F]) -> (WhirCommitment, Tree, Vec<F>) {
    let num_vars = evals.len().ilog2() as usize;
    assert_eq!(evals.len(), 1 << num_vars);
    let codeword = rs_encode_with_rate(evals, WHIR_RATE_LOG);
    let tree = Tree::compact_new_with_field_elems::<F, F>(codeword.clone());
    let root = tree.root();
    (WhirCommitment { root, num_vars }, tree, codeword)
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
        n_input_vars
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
            unimplemented!("WHIR MPI not yet supported");
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
            unimplemented!("WHIR MPI not yet supported");
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
