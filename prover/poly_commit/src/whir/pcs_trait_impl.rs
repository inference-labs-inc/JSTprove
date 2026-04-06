use std::marker::PhantomData;

use arith::{ExtensionField, FFTField, Field, SimdField};
use gkr_engine::{
    ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, MPIEngine, PolynomialCommitmentType,
    StructuredReferenceString, Transcript,
};
use polynomials::{MultiLinearPoly, MultilinearExtension};
use tree::{Tree, LEAF_BYTES};

use super::parameters::{whir_queries_for_round, WHIR_RATE_LOG};
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

#[allow(clippy::too_many_lines)]
fn whir_open<F, EvalF>(
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

    let base_elems_per_leaf = LEAF_BYTES / F::SIZE;
    let ext_elems_per_leaf = LEAF_BYTES / EvalF::SIZE;

    let mut round_query_proofs: Vec<Vec<WhirRoundQueryProof<EvalF>>> =
        Vec::with_capacity(committed_rounds);

    for round in 0..committed_rounds {
        let q_count = whir_queries_for_round(round);
        let target_len = round_codewords[round].len();
        let query_indices = generate_query_indices(transcript, target_len / 2, q_count);

        let source_tree = if round == 0 {
            initial_tree
        } else {
            &round_trees[round - 1]
        };
        let target_tree = &round_trees[round];
        let target_cw = &round_codewords[round];

        let source_elems_per_leaf = if round == 0 {
            base_elems_per_leaf
        } else {
            ext_elems_per_leaf
        };

        let mut proofs = Vec::with_capacity(query_indices.len());

        for &pair_idx in &query_indices {
            let source_even = pair_idx * 2;
            let source_odd = source_even + 1;
            let source_even_leaf = source_even / source_elems_per_leaf;
            let source_odd_leaf = source_odd / source_elems_per_leaf;

            let source_leaf_proof = source_tree.index_query(source_even_leaf);
            let source_sibling_proof = if source_odd_leaf != source_even_leaf {
                source_tree.index_query(source_odd_leaf)
            } else {
                source_leaf_proof.clone()
            };

            let target_even = (pair_idx / 2) * 2;
            let target_odd = target_even + 1;
            let target_even_leaf = target_even / ext_elems_per_leaf;
            let target_odd_leaf = target_odd / ext_elems_per_leaf;

            let target_leaf_proof = target_tree.index_query(target_even_leaf);
            let t_start = target_even_leaf * ext_elems_per_leaf;
            let t_end = (t_start + ext_elems_per_leaf).min(target_cw.len());
            let target_leaf_values = target_cw[t_start..t_end].to_vec();

            let (target_sibling_proof, target_sibling_values) =
                if target_odd_leaf != target_even_leaf {
                    let proof = target_tree.index_query(target_odd_leaf);
                    let s_start = target_odd_leaf * ext_elems_per_leaf;
                    let s_end = (s_start + ext_elems_per_leaf).min(target_cw.len());
                    (proof, target_cw[s_start..s_end].to_vec())
                } else {
                    (target_leaf_proof.clone(), target_leaf_values.clone())
                };

            proofs.push(WhirRoundQueryProof {
                source_leaf_proof,
                source_sibling_proof,
                target_leaf_proof,
                target_sibling_proof,
                target_leaf_values,
                target_sibling_values,
            });
        }

        round_query_proofs.push(proofs);
    }

    WhirOpening {
        round_commitments,
        sumcheck_messages,
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
fn whir_verify<F, EvalF>(
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

    let committed_rounds = opening.round_commitments.len();
    if opening.round_query_proofs.len() != committed_rounds {
        return false;
    }

    let min_elems = min_tree_elems::<EvalF>();

    let expected_final_poly_len = if committed_rounds < num_vars {
        1usize << (codeword_log - 1 - committed_rounds)
    } else {
        return false;
    };
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

    let base_elems_per_leaf = LEAF_BYTES / F::SIZE;
    let ext_elems_per_leaf = LEAF_BYTES / EvalF::SIZE;

    for round in 0..committed_rounds {
        let q_count = whir_queries_for_round(round);
        let target_len = 1usize << (codeword_log - 1 - round);
        let query_indices = generate_query_indices(transcript, target_len / 2, q_count);

        let round_proofs = &opening.round_query_proofs[round];
        if round_proofs.len() != query_indices.len() {
            return false;
        }

        let source_commitment = if round == 0 {
            &commitment.root
        } else {
            &opening.round_commitments[round - 1]
        };
        let target_commitment = &opening.round_commitments[round];

        let source_elems_per_leaf = if round == 0 {
            base_elems_per_leaf
        } else {
            ext_elems_per_leaf
        };

        for (qi, &pair_idx) in query_indices.iter().enumerate() {
            let qp = &round_proofs[qi];

            if !qp.source_leaf_proof.verify(source_commitment)
                || !qp.source_sibling_proof.verify(source_commitment)
                || !qp.target_leaf_proof.verify(target_commitment)
                || !qp.target_sibling_proof.verify(target_commitment)
            {
                return false;
            }

            let source_even = pair_idx * 2;
            let source_odd = source_even + 1;
            let source_even_leaf = source_even / source_elems_per_leaf;
            let source_odd_leaf = source_odd / source_elems_per_leaf;

            if qp.source_leaf_proof.index != source_even_leaf
                || qp.source_sibling_proof.index != source_odd_leaf
            {
                return false;
            }

            let even_offset = source_even % source_elems_per_leaf;
            let odd_offset = source_odd % source_elems_per_leaf;

            let (left, right): (EvalF, EvalF) = if round == 0 {
                let l =
                    match extract_base_from_leaf::<F>(&qp.source_leaf_proof.leaf.data, even_offset)
                    {
                        Some(v) => EvalF::from(v),
                        None => return false,
                    };
                let r = if source_odd_leaf == source_even_leaf {
                    match extract_base_from_leaf::<F>(&qp.source_leaf_proof.leaf.data, odd_offset) {
                        Some(v) => EvalF::from(v),
                        None => return false,
                    }
                } else {
                    match extract_base_from_leaf::<F>(
                        &qp.source_sibling_proof.leaf.data,
                        odd_offset,
                    ) {
                        Some(v) => EvalF::from(v),
                        None => return false,
                    }
                };
                (l, r)
            } else {
                let l = match extract_base_from_leaf::<EvalF>(
                    &qp.source_leaf_proof.leaf.data,
                    even_offset,
                ) {
                    Some(v) => v,
                    None => return false,
                };
                let r = if source_odd_leaf == source_even_leaf {
                    match extract_base_from_leaf::<EvalF>(
                        &qp.source_leaf_proof.leaf.data,
                        odd_offset,
                    ) {
                        Some(v) => v,
                        None => return false,
                    }
                } else {
                    match extract_base_from_leaf::<EvalF>(
                        &qp.source_sibling_proof.leaf.data,
                        odd_offset,
                    ) {
                        Some(v) => v,
                        None => return false,
                    }
                };
                (l, r)
            };

            let level = codeword_log - 1 - round;
            let twiddle = verifier_folding_coeff::<F>(level, pair_idx);
            let expected_folded = codeword_fold_single(left, right, challenges[round], twiddle);

            if !verify_leaf_values::<EvalF>(&qp.target_leaf_proof.leaf.data, &qp.target_leaf_values)
            {
                return false;
            }

            let target_offset = pair_idx % ext_elems_per_leaf;
            if target_offset >= qp.target_leaf_values.len() {
                return false;
            }
            let actual_folded = qp.target_leaf_values[target_offset];
            if actual_folded != expected_folded {
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
