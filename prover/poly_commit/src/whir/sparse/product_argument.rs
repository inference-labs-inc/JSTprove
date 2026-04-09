//! Thaler-style product-circuit GKR sumcheck for multiset hashes.
//!
//! Reduces a claim of the form
//!
//! ```text
//!     Π_{i ∈ {0,1}^k} a_i  =  c
//! ```
//!
//! to an evaluation claim
//!
//! ```text
//!     ã(r) = v ,    r ∈ E^k
//! ```
//!
//! about the multilinear extension of `a` at a sumcheck-random point.
//! The caller is responsible for verifying that final evaluation
//! against a polynomial commitment in a subsequent step (Phase 1d
//! does this for the sparse-MLE WHIR commitment via the WHIR §5.2
//! batched proximity test).
//!
//! Construction. We materialize the balanced binary product tree
//!
//! ```text
//!     L_0       = a                          (length 2^k)
//!     L_l[i]    = L_{l-1}[2i] · L_{l-1}[2i+1]
//!     L_k[0]    = c                          (the root)
//! ```
//!
//! Layer `l` has `2^{k-l}` elements. The reduction from layer `l` to
//! layer `l - 1` is the standard GKR identity
//!
//! ```text
//!     L_l(point) = Σ_{b ∈ {0,1}^{k-l}} eq(b, point) · L_{l-1}(b, 0) · L_{l-1}(b, 1)
//! ```
//!
//! which is a sumcheck over `k - l` variables of a degree-3 product
//! `(eq · L · L)`. The sumcheck random point `r_sc ∈ E^{k-l}` produces
//! two layer-`l-1` claims `L_{l-1}(r_sc, 0)` and `L_{l-1}(r_sc, 1)`,
//! which the prover sends explicitly. They are merged into a single
//! layer-`l-1` claim by drawing `β ← E` and forming
//!
//! ```text
//!     L_{l-1}((r_sc, β)) = (1 - β) · L_{l-1}(r_sc, 0) + β · L_{l-1}(r_sc, 1)
//! ```
//!
//! by linearity of the multilinear extension. Iterating from `l = k`
//! down to `l = 1` lands on a single claim about `L_0 = a` at a point
//! in `E^k`, the leaf evaluation the caller must validate.
//!
//! Round counts. Layer reduction `k → k-1` runs `0` sumcheck rounds
//! (the root layer has a single element, so the sumcheck claim is the
//! root value itself). Reduction `k - 1 → k - 2` runs `1` round,
//! `k - 2 → k - 3` runs `2` rounds, and so on. The leaf reduction
//! `1 → 0` runs `k - 1` rounds. Total sumcheck rounds therefore are
//!
//! ```text
//!     0 + 1 + 2 + … + (k-1)  =  k(k-1)/2
//! ```
//!
//! For `k = 20` (a million leaves) this is 190 rounds, each carrying
//! 4 extension-field evaluations.
//!
//! Reference: Thaler, "Time-Optimal Interactive Proofs for Circuit
//! Evaluation" (CRYPTO 2013), and Spartan §7.2 (`ProductLayerProof`)
//! for the four-product batched variant we wrap on top in
//! `prove_multiset_equality` further down.

use arith::{ExtensionField, Field};
use gkr_engine::Transcript;

use super::types::eval_eq_at_index;

/// Univariate polynomial round message — degree-3 polynomial sent in
/// evaluation form at the integer points `0, 1, 2, 3`. The prover
/// could compress to three evaluations using the sumcheck invariant
/// `g(0) + g(1) = running_claim`; we keep the explicit form to match
/// the eval-claim sumcheck and to keep verifier-side audits simple.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProductRound<F: Field> {
    pub evals: [F; 4],
}

/// Reduction transcript for one product-tree layer.
///
/// `sumcheck_rounds.len() = k - l` for the reduction from layer `l`
/// to layer `l - 1`, where `k` is the log of the leaf count. The two
/// `*_eval` fields are the prover's claimed evaluations of layer
/// `l - 1` at the post-sumcheck random point with the trailing
/// hypercube bit set to `0` and `1` respectively.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProductLayerProof<F: Field> {
    pub sumcheck_rounds: Vec<ProductRound<F>>,
    pub left_eval: F,
    pub right_eval: F,
}

/// Full product-circuit GKR proof — one layer reduction per layer of
/// the product tree, ordered top-down (root → leaves).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProductCircuitProof<F: Field> {
    pub layers: Vec<ProductLayerProof<F>>,
}

/// Final claim returned to the caller after the verifier accepts a
/// product-circuit proof: the sumcheck-random leaf point and the
/// claimed multilinear extension evaluation at that point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductCircuitClaim<F: Field> {
    pub leaf_point: Vec<F>,
    pub leaf_eval: F,
}

/// Build the product tree levels from a leaf vector of length `2^k`.
/// Returns `levels` such that `levels[0] = leaves` and
/// `levels.last()` is a length-1 vector containing the full product.
fn build_product_tree<F: Field>(leaves: &[F]) -> Vec<Vec<F>> {
    assert!(
        leaves.len().is_power_of_two(),
        "build_product_tree: leaf count must be a power of two"
    );
    let mut levels: Vec<Vec<F>> = Vec::with_capacity(1 + leaves.len().trailing_zeros() as usize);
    levels.push(leaves.to_vec());
    while levels.last().unwrap().len() > 1 {
        let prev = levels.last().unwrap();
        let next: Vec<F> = (0..prev.len() / 2)
            .map(|i| prev[2 * i] * prev[2 * i + 1])
            .collect();
        levels.push(next);
    }
    levels
}

/// Lift a base-field leaf vector into the extension field.
///
/// Provided for callers that originate the product over base-field
/// leaves (e.g. the per-axis multiset hash sets, which compose
/// `h_γ(addr, val, ts)` from base-field components). Currently used
/// only by the in-crate tests; the production caller (Phase 1d) will
/// invoke this directly when materializing the four product circuits
/// per address axis.
#[cfg(test)]
fn lift_leaves<F: Field, E: ExtensionField<BaseField = F>>(leaves: &[F]) -> Vec<E> {
    leaves.iter().map(|&v| E::from(v)).collect()
}

/// Run the prover side of the product-circuit GKR sumcheck.
///
/// `leaves` is the dense extension-field leaf vector of length
/// `2^k`. Returns the asserted root and a [`ProductCircuitProof`]
/// the verifier can replay against the same transcript.
///
/// # Panics
/// Panics if `leaves` is empty or its length is not a power of two.
pub fn prove_product_circuit<E: ExtensionField>(
    leaves: &[E],
    transcript: &mut impl Transcript,
) -> (E, ProductCircuitProof<E>) {
    assert!(
        !leaves.is_empty() && leaves.len().is_power_of_two(),
        "prove_product_circuit: leaves length must be a power of two ≥ 1"
    );
    let k = leaves.len().trailing_zeros() as usize;

    // Trivial case: a single leaf — the root is the leaf itself and
    // there is nothing to reduce.
    if k == 0 {
        return (leaves[0], ProductCircuitProof { layers: Vec::new() });
    }

    let levels = build_product_tree::<E>(leaves);
    let root = levels.last().unwrap()[0];

    // The current claim is `L_l(point) = current_claim` for the
    // current layer l (initially the root, l = k, point of length 0).
    let mut current_layer = k;
    let mut current_point: Vec<E> = Vec::new();
    let mut current_claim = root;

    let mut layer_proofs: Vec<ProductLayerProof<E>> = Vec::with_capacity(k);

    while current_layer > 0 {
        // Reduce from layer `current_layer` to layer `current_layer - 1`.
        // The sumcheck has `k - current_layer` variables; the next-layer
        // (layer current_layer - 1) is `levels[current_layer - 1]`.
        let next_layer = &levels[current_layer - 1];
        let n_sub = next_layer.len(); // = 2^{k - (current_layer - 1)}
        debug_assert_eq!(n_sub, 1usize << (k - current_layer + 1));

        let num_rounds = k - current_layer; // = log2(n_sub) - 1

        // Build the three working tables for the sumcheck:
        //   left[i]  = L_{l-1}(i, 0)
        //   right[i] = L_{l-1}(i, 1)
        //   eq[i]    = ẽq(i, current_point)
        // where i ranges over {0,1}^{num_rounds}.
        let half = n_sub / 2;
        let mut left_table: Vec<E> = (0..half).map(|i| next_layer[2 * i]).collect();
        let mut right_table: Vec<E> = (0..half).map(|i| next_layer[2 * i + 1]).collect();
        let mut eq_table: Vec<E> = (0..half)
            .map(|i| eval_eq_at_index(&current_point, i))
            .collect();

        let mut sumcheck_rounds: Vec<ProductRound<E>> = Vec::with_capacity(num_rounds);
        let mut sumcheck_random: Vec<E> = Vec::with_capacity(num_rounds);
        let mut running = current_claim;

        for _ in 0..num_rounds {
            let h = left_table.len() / 2;
            let mut g = [E::ZERO; 4];
            for (e_idx, slot) in g.iter_mut().enumerate() {
                let e_pt = E::from(e_idx as u32);
                let one_minus_e = E::ONE - e_pt;
                let mut acc = E::ZERO;
                for k_idx in 0..h {
                    let l = left_table[2 * k_idx] * one_minus_e + left_table[2 * k_idx + 1] * e_pt;
                    let r =
                        right_table[2 * k_idx] * one_minus_e + right_table[2 * k_idx + 1] * e_pt;
                    let q = eq_table[2 * k_idx] * one_minus_e + eq_table[2 * k_idx + 1] * e_pt;
                    acc += q * l * r;
                }
                *slot = acc;
            }
            debug_assert_eq!(
                g[0] + g[1],
                running,
                "product sumcheck invariant violated at layer {current_layer} round {}",
                sumcheck_rounds.len()
            );

            for ev in &g {
                transcript.append_field_element(ev);
            }
            let r: E = transcript.generate_field_element();
            sumcheck_random.push(r);
            running = lagrange_interpolate_degree3(&g, r);

            let one_minus_r = E::ONE - r;
            for k_idx in 0..h {
                left_table[k_idx] =
                    left_table[2 * k_idx] * one_minus_r + left_table[2 * k_idx + 1] * r;
                right_table[k_idx] =
                    right_table[2 * k_idx] * one_minus_r + right_table[2 * k_idx + 1] * r;
                eq_table[k_idx] = eq_table[2 * k_idx] * one_minus_r + eq_table[2 * k_idx + 1] * r;
            }
            left_table.truncate(h);
            right_table.truncate(h);
            eq_table.truncate(h);

            sumcheck_rounds.push(ProductRound { evals: g });
        }

        // After the sumcheck, the prover sends explicit evaluations
        // L_{l-1}(r_sc, 0) and L_{l-1}(r_sc, 1) at the random point.
        let left_eval = left_table[0];
        let right_eval = right_table[0];
        transcript.append_field_element(&left_eval);
        transcript.append_field_element(&right_eval);

        // Verifier consistency check (recorded by us so the test can
        // mechanically reproduce it): eq(r_sc, current_point) · l · r
        // must equal the sumcheck final claim `running`.
        let eq_at_r = eq_table.first().copied().unwrap_or(E::ONE);
        debug_assert_eq!(
            eq_at_r * left_eval * right_eval,
            running,
            "product sumcheck final claim inconsistent at layer {current_layer}"
        );

        // Merge the two sub-claims into a single claim about
        // L_{l-1} at point (β, r_sc). β occupies position 0 — the
        // LSB — because that is the variable distinguishing
        // L_{l-1}((0, r_sc)) (= left_eval) from L_{l-1}((1, r_sc))
        // (= right_eval) under the little-endian MLE convention used
        // by `eval_eq_at_index` and the rest of the prover.
        let beta: E = transcript.generate_field_element();
        let merged = (E::ONE - beta) * left_eval + beta * right_eval;
        let mut next_point = Vec::with_capacity(sumcheck_random.len() + 1);
        next_point.push(beta);
        next_point.extend(sumcheck_random);

        layer_proofs.push(ProductLayerProof {
            sumcheck_rounds,
            left_eval,
            right_eval,
        });

        current_layer -= 1;
        current_point = next_point;
        current_claim = merged;
    }

    // current_claim is now the claim about L_0 (the leaves) at
    // current_point. The caller will validate this by opening the
    // leaves polynomial commitment at current_point.
    let _ = current_claim; // returned by `verify` indirectly via the proof
    (
        root,
        ProductCircuitProof {
            layers: layer_proofs,
        },
    )
}

/// Run the verifier side of the product-circuit GKR sumcheck.
///
/// Returns `Some(ProductCircuitClaim { leaf_point, leaf_eval })` on
/// acceptance. The caller must additionally verify
/// `leaves_mle(leaf_point) = leaf_eval` against a polynomial
/// commitment to the leaves.
pub fn verify_product_circuit<E: ExtensionField>(
    log_n: usize,
    claimed_root: E,
    proof: &ProductCircuitProof<E>,
    transcript: &mut impl Transcript,
) -> Option<ProductCircuitClaim<E>> {
    if log_n == 0 {
        // No layers; the root is the lone leaf and the leaf point
        // is empty. Accept iff the proof has no layers and the
        // claimed root is the leaf evaluation.
        if !proof.layers.is_empty() {
            return None;
        }
        return Some(ProductCircuitClaim {
            leaf_point: Vec::new(),
            leaf_eval: claimed_root,
        });
    }

    if proof.layers.len() != log_n {
        return None;
    }

    let mut current_layer = log_n;
    let mut current_point: Vec<E> = Vec::new();
    let mut current_claim = claimed_root;

    for layer in &proof.layers {
        let num_rounds = log_n - current_layer;
        if layer.sumcheck_rounds.len() != num_rounds {
            return None;
        }

        let mut running = current_claim;
        let mut sumcheck_random: Vec<E> = Vec::with_capacity(num_rounds);
        for round in &layer.sumcheck_rounds {
            // Sumcheck invariant: g(0) + g(1) == running.
            if round.evals[0] + round.evals[1] != running {
                return None;
            }
            for ev in &round.evals {
                transcript.append_field_element(ev);
            }
            let r: E = transcript.generate_field_element();
            sumcheck_random.push(r);
            running = lagrange_interpolate_degree3(&round.evals, r);
        }

        // Read explicit (left, right) sub-claims and consistency-check
        // them against the sumcheck final value via eq(r_sc, current_point).
        transcript.append_field_element(&layer.left_eval);
        transcript.append_field_element(&layer.right_eval);
        let eq_at_r = eq_eval(&sumcheck_random, &current_point);
        if eq_at_r * layer.left_eval * layer.right_eval != running {
            return None;
        }

        // Merge via β. See `prove_product_circuit` for the LSB-first
        // ordering rationale.
        let beta: E = transcript.generate_field_element();
        let merged = (E::ONE - beta) * layer.left_eval + beta * layer.right_eval;
        let mut next_point = Vec::with_capacity(sumcheck_random.len() + 1);
        next_point.push(beta);
        next_point.extend(sumcheck_random);

        current_layer -= 1;
        current_point = next_point;
        current_claim = merged;
    }

    Some(ProductCircuitClaim {
        leaf_point: current_point,
        leaf_eval: current_claim,
    })
}

/// `ẽq(a, b) = Π_i (a_i · b_i + (1 - a_i)·(1 - b_i))` for two equal-length
/// extension-field tuples. Used by the verifier to compute the
/// `eq(r_sc, current_point)` factor without instantiating the full
/// hypercube table.
#[inline]
fn eq_eval<E: Field>(a: &[E], b: &[E]) -> E {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = E::ONE;
    for (ai, bi) in a.iter().zip(b.iter()) {
        acc *= *ai * *bi + (E::ONE - *ai) * (E::ONE - *bi);
    }
    acc
}

/// Lagrange interpolation through the four integer nodes `(0, evals[0])
/// (1, evals[1]) (2, evals[2]) (3, evals[3])` evaluated at `r`. The
/// degree-3 inverse denominators are constants pre-computed in
/// closed form so we avoid `Field::inv` and the round polynomial
/// can be evaluated in O(1) field operations.
///
/// Closed form coefficients (where `i` is the basis index):
///   denom_0 = (0-1)(0-2)(0-3) = -6
///   denom_1 = (1-0)(1-2)(1-3) =  2
///   denom_2 = (2-0)(2-1)(2-3) = -2
///   denom_3 = (3-0)(3-1)(3-2) =  6
#[inline]
fn lagrange_interpolate_degree3<F: Field>(evals: &[F; 4], r: F) -> F {
    let r0 = r;
    let r1 = r - F::from(1u32);
    let r2 = r - F::from(2u32);
    let r3 = r - F::from(3u32);

    let inv_neg_6 = F::from(6u32)
        .inv()
        .expect("|F| > 6 — true for any practical field")
        .neg();
    let inv_pos_2 = F::from(2u32)
        .inv()
        .expect("|F| > 2 — true for any practical field");
    let inv_neg_2 = inv_pos_2.neg();
    let inv_pos_6 = F::from(6u32)
        .inv()
        .expect("|F| > 6 — true for any practical field");

    let l0 = r1 * r2 * r3 * inv_neg_6;
    let l1 = r0 * r2 * r3 * inv_pos_2;
    let l2 = r0 * r1 * r3 * inv_neg_2;
    let l3 = r0 * r1 * r2 * inv_pos_6;

    evals[0] * l0 + evals[1] * l1 + evals[2] * l2 + evals[3] * l3
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whir::sparse::types::eval_eq_at_index;
    use arith::Field;
    use goldilocks::{Goldilocks, GoldilocksExt3};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use transcript::BytesHashTranscript;
    type Sha2T = BytesHashTranscript<gkr_hashers::SHA256hasher>;

    fn rng_for(label: &str) -> ChaCha20Rng {
        let mut seed = [0u8; 32];
        let bytes = label.as_bytes();
        let n = bytes.len().min(32);
        seed[..n].copy_from_slice(&bytes[..n]);
        ChaCha20Rng::from_seed(seed)
    }

    fn dense_mle_eval<E: Field>(coeffs: &[E], point: &[E]) -> E {
        let mut acc = E::ZERO;
        for (b, &c) in coeffs.iter().enumerate() {
            acc += c * eval_eq_at_index(point, b);
        }
        acc
    }

    #[test]
    fn lagrange_degree3_round_trip_through_random_quartic_truncation() {
        // Test cubic polynomial g(x) = 1 + 2x - 3x^2 + 4x^3
        let mut rng = rng_for("lagrange_d3");
        let g = |x: GoldilocksExt3| -> GoldilocksExt3 {
            GoldilocksExt3::from(1u64) + GoldilocksExt3::from(2u64) * x
                - GoldilocksExt3::from(3u64) * x.square()
                + GoldilocksExt3::from(4u64) * x.square() * x
        };
        let evals = [
            g(GoldilocksExt3::from(0u32)),
            g(GoldilocksExt3::from(1u32)),
            g(GoldilocksExt3::from(2u32)),
            g(GoldilocksExt3::from(3u32)),
        ];
        let r = GoldilocksExt3::random_unsafe(&mut rng);
        assert_eq!(lagrange_interpolate_degree3(&evals, r), g(r));
    }

    #[test]
    fn product_circuit_single_leaf_round_trip() {
        let leaves = vec![GoldilocksExt3::from(42u64)];
        let mut p_t = Sha2T::new();
        let (root, proof) = prove_product_circuit(&leaves, &mut p_t);
        assert_eq!(root, GoldilocksExt3::from(42u64));
        assert!(proof.layers.is_empty());

        let mut v_t = Sha2T::new();
        let claim = verify_product_circuit(0, root, &proof, &mut v_t).unwrap();
        assert!(claim.leaf_point.is_empty());
        assert_eq!(claim.leaf_eval, root);
    }

    #[test]
    fn product_circuit_two_leaves_round_trip() {
        let leaves = vec![GoldilocksExt3::from(3u64), GoldilocksExt3::from(7u64)];
        let mut p_t = Sha2T::new();
        let (root, proof) = prove_product_circuit(&leaves, &mut p_t);
        assert_eq!(root, GoldilocksExt3::from(21u64));
        assert_eq!(proof.layers.len(), 1);
        assert!(proof.layers[0].sumcheck_rounds.is_empty());

        let mut v_t = Sha2T::new();
        let claim = verify_product_circuit(1, root, &proof, &mut v_t).unwrap();
        assert_eq!(claim.leaf_point.len(), 1);
        // The leaf eval at the random merge point should equal the
        // multilinear extension of the leaf vector at that point.
        let dense = dense_mle_eval(&leaves, &claim.leaf_point);
        assert_eq!(dense, claim.leaf_eval);
    }

    #[test]
    fn product_circuit_round_trip_log_n_3() {
        let mut rng = rng_for("product_log3");
        let leaves: Vec<GoldilocksExt3> = (0..8)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let expected_root: GoldilocksExt3 = leaves.iter().copied().product();

        let mut p_t = Sha2T::new();
        let (root, proof) = prove_product_circuit(&leaves, &mut p_t);
        assert_eq!(root, expected_root);
        assert_eq!(proof.layers.len(), 3);

        let mut v_t = Sha2T::new();
        let claim = verify_product_circuit(3, root, &proof, &mut v_t).unwrap();
        assert_eq!(claim.leaf_point.len(), 3);
        assert_eq!(dense_mle_eval(&leaves, &claim.leaf_point), claim.leaf_eval);
    }

    #[test]
    fn product_circuit_round_trip_log_n_5() {
        let mut rng = rng_for("product_log5");
        let leaves: Vec<GoldilocksExt3> = (0..32)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let expected_root: GoldilocksExt3 = leaves.iter().copied().product();

        let mut p_t = Sha2T::new();
        let (root, proof) = prove_product_circuit(&leaves, &mut p_t);
        assert_eq!(root, expected_root);
        assert_eq!(proof.layers.len(), 5);
        // Round counts: 0, 1, 2, 3, 4 across the five layers
        for (i, layer) in proof.layers.iter().enumerate() {
            assert_eq!(layer.sumcheck_rounds.len(), i);
        }

        let mut v_t = Sha2T::new();
        let claim = verify_product_circuit(5, root, &proof, &mut v_t).unwrap();
        assert_eq!(claim.leaf_point.len(), 5);
        assert_eq!(dense_mle_eval(&leaves, &claim.leaf_point), claim.leaf_eval);
    }

    #[test]
    fn product_circuit_rejects_wrong_root() {
        let mut rng = rng_for("product_wrong_root");
        let leaves: Vec<GoldilocksExt3> = (0..8)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let mut p_t = Sha2T::new();
        let (root, proof) = prove_product_circuit(&leaves, &mut p_t);
        let wrong_root = root + GoldilocksExt3::ONE;

        let mut v_t = Sha2T::new();
        let result = verify_product_circuit(3, wrong_root, &proof, &mut v_t);
        assert!(
            result.is_none(),
            "verifier must reject when the claimed root does not match the proof"
        );
    }

    #[test]
    fn product_circuit_rejects_tampered_left_eval() {
        let mut rng = rng_for("product_tamper_left");
        let leaves: Vec<GoldilocksExt3> = (0..8)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let mut p_t = Sha2T::new();
        let (root, mut proof) = prove_product_circuit(&leaves, &mut p_t);

        // Tamper with the first layer's left_eval.
        proof.layers[0].left_eval = proof.layers[0].left_eval + GoldilocksExt3::ONE;

        let mut v_t = Sha2T::new();
        let result = verify_product_circuit(3, root, &proof, &mut v_t);
        assert!(
            result.is_none(),
            "verifier must reject tampered layer evaluations"
        );
    }

    #[test]
    fn product_circuit_rejects_tampered_round_eval() {
        let mut rng = rng_for("product_tamper_round");
        let leaves: Vec<GoldilocksExt3> = (0..16)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let mut p_t = Sha2T::new();
        let (root, mut proof) = prove_product_circuit(&leaves, &mut p_t);

        // Tamper with a round message in a non-trivial layer.
        // Layer 2 of a log_n=4 proof has 2 sumcheck rounds.
        proof.layers[2].sumcheck_rounds[0].evals[0] =
            proof.layers[2].sumcheck_rounds[0].evals[0] + GoldilocksExt3::ONE;

        let mut v_t = Sha2T::new();
        let result = verify_product_circuit(4, root, &proof, &mut v_t);
        assert!(
            result.is_none(),
            "verifier must reject tampered sumcheck round messages"
        );
    }

    #[test]
    fn product_circuit_works_for_base_field_lifted_leaves() {
        // Sanity check that lift_leaves -> prove -> verify round-trips
        // when the leaves originate in the base Goldilocks field.
        let mut rng = rng_for("product_lift");
        let base_leaves: Vec<Goldilocks> = (0..16)
            .map(|_| Goldilocks::random_unsafe(&mut rng))
            .collect();
        let lifted: Vec<GoldilocksExt3> = lift_leaves::<Goldilocks, GoldilocksExt3>(&base_leaves);

        let expected: GoldilocksExt3 = lifted.iter().copied().product();
        let mut p_t = Sha2T::new();
        let (root, proof) = prove_product_circuit(&lifted, &mut p_t);
        assert_eq!(root, expected);

        let mut v_t = Sha2T::new();
        let claim = verify_product_circuit(4, root, &proof, &mut v_t).unwrap();
        assert_eq!(dense_mle_eval(&lifted, &claim.leaf_point), claim.leaf_eval);
    }
}
