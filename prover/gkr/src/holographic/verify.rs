//! Holographic GKR verify phase.
//!
//! Phase 2d: replays a [`HolographicProof`] against the
//! [`HolographicVerifyingKey`] produced by Phase 2b's setup. For
//! each layer, the verifier looks up the per-layer mul / add
//! wiring commitment in the VK and runs `sparse_verify_full`
//! against the matching opening in the proof, using the same
//! per-layer eval points the prover used.
//!
//! The verifier never materializes the circuit topology — only the
//! VK + the proof + the per-layer eval points. The eval points
//! typically come from the GKR per-layer sumcheck reduction; the
//! current Phase 2d takes them as a parameter for the same reason
//! Phase 2c does (the sumcheck integration lands in a follow-up
//! commit).

use arith::{ExtensionField, FFTField, SimdField};
use gkr_engine::{FieldEngine, Transcript};
use poly_commit::whir::{sparse_verify_full, SparseVerifyError, WhirCommitment};

use super::prove::{HolographicProof, LayerEvalPoint};
use super::setup::{HolographicVerifyingKey, LayerVerifyingEntry, LayerWiringCommitment};

/// Errors raised by the holographic GKR verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// The proof's per-layer opening list does not match the VK's
    /// layer count.
    LayerCountMismatch { expected: usize, got: usize },
    /// The eval-point list length does not match the VK's layer
    /// count.
    EvalPointCountMismatch { expected: usize, got: usize },
    /// A mul / add wiring is present in the VK but absent from the
    /// proof's opening (or vice versa).
    LayerOpeningSchemaMismatch { layer: usize, which: &'static str },
    /// `sparse_verify_full` rejected one of the per-layer
    /// openings.
    SparseVerify {
        layer: usize,
        which: &'static str,
        inner: SparseVerifyError,
    },
    LayerIndexMismatch {
        position: usize,
        expected: usize,
        got_proof: usize,
        got_eval: usize,
    },
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LayerCountMismatch { expected, got } => write!(
                f,
                "holographic verify: expected {expected} layer openings, got {got}"
            ),
            Self::EvalPointCountMismatch { expected, got } => write!(
                f,
                "holographic verify: expected {expected} eval points, got {got}"
            ),
            Self::LayerOpeningSchemaMismatch { layer, which } => write!(
                f,
                "holographic verify: layer {layer} {which} schema mismatch \
                 (VK and proof disagree on presence)"
            ),
            Self::SparseVerify {
                layer,
                which,
                inner,
            } => write!(
                f,
                "holographic verify: layer {layer} {which} sparse_verify rejected: {inner}"
            ),
            Self::LayerIndexMismatch {
                position,
                expected,
                got_proof,
                got_eval,
            } => write!(
                f,
                "holographic verify: layer index mismatch at position {position}: \
                 expected {expected}, proof has {got_proof}, eval_point has {got_eval}"
            ),
        }
    }
}

impl std::error::Error for VerifyError {}

/// Run the holographic GKR verifier.
///
/// Mirrors [`super::prove::prove`]: walks each layer, looks up the
/// VK commitment, and runs `sparse_verify_full` against the proof's
/// matching opening. Returns `Ok(())` on full acceptance,
/// `Err(VerifyError)` on any per-layer rejection.
///
/// # Errors
/// Returns the first per-layer error encountered. The caller can
/// inspect the error to identify which layer / kind failed.
#[allow(clippy::too_many_arguments)]
pub fn verify<C, E, T>(
    vk: &HolographicVerifyingKey,
    eval_points: &[LayerEvalPoint<E>],
    proof: &HolographicProof<E>,
    transcript: &mut T,
) -> Result<(), VerifyError>
where
    C: FieldEngine,
    C::CircuitField: FFTField + SimdField<Scalar = C::CircuitField>,
    E: ExtensionField<BaseField = C::CircuitField>
        + FFTField
        + SimdField<Scalar = E>
        + std::ops::Add<C::CircuitField, Output = E>
        + std::ops::Mul<C::CircuitField, Output = E>,
    T: Transcript,
{
    if proof.layers.len() != vk.layers.len() {
        return Err(VerifyError::LayerCountMismatch {
            expected: vk.layers.len(),
            got: proof.layers.len(),
        });
    }
    if eval_points.len() != vk.layers.len() {
        return Err(VerifyError::EvalPointCountMismatch {
            expected: vk.layers.len(),
            got: eval_points.len(),
        });
    }

    for (layer_idx, ((vk_entry, proof_entry), eval_point)) in vk
        .layers
        .iter()
        .zip(proof.layers.iter())
        .zip(eval_points.iter())
        .enumerate()
    {
        if proof_entry.layer_index != layer_idx || eval_point.layer_index != layer_idx {
            return Err(VerifyError::LayerIndexMismatch {
                position: layer_idx,
                expected: layer_idx,
                got_proof: proof_entry.layer_index,
                got_eval: eval_point.layer_index,
            });
        }
        verify_layer::<C, E, T>(layer_idx, vk_entry, proof_entry, eval_point, transcript)?;
    }

    Ok(())
}

fn verify_layer<C, E, T>(
    layer_idx: usize,
    vk_entry: &LayerVerifyingEntry,
    proof_entry: &super::prove::LayerHolographicOpening<E>,
    eval_point: &LayerEvalPoint<E>,
    transcript: &mut T,
) -> Result<(), VerifyError>
where
    C: FieldEngine,
    C::CircuitField: FFTField + SimdField<Scalar = C::CircuitField>,
    E: ExtensionField<BaseField = C::CircuitField>
        + FFTField
        + SimdField<Scalar = E>
        + std::ops::Add<C::CircuitField, Output = E>
        + std::ops::Mul<C::CircuitField, Output = E>,
    T: Transcript,
{
    // mul wiring
    match (&vk_entry.mul, &proof_entry.mul) {
        (Some(vk_mul), Some(proof_mul)) => {
            verify_one_wiring::<C, E, T>(
                layer_idx,
                "mul",
                vk_mul,
                proof_mul,
                vk_entry.n_z,
                vk_entry.n_x,
                vk_entry.n_x, // mul has n_y = n_x
                &eval_point.mul_z,
                &eval_point.mul_x,
                &eval_point.mul_y,
                poly_commit::whir::SparseArity::Three,
                transcript,
            )?;
        }
        (None, None) => {}
        _ => {
            return Err(VerifyError::LayerOpeningSchemaMismatch {
                layer: layer_idx,
                which: "mul",
            });
        }
    }

    // add wiring
    match (&vk_entry.add, &proof_entry.add) {
        (Some(vk_add), Some(proof_add)) => {
            verify_one_wiring::<C, E, T>(
                layer_idx,
                "add",
                vk_add,
                proof_add,
                vk_entry.n_z,
                vk_entry.n_x,
                0, // add has n_y = 0
                &eval_point.add_z,
                &eval_point.add_x,
                &[],
                poly_commit::whir::SparseArity::Two,
                transcript,
            )?;
        }
        (None, None) => {}
        _ => {
            return Err(VerifyError::LayerOpeningSchemaMismatch {
                layer: layer_idx,
                which: "add",
            });
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_one_wiring<C, E, T>(
    layer_idx: usize,
    which: &'static str,
    vk_wiring: &LayerWiringCommitment,
    proof_opening: &poly_commit::whir::SparseMle3FullOpening<E>,
    n_z: usize,
    n_x: usize,
    n_y: usize,
    z: &[E],
    x: &[E],
    y: &[E],
    arity: poly_commit::whir::SparseArity,
    transcript: &mut T,
) -> Result<(), VerifyError>
where
    C: FieldEngine,
    C::CircuitField: FFTField + SimdField<Scalar = C::CircuitField>,
    E: ExtensionField<BaseField = C::CircuitField>
        + FFTField
        + SimdField<Scalar = E>
        + std::ops::Add<C::CircuitField, Output = E>
        + std::ops::Mul<C::CircuitField, Output = E>,
    T: Transcript,
{
    let setup_commitment = WhirCommitment {
        root: vk_wiring.commitment.batched_root.clone(),
        num_vars: vk_wiring.commitment.batched_num_vars,
    };
    let log_nnz = vk_wiring.commitment.log_nnz;
    sparse_verify_full::<C::CircuitField, E, T>(
        &setup_commitment,
        &vk_wiring.layout,
        arity,
        log_nnz,
        n_z,
        n_x,
        n_y,
        z,
        x,
        y,
        proof_opening,
        transcript,
    )
    .map_err(|e| VerifyError::SparseVerify {
        layer: layer_idx,
        which,
        inner: e,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holographic::prove::prove;
    use crate::holographic::setup::setup;
    use circuit::{Circuit, CircuitLayer, CoefType, GateAdd, GateMul, StructureInfo};
    use gkr_engine::Goldilocksx1Config;
    type C = Goldilocksx1Config;
    use arith::Field;
    use goldilocks::{Goldilocks, GoldilocksExt4};
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

    fn make_layer(input_var_num: usize, output_var_num: usize) -> CircuitLayer<C> {
        CircuitLayer {
            input_var_num,
            output_var_num,
            input_vals: Vec::new(),
            output_vals: Vec::new(),
            mul: Vec::new(),
            add: Vec::new(),
            const_: Vec::new(),
            uni: Vec::new(),
            structure_info: StructureInfo::default(),
        }
    }

    fn mul_gate(o: usize, x: usize, y: usize, coef: u64) -> GateMul<C> {
        GateMul {
            i_ids: [x, y],
            o_id: o,
            coef_type: CoefType::Constant,
            coef: Goldilocks::from(coef),
            gate_type: 0,
        }
    }

    fn add_gate(o: usize, x: usize, coef: u64) -> GateAdd<C> {
        GateAdd {
            i_ids: [x],
            o_id: o,
            coef_type: CoefType::Constant,
            coef: Goldilocks::from(coef),
            gate_type: 0,
        }
    }

    fn build_two_layer_circuit() -> Circuit<C> {
        let mut layer0 = make_layer(2, 2);
        layer0.mul.push(mul_gate(0, 1, 2, 5));
        layer0.mul.push(mul_gate(1, 2, 3, 7));
        layer0.add.push(add_gate(2, 0, 13));
        layer0.add.push(add_gate(3, 1, 17));
        let mut layer1 = make_layer(2, 2);
        layer1.mul.push(mul_gate(3, 0, 1, 19));
        layer1.add.push(add_gate(1, 2, 23));
        Circuit {
            layers: vec![layer0, layer1],
            public_input: Vec::new(),
            expected_num_output_zeros: 0,
            rnd_coefs_identified: false,
            rnd_coefs: Vec::new(),
        }
    }

    fn random_eval_points(
        rng: &mut ChaCha20Rng,
        pk: &super::super::setup::HolographicProvingKey<C>,
    ) -> Vec<LayerEvalPoint<GoldilocksExt4>> {
        pk.layers
            .iter()
            .map(|layer| {
                let n_z = layer.n_z;
                let n_x = layer.n_x;
                let mul_z: Vec<GoldilocksExt4> = (0..n_z)
                    .map(|_| GoldilocksExt4::random_unsafe(&mut *rng))
                    .collect();
                let mul_x: Vec<GoldilocksExt4> = (0..n_x)
                    .map(|_| GoldilocksExt4::random_unsafe(&mut *rng))
                    .collect();
                let mul_y: Vec<GoldilocksExt4> = (0..n_x)
                    .map(|_| GoldilocksExt4::random_unsafe(&mut *rng))
                    .collect();
                let add_z: Vec<GoldilocksExt4> = (0..n_z)
                    .map(|_| GoldilocksExt4::random_unsafe(&mut *rng))
                    .collect();
                let add_x: Vec<GoldilocksExt4> = (0..n_x)
                    .map(|_| GoldilocksExt4::random_unsafe(&mut *rng))
                    .collect();
                let mul_claim = layer
                    .mul
                    .as_ref()
                    .map(|w| w.poly.evaluate::<GoldilocksExt4>(&mul_z, &mul_x, &mul_y))
                    .unwrap_or(GoldilocksExt4::ZERO);
                let add_claim = layer
                    .add
                    .as_ref()
                    .map(|w| w.poly.evaluate::<GoldilocksExt4>(&add_z, &add_x, &[]))
                    .unwrap_or(GoldilocksExt4::ZERO);
                LayerEvalPoint {
                    layer_index: layer.layer_index,
                    mul_z,
                    mul_x,
                    mul_y,
                    add_z,
                    add_x,
                    mul_claim,
                    add_claim,
                }
            })
            .collect()
    }

    #[test]
    fn end_to_end_two_layer_holographic_proof() {
        let mut rng = rng_for("e2e_holo_two_layer");
        let circuit = build_two_layer_circuit();
        let (pk, vk) = setup::<C>(circuit).unwrap();
        let eval_points = random_eval_points(&mut rng, &pk);

        let mut prover_t = Sha2T::new();
        let proof = prove::<C, GoldilocksExt4, Sha2T>(&pk, &eval_points, &mut prover_t).unwrap();

        let mut verifier_t = Sha2T::new();
        verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut verifier_t)
            .expect("verifier must accept honest holographic proof");
    }

    #[test]
    fn end_to_end_layer_with_only_mul_or_only_add() {
        let mut rng = rng_for("e2e_holo_partial");
        let mut layer_mul_only = make_layer(2, 2);
        layer_mul_only.mul.push(mul_gate(0, 1, 2, 5));
        let mut layer_add_only = make_layer(2, 2);
        layer_add_only.add.push(add_gate(1, 0, 7));
        let circuit: Circuit<C> = Circuit {
            layers: vec![layer_mul_only, layer_add_only],
            public_input: Vec::new(),
            expected_num_output_zeros: 0,
            rnd_coefs_identified: false,
            rnd_coefs: Vec::new(),
        };
        let (pk, vk) = setup::<C>(circuit).unwrap();
        let eval_points = random_eval_points(&mut rng, &pk);

        let mut prover_t = Sha2T::new();
        let proof = prove::<C, GoldilocksExt4, Sha2T>(&pk, &eval_points, &mut prover_t).unwrap();
        let mut verifier_t = Sha2T::new();
        verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut verifier_t)
            .expect("verifier must accept honest partial-wiring proof");
    }

    #[test]
    fn verify_rejects_layer_count_mismatch() {
        let mut rng = rng_for("verify_count_mismatch");
        let circuit = build_two_layer_circuit();
        let (pk, vk) = setup::<C>(circuit).unwrap();
        let eval_points = random_eval_points(&mut rng, &pk);
        let mut prover_t = Sha2T::new();
        let mut proof =
            prove::<C, GoldilocksExt4, Sha2T>(&pk, &eval_points, &mut prover_t).unwrap();
        proof.layers.pop(); // simulate truncated proof

        let mut verifier_t = Sha2T::new();
        let err = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut verifier_t)
            .unwrap_err();
        assert!(matches!(
            err,
            VerifyError::LayerCountMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn verify_rejects_tampered_proof_claim() {
        let mut rng = rng_for("verify_tamper_proof_claim");
        let circuit = build_two_layer_circuit();
        let (pk, vk) = setup::<C>(circuit).unwrap();
        let eval_points = random_eval_points(&mut rng, &pk);

        // Build an honest proof.
        let mut prover_t = Sha2T::new();
        let mut proof =
            prove::<C, GoldilocksExt4, Sha2T>(&pk, &eval_points, &mut prover_t).unwrap();

        // Tamper with one field of the layer-0 mul opening's
        // eval-claim sumcheck so the verifier sees an inconsistent
        // proof. We mutate the SparseFullOpening's evalclaim
        // claimed_eval directly — the eval-claim sumcheck verifier
        // will now see a starting claim that does not match the
        // prover's transcript and reject.
        if let Some(mul) = proof.layers[0].mul.as_mut() {
            let original = mul.skeleton.evalclaim.claimed_eval;
            mul.skeleton.evalclaim.claimed_eval = original + GoldilocksExt4::ONE;
        }

        let mut verifier_t = Sha2T::new();
        let result = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut verifier_t);
        assert!(
            matches!(result, Err(VerifyError::SparseVerify { .. })),
            "verifier must reject when the proof's claimed_eval is tampered; got {result:?}"
        );
    }
}
