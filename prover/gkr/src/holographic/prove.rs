//! Holographic GKR proving phase.
//!
//! Phase 2c: produce per-layer sparse-MLE openings of the wiring
//! polynomials committed in a [`HolographicProvingKey`]. Given a
//! list of per-layer evaluation points (one for the layer's mul
//! wiring and one for its add wiring), the prove routine runs
//! [`sparse_open_full`] against each layer's setup commitment and
//! per-layer scratch, producing a [`HolographicProof`] suitable for
//! the matching verifier.
//!
//! The eval points typically come from the GKR per-layer sumcheck
//! reduction (the sumcheck-random `(z, x, y)` produced when
//! reducing layer `l`'s claim to a layer `l-1` claim). Phase 2c
//! takes them as input rather than driving the sumcheck itself —
//! the sumcheck integration is layered on in the integration
//! commit. The current scope is the cryptographic core: prove that
//! the prover knows openings of the committed wiring polynomials
//! at the supplied points.
//!
//! Per-axis padding. The wiring polynomials have axis lengths
//! `(n_z, n_x, n_y) = (output_var_num, input_var_num, input_var_num)`
//! for the mul wiring and `(n_z, n_x, 0)` for the add wiring. The
//! caller must supply eval points of the right length per axis;
//! the prove routine asserts and panics on mismatch (this would
//! indicate a bug in whatever is supplying the points).

use arith::{ExtensionField, FFTField, Field, SimdField};
use gkr_engine::{FieldEngine, Transcript};
use poly_commit::whir::{sparse_open_full, SparseMle3FullOpening};

use super::setup::{HolographicProvingKey, LayerProvingWiring};

/// Eval points for one layer of a holographic GKR proof.
#[derive(Debug, Clone)]
pub struct LayerEvalPoint<E: Field> {
    pub layer_index: usize,
    /// Eval point for the mul wiring polynomial. `mul_z` has length
    /// `output_var_num`, `mul_x` and `mul_y` each have length
    /// `input_var_num`. Ignored if the layer has no mul wiring.
    pub mul_z: Vec<E>,
    pub mul_x: Vec<E>,
    pub mul_y: Vec<E>,
    /// Eval point for the add wiring polynomial. `add_z` has length
    /// `output_var_num`, `add_x` has length `input_var_num`. Ignored
    /// if the layer has no add wiring.
    pub add_z: Vec<E>,
    pub add_x: Vec<E>,
    /// Claimed evaluations the prover asserts at the supplied
    /// points. Used as the eval-claim sumcheck starting value.
    pub mul_claim: E,
    pub add_claim: E,
}

/// One layer's contribution to a holographic GKR proof.
#[derive(Debug, Clone)]
pub struct LayerHolographicOpening<E: ExtensionField> {
    pub layer_index: usize,
    pub mul: Option<SparseMle3FullOpening<E>>,
    pub add: Option<SparseMle3FullOpening<E>>,
}

/// Holographic GKR proof: per-layer wiring openings.
#[derive(Debug, Clone)]
pub struct HolographicProof<E: ExtensionField> {
    pub layers: Vec<LayerHolographicOpening<E>>,
}

/// Errors raised by the holographic GKR prover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProveError {
    /// The number of layer eval-point entries does not match the
    /// proving key's layer count.
    LayerCountMismatch { expected: usize, got: usize },
    /// The layer index in the eval-point entry does not match the
    /// position in the input vector.
    LayerIndexMismatch { position: usize, got: usize },
    /// A layer in the proving key has a mul or add wiring but the
    /// corresponding eval point in the input is missing or has the
    /// wrong length.
    EvalPointShapeMismatch {
        layer: usize,
        which: &'static str,
        expected_n_z: usize,
        expected_n_x: usize,
        expected_n_y: usize,
    },
}

impl std::fmt::Display for ProveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LayerCountMismatch { expected, got } => write!(
                f,
                "holographic prove: expected {expected} layer eval points, got {got}"
            ),
            Self::LayerIndexMismatch { position, got } => write!(
                f,
                "holographic prove: eval point at position {position} reports layer index {got}"
            ),
            Self::EvalPointShapeMismatch {
                layer,
                which,
                expected_n_z,
                expected_n_x,
                expected_n_y,
            } => write!(
                f,
                "holographic prove: layer {layer} {which} eval point shape mismatch \
                 (expected n_z={expected_n_z}, n_x={expected_n_x}, n_y={expected_n_y})"
            ),
        }
    }
}

impl std::error::Error for ProveError {}

/// Run the holographic GKR prover.
///
/// For each layer in the proving key, opens the layer's mul and
/// add wiring polynomials at the corresponding entry of
/// `eval_points`, producing a [`HolographicProof`] the verifier
/// can replay against the verifying key.
///
/// # Errors
/// Returns [`ProveError`] if the eval-point list is the wrong
/// length, has the wrong layer indices, or has eval points with
/// the wrong shape for any layer's wiring.
pub fn prove<C, E, T>(
    pk: &HolographicProvingKey<C>,
    eval_points: &[LayerEvalPoint<E>],
    transcript: &mut T,
) -> Result<HolographicProof<E>, ProveError>
where
    C: FieldEngine,
    C::CircuitField: FFTField + SimdField<Scalar = C::CircuitField>,
    E: ExtensionField<BaseField = C::CircuitField>
        + FFTField
        + SimdField<Scalar = E>
        + std::ops::Add<C::CircuitField, Output = E>
        + std::ops::Mul<C::CircuitField, Output = E>,
    T: Transcript + Clone,
{
    if eval_points.len() != pk.layers.len() {
        return Err(ProveError::LayerCountMismatch {
            expected: pk.layers.len(),
            got: eval_points.len(),
        });
    }

    let mut layer_openings = Vec::with_capacity(pk.layers.len());
    for (position, (layer_pk, eval_point)) in pk.layers.iter().zip(eval_points.iter()).enumerate() {
        if eval_point.layer_index != position {
            return Err(ProveError::LayerIndexMismatch {
                position,
                got: eval_point.layer_index,
            });
        }

        let mul_opening = if let Some(mul_wiring) = &layer_pk.mul {
            check_mul_eval_point_shape(layer_pk.layer_index, mul_wiring, eval_point)?;
            Some(open_layer_wiring::<C::CircuitField, E, T>(
                mul_wiring,
                eval_point.mul_claim,
                &eval_point.mul_z,
                &eval_point.mul_x,
                &eval_point.mul_y,
                transcript,
            ))
        } else {
            None
        };

        let add_opening = if let Some(add_wiring) = &layer_pk.add {
            check_add_eval_point_shape(layer_pk.layer_index, add_wiring, eval_point)?;
            Some(open_layer_wiring::<C::CircuitField, E, T>(
                add_wiring,
                eval_point.add_claim,
                &eval_point.add_z,
                &eval_point.add_x,
                &[],
                transcript,
            ))
        } else {
            None
        };

        layer_openings.push(LayerHolographicOpening {
            layer_index: layer_pk.layer_index,
            mul: mul_opening,
            add: add_opening,
        });
    }

    Ok(HolographicProof {
        layers: layer_openings,
    })
}

fn open_layer_wiring<F, E, T>(
    wiring: &LayerProvingWiring<F>,
    claimed_eval: E,
    z: &[E],
    x: &[E],
    y: &[E],
    transcript: &mut T,
) -> SparseMle3FullOpening<E>
where
    F: FFTField + SimdField<Scalar = F>,
    E: ExtensionField<BaseField = F>
        + FFTField
        + SimdField<Scalar = E>
        + std::ops::Add<F, Output = E>
        + std::ops::Mul<F, Output = E>,
    T: Transcript + Clone,
{
    let whir_commitment = poly_commit::whir::WhirCommitment {
        root: wiring.tree.root(),
        num_vars: wiring.layout.total_vars,
    };
    let (full, _open_scratch) = sparse_open_full::<F, E, T>(
        &whir_commitment,
        &wiring.combined_evals,
        &wiring.codeword,
        &wiring.tree,
        &wiring.layout,
        &wiring.scratch,
        claimed_eval,
        z,
        x,
        y,
        transcript,
    );
    full
}

fn check_mul_eval_point_shape<F: Field>(
    layer_idx: usize,
    wiring: &LayerProvingWiring<F>,
    eval_point: &LayerEvalPoint<impl Field>,
) -> Result<(), ProveError> {
    if eval_point.mul_z.len() != wiring.poly.n_z
        || eval_point.mul_x.len() != wiring.poly.n_x
        || eval_point.mul_y.len() != wiring.poly.n_y
    {
        return Err(ProveError::EvalPointShapeMismatch {
            layer: layer_idx,
            which: "mul",
            expected_n_z: wiring.poly.n_z,
            expected_n_x: wiring.poly.n_x,
            expected_n_y: wiring.poly.n_y,
        });
    }
    Ok(())
}

fn check_add_eval_point_shape<F: Field>(
    layer_idx: usize,
    wiring: &LayerProvingWiring<F>,
    eval_point: &LayerEvalPoint<impl Field>,
) -> Result<(), ProveError> {
    if eval_point.add_z.len() != wiring.poly.n_z || eval_point.add_x.len() != wiring.poly.n_x {
        return Err(ProveError::EvalPointShapeMismatch {
            layer: layer_idx,
            which: "add",
            expected_n_z: wiring.poly.n_z,
            expected_n_x: wiring.poly.n_x,
            expected_n_y: 0,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
        pk: &HolographicProvingKey<C>,
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
                // Compute the actual claim from the proving key's
                // sparse polynomials so the eval-claim sumcheck
                // starts from a consistent value.
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
    fn prove_two_layer_circuit_produces_per_layer_openings() {
        let mut rng = rng_for("prove_two_layer");
        let circuit = build_two_layer_circuit();
        let (pk, _vk) = setup::<C>(circuit).unwrap();
        let eval_points = random_eval_points(&mut rng, &pk);

        let mut transcript = Sha2T::new();
        let proof = prove::<C, GoldilocksExt4, Sha2T>(&pk, &eval_points, &mut transcript)
            .expect("prove must succeed on honest inputs");

        assert_eq!(proof.layers.len(), 2);
        // Layer 0 has both mul and add wiring
        assert!(proof.layers[0].mul.is_some());
        assert!(proof.layers[0].add.is_some());
        // Layer 1 has both as well
        assert!(proof.layers[1].mul.is_some());
        assert!(proof.layers[1].add.is_some());
    }

    #[test]
    fn prove_rejects_layer_count_mismatch() {
        let mut rng = rng_for("prove_count_mismatch");
        let circuit = build_two_layer_circuit();
        let (pk, _vk) = setup::<C>(circuit).unwrap();
        let mut eval_points = random_eval_points(&mut rng, &pk);
        eval_points.pop();
        let mut transcript = Sha2T::new();
        let err =
            prove::<C, GoldilocksExt4, Sha2T>(&pk, &eval_points, &mut transcript).unwrap_err();
        assert!(matches!(
            err,
            ProveError::LayerCountMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn prove_rejects_layer_index_mismatch() {
        let mut rng = rng_for("prove_index_mismatch");
        let circuit = build_two_layer_circuit();
        let (pk, _vk) = setup::<C>(circuit).unwrap();
        let mut eval_points = random_eval_points(&mut rng, &pk);
        eval_points[1].layer_index = 99;
        let mut transcript = Sha2T::new();
        let err =
            prove::<C, GoldilocksExt4, Sha2T>(&pk, &eval_points, &mut transcript).unwrap_err();
        assert!(matches!(
            err,
            ProveError::LayerIndexMismatch {
                position: 1,
                got: 99
            }
        ));
    }

    #[test]
    fn prove_rejects_eval_point_shape_mismatch() {
        let mut rng = rng_for("prove_shape_mismatch");
        let circuit = build_two_layer_circuit();
        let (pk, _vk) = setup::<C>(circuit).unwrap();
        let mut eval_points = random_eval_points(&mut rng, &pk);
        eval_points[0].mul_z.push(GoldilocksExt4::ZERO); // wrong length
        let mut transcript = Sha2T::new();
        let err =
            prove::<C, GoldilocksExt4, Sha2T>(&pk, &eval_points, &mut transcript).unwrap_err();
        assert!(matches!(
            err,
            ProveError::EvalPointShapeMismatch {
                layer: 0,
                which: "mul",
                ..
            }
        ));
    }

    #[test]
    fn prove_layer_with_only_mul_or_only_add() {
        let mut rng = rng_for("prove_partial");
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
        let (pk, _vk) = setup::<C>(circuit).unwrap();
        let eval_points = random_eval_points(&mut rng, &pk);
        let mut transcript = Sha2T::new();
        let proof = prove::<C, GoldilocksExt4, Sha2T>(&pk, &eval_points, &mut transcript).unwrap();
        assert!(proof.layers[0].mul.is_some());
        assert!(proof.layers[0].add.is_none());
        assert!(proof.layers[1].mul.is_none());
        assert!(proof.layers[1].add.is_some());
    }
}
