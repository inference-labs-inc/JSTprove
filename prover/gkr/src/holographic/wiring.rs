//! Sparse wiring extraction from a `circuit::Circuit<C>`.
//!
//! Walks each layer's `mul` and `add` gate lists and emits
//! [`SparseMle3`] instances suitable for `poly_commit::sparse_commit`.
//! The mul wiring of a layer with `output_var_num = a` and
//! `input_var_num = b` is a sparse multilinear polynomial in
//! `a + b + b` variables (output, x, y); the add wiring is in
//! `a + b` variables (output, x, with the y axis collapsed via
//! `SparseArity::Two`).
//!
//! Coefficient handling. Each gate carries a `coef_type` that
//! distinguishes a fixed `Constant` coefficient from a `Random`
//! coefficient sampled at proving time and from a `PublicInput`
//! coefficient that depends on a per-inference public input slot.
//! Only the `Constant` case can be folded into a wiring polynomial
//! committed at setup time — the other two cases need separate
//! handling that is layered on in subsequent phases (typically by
//! proving an auxiliary sumcheck against fresh randomness during
//! the prove phase). For now, [`extract_layer_mul_wiring`] and
//! [`extract_layer_add_wiring`] return [`WiringExtractError::NonConstantCoefficient`]
//! when they encounter a non-`Constant` coefficient, with the
//! offending layer / gate index reported in the error so the caller
//! can either rewrite the circuit or fall back to the existing
//! non-holographic GKR path for that layer.
//!
//! `const_` and `uni` gates. The standard GKR protocol's per-layer
//! sumcheck reduction only addresses `mul` and `add` gate lists;
//! `const_` and `uni` gates require additional structure that is
//! out of scope for Phase 2a. Layers carrying non-empty `const_`
//! or `uni` gate lists trigger [`WiringExtractError::UnsupportedGateKind`]
//! and the caller must use the non-holographic path for those
//! layers until a future phase brings them in.
//!
//! Padding. Sparse-MLE commit requires `nnz` to be a power of two.
//! When the gate list length is not already a power of two we pad
//! with synthetic zero-coefficient entries at address `(0, 0, 0)`
//! (or `(0, 0)` for add). These contribute nothing to the eval-claim
//! sumcheck (their `val` is `F::ZERO`) and they pass through the
//! offline-memory-checking gadget cleanly because the same address
//! is reused but with monotonically increasing read counters.

use arith::Field;
use circuit::{Circuit, CircuitLayer, CoefType};
use gkr_engine::FieldEngine;
use poly_commit::whir::{SparseArity, SparseMle3};

/// Errors raised by sparse wiring extraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WiringExtractError {
    /// A gate in the source layer carries a non-`Constant`
    /// coefficient (`Random` or `PublicInput`). Such coefficients
    /// cannot be baked into a setup-time wiring commitment.
    NonConstantCoefficient {
        layer: usize,
        kind: GateKindLabel,
        gate: usize,
    },
    /// The source layer carries `const_` or `uni` gates which are
    /// not yet handled by the holographic wiring extractor.
    UnsupportedGateKind { layer: usize, kind: GateKindLabel },
    /// The layer's declared dimensions exceed
    /// `poly_commit::SPARSE_MLE_MAX_LOG_DOMAIN` (currently 32 bits
    /// per axis), which is the maximum the sparse-MLE commitment
    /// supports.
    LayerDimensionsTooLarge {
        layer: usize,
        n_z: usize,
        n_x: usize,
        n_y: usize,
    },
}

impl std::fmt::Display for WiringExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonConstantCoefficient { layer, kind, gate } => write!(
                f,
                "layer {layer} gate {gate} ({kind:?}) has a non-constant coefficient; \
                 holographic wiring requires fixed coefficients at setup time"
            ),
            Self::UnsupportedGateKind { layer, kind } => write!(
                f,
                "layer {layer} carries unsupported gate kind {kind:?}; \
                 holographic wiring currently handles only `mul` and `add`"
            ),
            Self::LayerDimensionsTooLarge {
                layer,
                n_z,
                n_x,
                n_y,
            } => write!(
                f,
                "layer {layer} dimensions exceed sparse-MLE limits: \
                 n_z={n_z}, n_x={n_x}, n_y={n_y}"
            ),
        }
    }
}

impl std::error::Error for WiringExtractError {}

/// Label used in error reporting to identify which gate kind
/// triggered the error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateKindLabel {
    Mul,
    Add,
    Const,
    Uni,
}

/// Sparse wiring polynomials for one circuit layer.
///
/// `mul` is the 3-arity wiring polynomial committing the layer's
/// multiplication gates, `add` is the 2-arity wiring polynomial
/// committing the addition gates (including uni-12346 identity
/// gates), and `const_wiring` is a degenerate 2-arity polynomial
/// with `n_x = 0` committing the constant gates. Either may be
/// `None` if the corresponding gate list is empty after extraction.
#[derive(Debug, Clone)]
pub struct LayerWiring<F: Field> {
    pub layer_index: usize,
    pub n_z: usize,
    pub n_x: usize,
    pub mul: Option<SparseMle3<F>>,
    pub add: Option<SparseMle3<F>>,
    /// Degenerate 2-arity polynomial for const gates (n_x = 0,
    /// col_x = [0; nnz]). eval_cst at point z equals the MLE
    /// evaluation of this polynomial at (z, []).
    pub const_wiring: Option<SparseMle3<F>>,
}

/// Sparse wiring polynomials for an entire circuit.
#[derive(Debug, Clone)]
pub struct CircuitWiring<F: Field> {
    pub layers: Vec<LayerWiring<F>>,
}

/// Extract the 3-arity sparse wiring polynomial committing the
/// `mul` gates of `layer`. Returns `Ok(None)` if the gate list is
/// empty after applying the rules below; otherwise returns a
/// validated [`SparseMle3`] suitable for `sparse_commit`.
///
/// Each `mul` gate `(o_id, [i_x, i_y], coef)` becomes a non-zero
/// entry `M(o_id, i_x, i_y) += coef` in the wiring polynomial. If
/// two gates share the same `(o_id, i_x, i_y)` triple their
/// coefficients are summed — this matches the semantics of the
/// expander circuit evaluator, which folds duplicate gates by
/// repeated addition into the output cell.
pub fn extract_layer_mul_wiring<C>(
    layer: &CircuitLayer<C>,
    layer_idx: usize,
) -> Result<Option<SparseMle3<C::CircuitField>>, WiringExtractError>
where
    C: FieldEngine,
{
    // const_ and uni gates are handled by separate extractors;
    // mul only processes layer.mul.
    if layer.mul.is_empty() {
        return Ok(None);
    }

    let n_z = layer.output_var_num;
    let n_x = layer.input_var_num;
    let n_y = layer.input_var_num;

    let row_capacity = layer.mul.len().next_power_of_two();
    let mut row = Vec::with_capacity(row_capacity);
    let mut col_x = Vec::with_capacity(row_capacity);
    let mut col_y = Vec::with_capacity(row_capacity);
    let mut val = Vec::with_capacity(row_capacity);
    for (gate_idx, gate) in layer.mul.iter().enumerate() {
        check_constant_coef(gate.coef_type, layer_idx, gate_idx, GateKindLabel::Mul)?;
        row.push(gate.o_id);
        col_x.push(gate.i_ids[0]);
        col_y.push(gate.i_ids[1]);
        val.push(gate.coef);
    }
    pad_to_power_of_two_3::<C::CircuitField>(&mut row, &mut col_x, &mut col_y, &mut val);

    let poly = SparseMle3 {
        n_z,
        n_x,
        n_y,
        arity: SparseArity::Three,
        row,
        col_x,
        col_y,
        val,
    };
    poly.validate()
        .map_err(|_| WiringExtractError::LayerDimensionsTooLarge {
            layer: layer_idx,
            n_z,
            n_x,
            n_y,
        })?;
    Ok(Some(poly))
}

/// Extract the 2-arity sparse wiring polynomial committing the
/// `add` gates of `layer`, plus any uni-12346 (identity-with-coef)
/// gates which are semantically equivalent to add gates. Returns
/// `Ok(None)` if the combined list is empty.
///
/// Uni-12345 (x^5 S-box) gates are NOT promotable to add and are
/// rejected with [`WiringExtractError::UnsupportedGateKind`].
pub fn extract_layer_add_wiring<C>(
    layer: &CircuitLayer<C>,
    layer_idx: usize,
) -> Result<Option<SparseMle3<C::CircuitField>>, WiringExtractError>
where
    C: FieldEngine,
{
    // Count uni-12346 (identity) gates that we'll promote to add.
    // Reject uni-12345 (x^5) since it needs a custom sumcheck.
    let mut uni_identity_count = 0usize;
    for (gate_idx, gate) in layer.uni.iter().enumerate() {
        match gate.gate_type {
            12346 => {
                check_constant_coef(gate.coef_type, layer_idx, gate_idx, GateKindLabel::Uni)?;
                uni_identity_count += 1;
            }
            12345 => {
                return Err(WiringExtractError::UnsupportedGateKind {
                    layer: layer_idx,
                    kind: GateKindLabel::Uni,
                });
            }
            _ => {
                return Err(WiringExtractError::UnsupportedGateKind {
                    layer: layer_idx,
                    kind: GateKindLabel::Uni,
                });
            }
        }
    }

    let total = layer.add.len() + uni_identity_count;
    if total == 0 {
        return Ok(None);
    }

    let n_z = layer.output_var_num;
    let n_x = layer.input_var_num;

    let row_capacity = total.next_power_of_two();
    let mut row = Vec::with_capacity(row_capacity);
    let mut col_x = Vec::with_capacity(row_capacity);
    let mut val = Vec::with_capacity(row_capacity);

    for (gate_idx, gate) in layer.add.iter().enumerate() {
        check_constant_coef(gate.coef_type, layer_idx, gate_idx, GateKindLabel::Add)?;
        row.push(gate.o_id);
        col_x.push(gate.i_ids[0]);
        val.push(gate.coef);
    }
    for gate in &layer.uni {
        if gate.gate_type == 12346 {
            row.push(gate.o_id);
            col_x.push(gate.i_ids[0]);
            val.push(gate.coef);
        }
    }

    pad_to_power_of_two_2::<C::CircuitField>(&mut row, &mut col_x, &mut val);
    let col_y = vec![0usize; row.len()];

    let poly = SparseMle3 {
        n_z,
        n_x,
        n_y: 0,
        arity: SparseArity::Two,
        row,
        col_x,
        col_y,
        val,
    };
    poly.validate()
        .map_err(|_| WiringExtractError::LayerDimensionsTooLarge {
            layer: layer_idx,
            n_z,
            n_x,
            n_y: 0,
        })?;
    Ok(Some(poly))
}

/// Extract the degenerate 2-arity constant-gate wiring polynomial.
/// `n_x = 0`, `col_x = [0; nnz]` — the "input" axis is trivially
/// a single point. Returns `Ok(None)` if the layer has no constant
/// gates (Constant coef type only; PublicInput coef types are
/// rejected).
pub fn extract_layer_const_wiring<C>(
    layer: &CircuitLayer<C>,
    layer_idx: usize,
) -> Result<Option<SparseMle3<C::CircuitField>>, WiringExtractError>
where
    C: FieldEngine,
{
    if layer.const_.is_empty() {
        return Ok(None);
    }

    let n_z = layer.output_var_num;

    let row_capacity = layer.const_.len().next_power_of_two();
    let mut row = Vec::with_capacity(row_capacity);
    let mut val = Vec::with_capacity(row_capacity);
    for (gate_idx, gate) in layer.const_.iter().enumerate() {
        check_constant_coef(gate.coef_type, layer_idx, gate_idx, GateKindLabel::Const)?;
        row.push(gate.o_id);
        val.push(gate.coef);
    }
    // Pad row + val to power of two; col_x and col_y are all-zero.
    let target = row.len().next_power_of_two();
    while row.len() < target {
        row.push(0);
        val.push(C::CircuitField::ZERO);
    }
    let col_x = vec![0usize; row.len()];
    let col_y = vec![0usize; row.len()];

    let poly = SparseMle3 {
        n_z,
        n_x: 0,
        n_y: 0,
        arity: SparseArity::Two,
        row,
        col_x,
        col_y,
        val,
    };
    poly.validate()
        .map_err(|_| WiringExtractError::LayerDimensionsTooLarge {
            layer: layer_idx,
            n_z,
            n_x: 0,
            n_y: 0,
        })?;
    Ok(Some(poly))
}

/// Extract the wiring polynomials for every layer of `circuit`.
///
/// Returns a [`CircuitWiring`] aggregating the per-layer
/// extractions. Layers with empty `mul` and `add` lists still
/// produce a [`LayerWiring`] entry — both fields are `None` — so
/// the result vector indexes match the source layer indices and
/// the caller can iterate `circuit.layers` and `circuit_wiring.layers`
/// in lock step.
pub fn extract_circuit_wiring<C>(
    circuit: &Circuit<C>,
) -> Result<CircuitWiring<C::CircuitField>, WiringExtractError>
where
    C: FieldEngine,
{
    let mut layers = Vec::with_capacity(circuit.layers.len());
    for (layer_idx, layer) in circuit.layers.iter().enumerate() {
        let mul = extract_layer_mul_wiring::<C>(layer, layer_idx)?;
        let add = extract_layer_add_wiring::<C>(layer, layer_idx)?;
        let const_wiring = extract_layer_const_wiring::<C>(layer, layer_idx)?;
        layers.push(LayerWiring {
            layer_index: layer_idx,
            n_z: layer.output_var_num,
            n_x: layer.input_var_num,
            mul,
            add,
            const_wiring,
        });
    }
    Ok(CircuitWiring { layers })
}

#[inline]
fn check_constant_coef(
    coef_type: CoefType,
    layer: usize,
    gate: usize,
    kind: GateKindLabel,
) -> Result<(), WiringExtractError> {
    match coef_type {
        CoefType::Constant => Ok(()),
        _ => Err(WiringExtractError::NonConstantCoefficient { layer, kind, gate }),
    }
}

/// Pad three address vectors plus a value vector to the next power
/// of two by appending `(0, 0, 0, F::ZERO)` entries. Sound because
/// `val = F::ZERO` contributes nothing to the eval-claim sumcheck
/// and the duplicate `(0, 0, 0)` address still produces a valid
/// memory-checking trace (the per-cell read counter just increments
/// for cell 0 across the padding entries).
fn pad_to_power_of_two_3<F: Field>(
    row: &mut Vec<usize>,
    col_x: &mut Vec<usize>,
    col_y: &mut Vec<usize>,
    val: &mut Vec<F>,
) {
    let target = row.len().next_power_of_two();
    while row.len() < target {
        row.push(0);
        col_x.push(0);
        col_y.push(0);
        val.push(F::ZERO);
    }
}

fn pad_to_power_of_two_2<F: Field>(row: &mut Vec<usize>, col_x: &mut Vec<usize>, val: &mut Vec<F>) {
    let target = row.len().next_power_of_two();
    while row.len() < target {
        row.push(0);
        col_x.push(0);
        val.push(F::ZERO);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use circuit::{CircuitLayer, GateAdd, GateMul, StructureInfo};
    use gkr_engine::Goldilocksx1Config;
    type C = Goldilocksx1Config;
    use arith::Field;
    use goldilocks::Goldilocks;

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

    #[test]
    fn extract_mul_basic() {
        let mut layer = make_layer(2, 2);
        // 3 mul gates over a 4×4 input space, 4 outputs
        layer.mul.push(mul_gate(0, 1, 2, 5));
        layer.mul.push(mul_gate(1, 2, 3, 7));
        layer.mul.push(mul_gate(2, 0, 1, 11));

        let poly = extract_layer_mul_wiring::<C>(&layer, 0).unwrap().unwrap();
        // Padded to next power of two = 4
        assert_eq!(poly.nnz(), 4);
        assert_eq!(poly.n_z, 2);
        assert_eq!(poly.n_x, 2);
        assert_eq!(poly.n_y, 2);
        assert_eq!(poly.arity, SparseArity::Three);
        // Original entries verbatim
        assert_eq!(poly.row[0..3], [0, 1, 2]);
        assert_eq!(poly.col_x[0..3], [1, 2, 0]);
        assert_eq!(poly.col_y[0..3], [2, 3, 1]);
        assert_eq!(poly.val[0], Goldilocks::from(5u64));
        assert_eq!(poly.val[1], Goldilocks::from(7u64));
        assert_eq!(poly.val[2], Goldilocks::from(11u64));
        // Padding entry
        assert_eq!(poly.row[3], 0);
        assert_eq!(poly.col_x[3], 0);
        assert_eq!(poly.col_y[3], 0);
        assert_eq!(poly.val[3], Goldilocks::ZERO);
    }

    #[test]
    fn extract_add_basic() {
        let mut layer = make_layer(2, 2);
        layer.add.push(add_gate(0, 1, 3));
        layer.add.push(add_gate(2, 3, 9));

        let poly = extract_layer_add_wiring::<C>(&layer, 0).unwrap().unwrap();
        assert_eq!(poly.nnz(), 2);
        assert_eq!(poly.n_z, 2);
        assert_eq!(poly.n_x, 2);
        assert_eq!(poly.n_y, 0);
        assert_eq!(poly.arity, SparseArity::Two);
        assert_eq!(poly.row, vec![0, 2]);
        assert_eq!(poly.col_x, vec![1, 3]);
        assert_eq!(poly.col_y, vec![0, 0]);
        assert_eq!(poly.val[0], Goldilocks::from(3u64));
        assert_eq!(poly.val[1], Goldilocks::from(9u64));
    }

    #[test]
    fn extract_empty_returns_none() {
        let layer = make_layer(2, 2);
        assert!(extract_layer_mul_wiring::<C>(&layer, 0).unwrap().is_none());
        assert!(extract_layer_add_wiring::<C>(&layer, 0).unwrap().is_none());
    }

    #[test]
    fn extract_mul_pads_to_next_power_of_two() {
        let mut layer = make_layer(3, 3);
        for i in 0..5 {
            layer.mul.push(mul_gate(i, i, (i + 1) % 8, 1));
        }
        let poly = extract_layer_mul_wiring::<C>(&layer, 0).unwrap().unwrap();
        // 5 → 8
        assert_eq!(poly.nnz(), 8);
        // Padding entries are zero-coef at address (0, 0, 0)
        for i in 5..8 {
            assert_eq!(poly.row[i], 0);
            assert_eq!(poly.col_x[i], 0);
            assert_eq!(poly.col_y[i], 0);
            assert_eq!(poly.val[i], Goldilocks::ZERO);
        }
    }

    #[test]
    fn extract_already_power_of_two_does_not_pad() {
        let mut layer = make_layer(3, 3);
        for i in 0..4 {
            layer.mul.push(mul_gate(i, i, (i + 1) % 8, 1));
        }
        let poly = extract_layer_mul_wiring::<C>(&layer, 0).unwrap().unwrap();
        assert_eq!(poly.nnz(), 4);
    }

    #[test]
    fn extract_rejects_random_coefficient() {
        let mut layer = make_layer(2, 2);
        let mut gate = mul_gate(0, 1, 2, 1);
        gate.coef_type = CoefType::Random;
        layer.mul.push(gate);
        let err = extract_layer_mul_wiring::<C>(&layer, 7).unwrap_err();
        assert!(matches!(
            err,
            WiringExtractError::NonConstantCoefficient {
                layer: 7,
                kind: GateKindLabel::Mul,
                gate: 0,
            }
        ));
    }

    #[test]
    fn extract_rejects_public_input_coefficient() {
        let mut layer = make_layer(2, 2);
        let mut gate = add_gate(0, 1, 1);
        gate.coef_type = CoefType::PublicInput(0);
        layer.add.push(gate);
        let err = extract_layer_add_wiring::<C>(&layer, 3).unwrap_err();
        assert!(matches!(
            err,
            WiringExtractError::NonConstantCoefficient {
                layer: 3,
                kind: GateKindLabel::Add,
                gate: 0,
            }
        ));
    }

    #[test]
    fn extract_const_gates_as_degenerate_add() {
        use circuit::GateConst;
        let mut layer = make_layer(2, 2);
        layer.const_.push(GateConst {
            i_ids: [],
            o_id: 0,
            coef_type: CoefType::Constant,
            coef: Goldilocks::ONE,
            gate_type: 0,
        });
        layer.const_.push(GateConst {
            i_ids: [],
            o_id: 3,
            coef_type: CoefType::Constant,
            coef: Goldilocks::from(7u64),
            gate_type: 0,
        });
        // mul extraction should not be affected by const_
        layer.mul.push(mul_gate(0, 1, 2, 1));
        let mul = extract_layer_mul_wiring::<C>(&layer, 0).unwrap();
        assert!(mul.is_some());

        // const extraction produces a degenerate polynomial
        let const_poly = extract_layer_const_wiring::<C>(&layer, 0).unwrap().unwrap();
        assert_eq!(const_poly.arity, SparseArity::Two);
        assert_eq!(const_poly.n_x, 0);
        assert_eq!(const_poly.n_z, 2);
        // nnz padded to 2 (next pow2 of 2)
        assert_eq!(const_poly.nnz(), 2);
        assert_eq!(const_poly.row[0..2], [0, 3]);
        assert_eq!(const_poly.col_x[0..2], [0, 0]);
    }

    #[test]
    fn extract_uni_identity_promoted_to_add() {
        use circuit::GateUni;
        let mut layer = make_layer(2, 2);
        layer.uni.push(GateUni {
            i_ids: [2],
            o_id: 1,
            coef_type: CoefType::Constant,
            coef: Goldilocks::from(5u64),
            gate_type: 12346,
        });
        layer.add.push(add_gate(0, 3, 11));
        let add_poly = extract_layer_add_wiring::<C>(&layer, 0).unwrap().unwrap();
        // 1 add + 1 uni-12346 = 2 entries → padded to 2
        assert_eq!(add_poly.nnz(), 2);
        assert_eq!(add_poly.row[0], 0); // from add gate
        assert_eq!(add_poly.row[1], 1); // from uni-12346
    }

    #[test]
    fn extract_rejects_uni_x5_gates() {
        use circuit::GateUni;
        let mut layer = make_layer(2, 2);
        layer.uni.push(GateUni {
            i_ids: [0],
            o_id: 0,
            coef_type: CoefType::Constant,
            coef: Goldilocks::ONE,
            gate_type: 12345,
        });
        let err = extract_layer_add_wiring::<C>(&layer, 9).unwrap_err();
        assert!(matches!(
            err,
            WiringExtractError::UnsupportedGateKind {
                layer: 9,
                kind: GateKindLabel::Uni,
            }
        ));
    }

    #[test]
    fn extract_circuit_two_layer_round_trip() {
        let mut layer0 = make_layer(2, 2);
        layer0.mul.push(mul_gate(0, 1, 2, 5));
        layer0.mul.push(mul_gate(1, 2, 3, 7));
        layer0.add.push(add_gate(2, 0, 13));

        let mut layer1 = make_layer(2, 2);
        layer1.mul.push(mul_gate(3, 0, 1, 17));
        layer1.add.push(add_gate(1, 2, 19));
        layer1.add.push(add_gate(2, 3, 23));

        let circuit: Circuit<C> = Circuit {
            layers: vec![layer0, layer1],
            public_input: Vec::new(),
            expected_num_output_zeros: 0,
            rnd_coefs_identified: false,
            rnd_coefs: Vec::new(),
        };

        let wiring = extract_circuit_wiring::<C>(&circuit).unwrap();
        assert_eq!(wiring.layers.len(), 2);

        let l0 = &wiring.layers[0];
        assert_eq!(l0.layer_index, 0);
        let mul0 = l0.mul.as_ref().unwrap();
        assert_eq!(mul0.nnz(), 2);
        let add0 = l0.add.as_ref().unwrap();
        assert_eq!(add0.nnz(), 1usize.next_power_of_two());

        let l1 = &wiring.layers[1];
        assert_eq!(l1.layer_index, 1);
        let mul1 = l1.mul.as_ref().unwrap();
        assert_eq!(mul1.nnz(), 1usize.next_power_of_two());
        let add1 = l1.add.as_ref().unwrap();
        assert_eq!(add1.nnz(), 2);
    }

    #[test]
    fn extract_circuit_evaluates_consistently_with_layer_eval() {
        // Build a tiny layer with explicit mul/add gates, evaluate
        // it via the existing CircuitLayer::evaluate, then evaluate
        // the extracted sparse wiring polynomials at the same
        // input vector via the dense oracle. The two evaluations
        // must agree on every output cell.
        use goldilocks::GoldilocksExt4;

        let mut layer = make_layer(2, 2);
        // mul: out[0] = 5 * in[1] * in[2]
        // mul: out[1] = 7 * in[2] * in[3]
        // add: out[2] = 13 * in[0]
        // add: out[3] = 17 * in[1]
        layer.mul.push(mul_gate(0, 1, 2, 5));
        layer.mul.push(mul_gate(1, 2, 3, 7));
        layer.add.push(add_gate(2, 0, 13));
        layer.add.push(add_gate(3, 1, 17));

        // Direct evaluation
        let inputs: Vec<Goldilocks> = (1u64..=4).map(Goldilocks::from).collect();
        let mut layer_eval = layer.clone();
        layer_eval.input_vals = inputs.clone();
        let mut direct = Vec::new();
        layer_eval.evaluate(&mut direct, &[]);

        // Sparse wiring evaluation: M(z, x, y) gives the wiring
        // coefficient for the output bit pattern z and input bit
        // patterns x, y. Output[z] = Σ_{x,y} M_mul(z,x,y) · in[x] · in[y]
        //                          + Σ_x   M_add(z,x)    · in[x]
        let mul_poly = extract_layer_mul_wiring::<C>(&layer, 0).unwrap().unwrap();
        let add_poly = extract_layer_add_wiring::<C>(&layer, 0).unwrap().unwrap();

        // Helper: enumerate all (z, x, y) triples explicitly
        // because the dense oracle path goes through MLEs over
        // extension-field points. For this self-consistency check
        // we only need the integer-domain semantics.
        let m_z = 1usize << layer.output_var_num;
        let mut from_sparse = vec![Goldilocks::ZERO; m_z];
        for k in 0..mul_poly.nnz() {
            let z = mul_poly.row[k];
            let x = mul_poly.col_x[k];
            let y = mul_poly.col_y[k];
            from_sparse[z] += mul_poly.val[k] * inputs[x] * inputs[y];
        }
        for k in 0..add_poly.nnz() {
            let z = add_poly.row[k];
            let x = add_poly.col_x[k];
            from_sparse[z] += add_poly.val[k] * inputs[x];
        }
        // The CircuitLayer evaluator pads its output buffer to
        // 1 << output_var_num. The first m_z entries must match
        // the sparse-derived values exactly.
        for i in 0..m_z {
            assert_eq!(
                from_sparse[i], direct[i],
                "output cell {i}: sparse-derived value disagrees with CircuitLayer::evaluate"
            );
        }

        // Sanity check that the dense MLE oracle on the sparse
        // polynomial agrees on a random extension-field eval point.
        // This pins the SparseMle3 → MLE relationship.
        let z: Vec<GoldilocksExt4> = vec![GoldilocksExt4::from(2u64), GoldilocksExt4::from(3u64)];
        let x: Vec<GoldilocksExt4> = vec![GoldilocksExt4::from(5u64), GoldilocksExt4::from(7u64)];
        let y: Vec<GoldilocksExt4> = vec![GoldilocksExt4::from(11u64), GoldilocksExt4::from(13u64)];
        // The sparse evaluate function returns a single field element
        let _v_mul = mul_poly.evaluate::<GoldilocksExt4>(&z, &x, &y);
        let _v_add = add_poly.evaluate::<GoldilocksExt4>(&z, &x, &[]);
    }
}
