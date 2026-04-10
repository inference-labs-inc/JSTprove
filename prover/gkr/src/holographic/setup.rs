//! Holographic GKR setup phase.
//!
//! Produces a `(ProvingKey, VerifyingKey)` pair from a layered
//! arithmetic circuit. The verifying key carries one sparse-MLE
//! WHIR commitment per (mul, add) wiring polynomial per layer plus
//! the per-layer dimensions the verifier needs to drive the GKR
//! sumcheck without re-materializing the circuit. The proving key
//! carries the original circuit (for the existing layered evaluator)
//! plus the prover scratch the sparse-MLE WHIR open phase consumes.
//!
//! Setup is the only time the wiring polynomial commitments are
//! built, so it is the only time the cost of `sparse_commit` is
//! paid. After setup the verifying key can be distributed (it is
//! the small "vk.bin" the user originally asked for) and the
//! proving key stays with the model owner. Per-inference openings
//! against the same verifying key go through the prove / verify
//! routines layered on in Phase 2c / 2d.

use arith::{FFTField, Field, SimdField};
use circuit::Circuit;
use gkr_engine::FieldEngine;
use poly_commit::whir::{
    sparse_commit_with_combined, SparseLayout, SparseMle3, SparseMle3Commitment,
    SparseMleScratchPad,
};
use serdes::{ExpSerde, SerdeError, SerdeResult};
use tree::Tree;

use super::wiring::{extract_circuit_wiring, LayerWiring, WiringExtractError};

/// Errors raised by holographic GKR setup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetupError {
    /// Wiring extraction failed; see the inner [`WiringExtractError`].
    WiringExtraction(WiringExtractError),
    /// One of the per-layer sparse_commit calls rejected its
    /// SparseMle3 input. The layer index pinpoints the source.
    SparseCommit { layer: usize },
}

impl std::fmt::Display for SetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WiringExtraction(e) => write!(f, "wiring extraction failed: {e}"),
            Self::SparseCommit { layer } => {
                write!(f, "sparse_commit rejected layer {layer} wiring")
            }
        }
    }
}

impl std::error::Error for SetupError {}

impl From<WiringExtractError> for SetupError {
    fn from(e: WiringExtractError) -> Self {
        Self::WiringExtraction(e)
    }
}

/// Public commitment to one sparse wiring polynomial in the
/// verifying key. The Merkle root and metadata both live here so
/// the verifier can reconstruct the underlying `SparseLayout` from
/// the commitment alone.
#[derive(Debug, Clone)]
pub struct LayerWiringCommitment {
    pub commitment: SparseMle3Commitment,
    pub layout: SparseLayout,
}

impl ExpSerde for LayerWiringCommitment {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.commitment.serialize_into(&mut writer)?;
        // Layout is fully determined by (arity, nnz, n_z, n_x, n_y)
        // which the commitment already carries — but `mu` and
        // `k_pad` are derived constants and cheap to write
        // explicitly so the verifier does not have to recompute
        // them. We serialize the redundant fields for forward
        // compatibility with future layout extensions.
        let arity_tag: u8 = match self.layout.arity {
            poly_commit::whir::SparseArity::Two => 0,
            poly_commit::whir::SparseArity::Three => 1,
        };
        arity_tag.serialize_into(&mut writer)?;
        (self.layout.mu as u64).serialize_into(&mut writer)?;
        (self.layout.k_pad as u64).serialize_into(&mut writer)?;
        (self.layout.log_k_pad as u64).serialize_into(&mut writer)?;
        (self.layout.total_vars as u64).serialize_into(&mut writer)?;
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let commitment = SparseMle3Commitment::deserialize_from(&mut reader)?;
        let arity_tag = u8::deserialize_from(&mut reader)?;
        let arity = match arity_tag {
            0 => poly_commit::whir::SparseArity::Two,
            1 => poly_commit::whir::SparseArity::Three,
            _ => return Err(SerdeError::DeserializeError),
        };
        let mu = u64::deserialize_from(&mut reader)? as usize;
        let k_pad = u64::deserialize_from(&mut reader)? as usize;
        let log_k_pad = u64::deserialize_from(&mut reader)? as usize;
        let total_vars = u64::deserialize_from(&mut reader)? as usize;
        Ok(Self {
            commitment,
            layout: SparseLayout {
                arity,
                mu,
                k_pad,
                log_k_pad,
                total_vars,
            },
        })
    }
}

/// Per-layer entry in the holographic GKR verifying key.
#[derive(Debug, Clone)]
pub struct LayerVerifyingEntry {
    pub layer_index: usize,
    pub n_z: usize,
    pub n_x: usize,
    pub mul: Option<LayerWiringCommitment>,
    pub add: Option<LayerWiringCommitment>,
}

impl ExpSerde for LayerVerifyingEntry {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        (self.layer_index as u64).serialize_into(&mut writer)?;
        (self.n_z as u64).serialize_into(&mut writer)?;
        (self.n_x as u64).serialize_into(&mut writer)?;
        let has_mul: u8 = u8::from(self.mul.is_some());
        has_mul.serialize_into(&mut writer)?;
        if let Some(m) = &self.mul {
            m.serialize_into(&mut writer)?;
        }
        let has_add: u8 = u8::from(self.add.is_some());
        has_add.serialize_into(&mut writer)?;
        if let Some(a) = &self.add {
            a.serialize_into(&mut writer)?;
        }
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let layer_index = u64::deserialize_from(&mut reader)? as usize;
        let n_z = u64::deserialize_from(&mut reader)? as usize;
        let n_x = u64::deserialize_from(&mut reader)? as usize;
        let has_mul = u8::deserialize_from(&mut reader)?;
        let mul = match has_mul {
            1 => Some(LayerWiringCommitment::deserialize_from(&mut reader)?),
            0 => None,
            _ => return Err(SerdeError::DeserializeError),
        };
        let has_add = u8::deserialize_from(&mut reader)?;
        let add = match has_add {
            1 => Some(LayerWiringCommitment::deserialize_from(&mut reader)?),
            0 => None,
            _ => return Err(SerdeError::DeserializeError),
        };
        Ok(Self {
            layer_index,
            n_z,
            n_x,
            mul,
            add,
        })
    }
}

/// Public verifying key for a holographic GKR proof.
///
/// Contains the per-layer wiring commitments and the layer
/// dimensions the verifier uses to drive the per-layer sumcheck.
/// This is the artifact that ships in `vk.bin`.
#[derive(Debug, Clone)]
pub struct HolographicVerifyingKey {
    /// Wire-format version. Bumped when the layer schema changes.
    pub version: u32,
    /// Number of layers in the circuit.
    pub n_layers: usize,
    /// Per-layer dimensions and wiring commitments.
    pub layers: Vec<LayerVerifyingEntry>,
}

impl HolographicVerifyingKey {
    /// Current wire format version. Loaders refuse keys with a
    /// different version.
    pub const CURRENT_VERSION: u32 = 1;
}

impl ExpSerde for HolographicVerifyingKey {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.version.serialize_into(&mut writer)?;
        (self.n_layers as u64).serialize_into(&mut writer)?;
        for layer in &self.layers {
            layer.serialize_into(&mut writer)?;
        }
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        const MAX_LAYERS: usize = 1 << 16;
        let version = u32::deserialize_from(&mut reader)?;
        if version != Self::CURRENT_VERSION {
            return Err(SerdeError::DeserializeError);
        }
        let n_layers = u64::deserialize_from(&mut reader)? as usize;
        if n_layers > MAX_LAYERS {
            return Err(SerdeError::DeserializeError);
        }
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(LayerVerifyingEntry::deserialize_from(&mut reader)?);
        }
        Ok(Self {
            version,
            n_layers,
            layers,
        })
    }
}

/// Per-layer prover scratch retained alongside the proving key.
///
/// Holds the dense `SparseMle3` source vectors for the layer's mul
/// and add wirings, the corresponding `SparseMleScratchPad`s the
/// open phase needs, and the underlying WHIR Merkle trees +
/// codewords. None of this is in the verifying key — it is the
/// "secret" prover state that lets `sparse_open_full` produce
/// proofs without rerunning the commit phase.
pub struct LayerProvingEntry<F: Field + Send + Sync + 'static> {
    pub layer_index: usize,
    pub n_z: usize,
    pub n_x: usize,
    pub mul: Option<LayerProvingWiring<F>>,
    pub add: Option<LayerProvingWiring<F>>,
}

/// Prover-side scratch for one wiring polynomial. The
/// `combined_evals` field is the dense vector that
/// `sparse_commit_with_combined` materialized internally; it is the
/// thing `sparse_open_full` consumes when issuing the WHIR opens
/// against the per-layer wiring commitment.
pub struct LayerProvingWiring<F: Field + Send + Sync + 'static> {
    pub poly: SparseMle3<F>,
    pub layout: SparseLayout,
    pub scratch: SparseMleScratchPad<F>,
    pub tree: Tree,
    pub codeword: Vec<F>,
    pub combined_evals: Vec<F>,
}

/// Proving key for a holographic GKR proof.
pub struct HolographicProvingKey<C: FieldEngine>
where
    C::CircuitField: Send + Sync + 'static,
{
    pub circuit: Circuit<C>,
    pub layers: Vec<LayerProvingEntry<C::CircuitField>>,
}

/// Run holographic GKR setup over a layered circuit.
///
/// Walks the circuit layer by layer, extracts the sparse wiring
/// polynomials via Phase 2a, runs `sparse_commit` on each, and
/// assembles the resulting commitments into a
/// [`HolographicVerifyingKey`] alongside the prover scratch packed
/// into a [`HolographicProvingKey`]. The proving key takes
/// ownership of the input circuit so subsequent prove calls
/// against the same setup do not have to be threaded the circuit
/// separately.
///
/// # Errors
/// Returns [`SetupError::WiringExtraction`] if any layer's wiring
/// extraction fails (non-constant coefficients, unsupported gate
/// kinds, dimension overflow), or [`SetupError::SparseCommit`] if
/// the sparse-MLE commit step rejects a layer's wiring polynomial
/// (which would indicate a bug in the wiring extractor since the
/// extractor pre-validates the SparseMle3).
pub fn setup<C>(
    circuit: Circuit<C>,
) -> Result<(HolographicProvingKey<C>, HolographicVerifyingKey), SetupError>
where
    C: FieldEngine,
    C::CircuitField: FFTField + SimdField<Scalar = C::CircuitField>,
{
    let wiring = extract_circuit_wiring::<C>(&circuit)?;
    let n_layers = wiring.layers.len();

    let mut vk_layers = Vec::with_capacity(n_layers);
    let mut pk_layers = Vec::with_capacity(n_layers);

    for layer_wiring in wiring.layers {
        let LayerWiring {
            layer_index,
            n_z,
            n_x,
            mul,
            add,
            const_wiring: _const_wiring,
        } = layer_wiring;

        let mul_committed = if let Some(poly) = mul {
            Some(commit_layer_wiring::<C::CircuitField>(layer_index, poly)?)
        } else {
            None
        };
        let add_committed = if let Some(poly) = add {
            Some(commit_layer_wiring::<C::CircuitField>(layer_index, poly)?)
        } else {
            None
        };

        let vk_entry = LayerVerifyingEntry {
            layer_index,
            n_z,
            n_x,
            mul: mul_committed.as_ref().map(|w| LayerWiringCommitment {
                commitment: w.commitment.clone(),
                layout: w.layout,
            }),
            add: add_committed.as_ref().map(|w| LayerWiringCommitment {
                commitment: w.commitment.clone(),
                layout: w.layout,
            }),
        };
        vk_layers.push(vk_entry);

        let pk_entry = LayerProvingEntry {
            layer_index,
            n_z,
            n_x,
            mul: mul_committed.map(|w| w.proving),
            add: add_committed.map(|w| w.proving),
        };
        pk_layers.push(pk_entry);
    }

    let vk = HolographicVerifyingKey {
        version: HolographicVerifyingKey::CURRENT_VERSION,
        n_layers,
        layers: vk_layers,
    };
    let pk = HolographicProvingKey {
        circuit,
        layers: pk_layers,
    };
    Ok((pk, vk))
}

/// Internal struct returned by `commit_layer_wiring` carrying both
/// the public commitment + layout (for the VK) and the prover
/// scratch (for the PK).
struct CommittedLayerWiring<F: Field + Send + Sync + 'static> {
    commitment: SparseMle3Commitment,
    layout: SparseLayout,
    proving: LayerProvingWiring<F>,
}

fn commit_layer_wiring<F>(
    layer_index: usize,
    poly: SparseMle3<F>,
) -> Result<CommittedLayerWiring<F>, SetupError>
where
    F: FFTField + SimdField<Scalar = F>,
{
    // Re-derive the layout that sparse_commit will use internally.
    // Both prover and verifier compute this identically from
    // (arity, nnz, m_z, m_x, m_y) so the VK does not strictly
    // need to carry it; we expose it on LayerWiringCommitment for
    // convenience.
    let m_z = 1usize << poly.n_z;
    let m_x = 1usize << poly.n_x;
    let m_y = if poly.arity == poly_commit::whir::SparseArity::Two {
        1
    } else {
        1usize << poly.n_y
    };
    let layout = SparseLayout::compute(poly.arity, poly.nnz(), m_z, m_x, m_y);

    let (commitment, scratch, tree, codeword, combined_evals) =
        sparse_commit_with_combined::<F>(&poly)
            .map_err(|_| SetupError::SparseCommit { layer: layer_index })?;

    let proving = LayerProvingWiring {
        poly: SparseMle3 {
            // We move poly into proving but the SparseMle3 fields
            // are owned by the proving entry. Clone the metadata
            // and reuse the original vectors.
            n_z: poly.n_z,
            n_x: poly.n_x,
            n_y: poly.n_y,
            arity: poly.arity,
            row: poly.row,
            col_x: poly.col_x,
            col_y: poly.col_y,
            val: poly.val,
        },
        layout,
        scratch: SparseMleScratchPad {
            arity: scratch.arity,
            n_z: scratch.n_z,
            n_x: scratch.n_x,
            n_y: scratch.n_y,
            nnz: scratch.nnz,
            log_nnz: scratch.log_nnz,
            row: scratch.row,
            col_x: scratch.col_x,
            col_y: scratch.col_y,
            val: scratch.val,
            ts_z: scratch.ts_z,
            ts_x: scratch.ts_x,
            ts_y: scratch.ts_y,
        },
        tree,
        codeword,
        combined_evals,
    };

    Ok(CommittedLayerWiring {
        commitment,
        layout,
        proving,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use circuit::{CircuitLayer, CoefType, GateAdd, GateMul, StructureInfo};
    use gkr_engine::Goldilocksx1Config;
    type C = Goldilocksx1Config;
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

    fn build_two_layer_circuit() -> Circuit<C> {
        let mut layer0 = make_layer(2, 2);
        layer0.mul.push(mul_gate(0, 1, 2, 5));
        layer0.mul.push(mul_gate(1, 2, 3, 7));
        layer0.add.push(add_gate(2, 0, 13));
        let mut layer1 = make_layer(2, 2);
        layer1.mul.push(mul_gate(3, 0, 1, 17));
        layer1.add.push(add_gate(1, 2, 19));
        layer1.add.push(add_gate(2, 3, 23));
        Circuit {
            layers: vec![layer0, layer1],
            public_input: Vec::new(),
            expected_num_output_zeros: 0,
            rnd_coefs_identified: false,
            rnd_coefs: Vec::new(),
        }
    }

    #[test]
    fn setup_two_layer_circuit_round_trip() {
        let circuit = build_two_layer_circuit();
        let (pk, vk) = setup::<C>(circuit).expect("setup must succeed");

        // VK shape
        assert_eq!(vk.version, HolographicVerifyingKey::CURRENT_VERSION);
        assert_eq!(vk.n_layers, 2);
        assert_eq!(vk.layers.len(), 2);
        assert_eq!(vk.layers[0].layer_index, 0);
        assert_eq!(vk.layers[0].n_z, 2);
        assert_eq!(vk.layers[0].n_x, 2);
        assert!(vk.layers[0].mul.is_some());
        assert!(vk.layers[0].add.is_some());
        assert_eq!(vk.layers[1].layer_index, 1);
        assert!(vk.layers[1].mul.is_some());
        assert!(vk.layers[1].add.is_some());

        // PK shape mirrors VK
        assert_eq!(pk.layers.len(), 2);
        assert!(pk.layers[0].mul.is_some());
        assert!(pk.layers[0].add.is_some());
        assert!(pk.layers[1].mul.is_some());
        assert!(pk.layers[1].add.is_some());
    }

    #[test]
    fn setup_layer_with_only_mul_or_only_add() {
        let mut layer_mul_only = make_layer(2, 2);
        layer_mul_only.mul.push(mul_gate(0, 1, 2, 1));
        let mut layer_add_only = make_layer(2, 2);
        layer_add_only.add.push(add_gate(1, 0, 1));
        let circuit: Circuit<C> = Circuit {
            layers: vec![layer_mul_only, layer_add_only],
            public_input: Vec::new(),
            expected_num_output_zeros: 0,
            rnd_coefs_identified: false,
            rnd_coefs: Vec::new(),
        };
        let (_pk, vk) = setup::<C>(circuit).unwrap();
        assert!(vk.layers[0].mul.is_some());
        assert!(vk.layers[0].add.is_none());
        assert!(vk.layers[1].mul.is_none());
        assert!(vk.layers[1].add.is_some());
    }

    #[test]
    fn setup_propagates_wiring_extraction_errors() {
        let mut layer = make_layer(2, 2);
        let mut bad = mul_gate(0, 1, 2, 1);
        bad.coef_type = CoefType::Random;
        layer.mul.push(bad);
        let circuit: Circuit<C> = Circuit {
            layers: vec![layer],
            public_input: Vec::new(),
            expected_num_output_zeros: 0,
            rnd_coefs_identified: false,
            rnd_coefs: Vec::new(),
        };
        let result = setup::<C>(circuit);
        match result {
            Ok(_) => panic!("setup must reject random coefficients"),
            Err(SetupError::WiringExtraction(_)) => {}
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn vk_serialization_round_trip() {
        let circuit = build_two_layer_circuit();
        let (_pk, vk) = setup::<C>(circuit).unwrap();
        let mut bytes = Vec::new();
        vk.serialize_into(&mut bytes).unwrap();
        let decoded = HolographicVerifyingKey::deserialize_from(&bytes[..]).unwrap();
        assert_eq!(decoded.version, vk.version);
        assert_eq!(decoded.n_layers, vk.n_layers);
        assert_eq!(decoded.layers.len(), vk.layers.len());
        for (a, b) in decoded.layers.iter().zip(vk.layers.iter()) {
            assert_eq!(a.layer_index, b.layer_index);
            assert_eq!(a.n_z, b.n_z);
            assert_eq!(a.n_x, b.n_x);
            assert_eq!(a.mul.is_some(), b.mul.is_some());
            assert_eq!(a.add.is_some(), b.add.is_some());
        }
    }

    #[test]
    fn vk_is_small() {
        // Critical property: the VK is meant to be the lightweight
        // bin distributed to validators. Setup a 2-layer circuit
        // and assert the serialized VK is under 1 KiB.
        let circuit = build_two_layer_circuit();
        let (_pk, vk) = setup::<C>(circuit).unwrap();
        let mut bytes = Vec::new();
        vk.serialize_into(&mut bytes).unwrap();
        assert!(
            bytes.len() < 1024,
            "VK too large: {} bytes (expected < 1024)",
            bytes.len()
        );
    }

    #[test]
    fn vk_distinct_circuits_distinct_roots() {
        // Two structurally different circuits must produce VKs with
        // distinct per-layer commitment roots.
        let circuit_a = build_two_layer_circuit();
        let mut layer_b = make_layer(2, 2);
        layer_b.mul.push(mul_gate(0, 1, 2, 99)); // different coefficient
        let circuit_b: Circuit<C> = Circuit {
            layers: vec![layer_b],
            public_input: Vec::new(),
            expected_num_output_zeros: 0,
            rnd_coefs_identified: false,
            rnd_coefs: Vec::new(),
        };
        let (_, vk_a) = setup::<C>(circuit_a).unwrap();
        let (_, vk_b) = setup::<C>(circuit_b).unwrap();
        let root_a = vk_a.layers[0]
            .mul
            .as_ref()
            .unwrap()
            .commitment
            .batched_root
            .clone();
        let root_b = vk_b.layers[0]
            .mul
            .as_ref()
            .unwrap()
            .commitment
            .batched_root
            .clone();
        // Roots are Node values; comparing via byte serialization
        let mut a_bytes = Vec::new();
        let mut b_bytes = Vec::new();
        root_a.serialize_into(&mut a_bytes).unwrap();
        root_b.serialize_into(&mut b_bytes).unwrap();
        assert_ne!(a_bytes, b_bytes);
    }
}
