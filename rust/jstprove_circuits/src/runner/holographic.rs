//! Runner glue for the holographic GKR setup / prove / verify
//! pipeline. Bridges the wire-format `circuit_bytes` (an
//! `expander_compiler::frontend::Circuit<C, NormalInputType>`) used
//! everywhere else in jstprove to the
//! `expander_circuit::Circuit<C::FieldConfig>` shape that
//! `gkr::holographic::setup` consumes, and serializes the resulting
//! `HolographicVerifyingKey` into a single `vk_bytes` blob suitable
//! for `jstprove_io::write_bundle_with_vk`.
//!
//! Phase 3b lands the setup half. The matching prove / verify
//! runner functions are tracked separately and require the GKR
//! sumcheck integration to land first (the per-layer eval points
//! that `gkr::holographic::prove` consumes have to come from the
//! GKR per-layer reduction, which the existing `prove_from_bytes`
//! does not yet expose).

use expander_compiler::expander_circuit;
use expander_compiler::frontend::{CircuitField, Config};
use expander_compiler::serdes::ExpSerde;
use gkr_engine::{FieldEngine, GKREngine};

use crate::runner::errors::RunError;
use crate::runner::main_runner::load_circuit_from_bytes;

/// Run holographic GKR setup over a circuit deserialized from
/// `circuit_bytes` and return the serialized
/// `gkr::holographic::HolographicVerifyingKey` as `vk_bytes`.
///
/// The output bytes are suitable for
/// `jstprove_io::bundle::write_bundle_with_vk` and for
/// `jstprove_io::bundle::read_vk_only` on the validator side.
///
/// Field constraint: `C::FieldConfig::CircuitField` must implement
/// `FFTField + SimdField<Scalar = Self>`. This is satisfied by all
/// the Goldilocks-family configs in production
/// (`Goldilocksx1ConfigSha2*`, `GoldilocksExt3x1ConfigSha2Whir`,
/// `GoldilocksExt4x1ConfigSha2Whir`).
///
/// # Errors
/// Returns [`RunError::Deserialize`] if the circuit bytes are
/// malformed, [`RunError::Compile`] if wiring extraction or sparse
/// commit rejects any layer, or [`RunError::Serialize`] if the VK
/// serialization fails.
pub fn holographic_setup_from_bytes<C>(circuit_bytes: &[u8]) -> Result<Vec<u8>, RunError>
where
    C: Config + GKREngine,
    <<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField:
        arith::FFTField + arith::SimdField<Scalar = CircuitField<C>>,
{
    let layered = load_circuit_from_bytes::<C>(circuit_bytes)?;
    let expander: expander_circuit::Circuit<<C as GKREngine>::FieldConfig> =
        layered.export_to_expander_flatten();

    let (_pk, vk) = gkr::holographic::setup::<<C as GKREngine>::FieldConfig>(expander)
        .map_err(|e| RunError::Compile(format!("holographic setup: {e}")))?;

    let mut bytes = Vec::new();
    vk.serialize_into(&mut bytes)
        .map_err(|e| RunError::Serialize(format!("holographic vk: {e:?}")))?;
    Ok(bytes)
}
