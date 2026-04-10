//! Holographic GKR — preprocessing variant of the GKR protocol.
//!
//! In the standard GKR protocol the verifier inherently materializes
//! the circuit topology because the per-layer sumcheck verification
//! evaluates the wiring polynomials `add_l(z,x,y)` and `mul_l(z,x,y)`
//! at the sumcheck-random point. Holographic GKR replaces those
//! verifier-side evaluations with prover-supplied openings of
//! commitments to the same wiring polynomials. The commitments live
//! in a verifying key that is fixed at setup time and contain
//! everything the verifier needs to bind the prover's openings — the
//! verifier never sees the circuit itself.
//!
//! This module is the bridge between the existing
//! `circuit::Circuit<C>` data structure and the sparse-MLE WHIR
//! commitment scheme implemented in `poly_commit::whir::sparse`.
//! Phase 2a adds the wiring-extraction layer:
//! [`wiring`] walks every layer's `mul` and `add` gate lists and
//! produces [`SparseMle3`] instances suitable for `sparse_commit`.
//!
//! Subsequent phases (2b/2c/2d in the holographic-vk task list)
//! layer the setup, prove, and verify routines on top.

pub mod challenge_extract;
pub mod combined_proof;
pub mod prove;
pub mod setup;
pub mod verify;
pub mod wiring;

pub use challenge_extract::{
    build_eval_points_from_challenges, extract_gkr_layer_challenges, ChallengeExtraction,
    PerLayerChallenge,
};
pub use combined_proof::{CombinedHolographicProof, PerLayerWiringClaims};
pub use prove::{prove, HolographicProof, LayerEvalPoint, LayerHolographicOpening, ProveError};
pub use setup::{
    setup, HolographicProvingKey, HolographicVerifyingKey, LayerProvingEntry, LayerProvingWiring,
    LayerVerifyingEntry, LayerWiringCommitment, SetupError,
};
pub use verify::{verify, VerifyError};
pub use wiring::{
    extract_circuit_wiring, extract_layer_add_wiring, extract_layer_mul_wiring, CircuitWiring,
    GateKindLabel, LayerWiring, WiringExtractError,
};
