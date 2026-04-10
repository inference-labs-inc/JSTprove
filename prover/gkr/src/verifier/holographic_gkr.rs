//! Circuit-free GKR verification driven entirely by a
//! `HolographicVerifyingKey` and per-layer wiring evaluations
//! from the proof. This is the top-level function validators
//! call — it never touches a `Circuit`.
//!
//! The function mirrors `gkr_verify` in `gkr_vanilla.rs` but at
//! every point where the standard verifier would compute
//! `eval_cst / eval_add / eval_mul` from the circuit's gate
//! lists, it instead reads the values from a per-layer
//! `WiringEvals` slice the caller has already verified against
//! the VK commitments via `sparse_verify_full`.

use std::io::Read;

use gkr_engine::{ExpanderDualVarChallenge, ExpanderSingleVarChallenge, FieldEngine, Transcript};
use sumcheck::VerifierScratchPad;
use transcript::transcript_verifier_sync;

use super::holographic_common::{sumcheck_verify_gkr_layer_holographic, LayerShape, WiringEvals};

/// Per-layer shape + wiring evaluations the holographic verifier
/// consumes. The caller (the holographic verify module in
/// `gkr::holographic`) builds this from the VK dimensions + the
/// verified sparse-MLE openings for each layer.
pub struct HolographicLayerInput<F: FieldEngine> {
    pub shape: LayerShape,
    pub wiring: WiringEvals<F>,
}

/// Run the circuit-free GKR verifier.
///
/// `layers` must be in the same order the standard GKR prover
/// produced the proof: last layer first (reverse of the circuit's
/// layer vector, i.e. output → input). Each entry carries the
/// per-layer dimensions (from the VK) and the wiring evaluations
/// (from verified holographic openings).
///
/// Returns `(verified, challenge, claimed_v0, claimed_v1)` with
/// the same semantics as `gkr_verify`. On success the caller
/// proceeds to verify the PCS opening at `challenge`.
#[allow(clippy::type_complexity)]
pub fn holographic_gkr_verify<F: FieldEngine>(
    proving_time_mpi_size: usize,
    layers: &[HolographicLayerInput<F>],
    output_var_num: usize,
    claimed_v: &F::ChallengeField,
    transcript: &mut impl Transcript,
    mut proof_reader: impl Read,
) -> (
    bool,
    ExpanderDualVarChallenge<F>,
    F::ChallengeField,
    Option<F::ChallengeField>,
) {
    let mut challenge: ExpanderDualVarChallenge<F> =
        ExpanderSingleVarChallenge::sample_from_transcript(
            transcript,
            output_var_num,
            proving_time_mpi_size,
        )
        .into();

    let mut alpha = None;
    let mut claimed_v0 = *claimed_v;
    let mut claimed_v1 = None;

    transcript.lock_proof();
    transcript.append_field_element(claimed_v);
    transcript.unlock_proof();

    // Build a scratch pad from the maximum layer dimensions.
    // The standard verifier builds it from the circuit; we build
    // it from the VK-supplied dimensions.
    let max_input_var = layers
        .iter()
        .map(|l| l.shape.input_var_num)
        .max()
        .unwrap_or(0);
    let max_output_var = layers
        .iter()
        .map(|l| l.shape.output_var_num)
        .max()
        .unwrap_or(0);
    let max_var = max_input_var.max(max_output_var);
    let mut sp = VerifierScratchPad::<F>::new_with_num_vars(max_var, proving_time_mpi_size);

    let mut verified = true;
    for (i, layer_input) in layers.iter().enumerate() {
        let cur_verified = sumcheck_verify_gkr_layer_holographic::<F>(
            proving_time_mpi_size,
            &layer_input.shape,
            &layer_input.wiring,
            &mut challenge,
            &mut claimed_v0,
            &mut claimed_v1,
            alpha,
            &mut proof_reader,
            transcript,
            &mut sp,
            i == 0, // first entry = output layer
        );

        verified &= cur_verified;
        alpha = if challenge.rz_1.is_some() {
            Some(transcript.generate_field_element::<F::ChallengeField>())
        } else {
            None
        };
    }

    transcript_verifier_sync(transcript, proving_time_mpi_size);

    let challenge = ExpanderDualVarChallenge::new(
        challenge.rz_0,
        challenge.rz_1,
        challenge.r_simd,
        challenge.r_mpi,
    );

    (verified, challenge, claimed_v0, claimed_v1)
}
