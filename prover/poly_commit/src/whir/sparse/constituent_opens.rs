//! Prover-side WHIR opens of sparse-MLE constituent polynomials.
//!
//! Phase 1d-4d-ii: extends Phase 1d-4c's `sparse_open_skeleton` with
//! the WHIR opens of every base-field constituent the verifier will
//! need to discharge the eval-claim final factor evaluations and the
//! per-axis multiset leaf claims.
//!
//! What gets opened. For an honest opening at outer eval point
//! `(z, x, y)`:
//!
//! * **Setup commitment** (in VK):
//!     - `val` at the eval-claim sumcheck random point `r_sc`
//!     - For each axis (z, x, [y]):
//!         - `row_axis` at `r_RS_axis` (RS leaf claim point)
//!         - `read_ts_axis` at `r_RS_axis`
//!         - `row_axis` at `r_WS_axis` (WS leaf claim point)
//!         - `read_ts_axis` at `r_WS_axis`
//!         - `audit_ts_axis` at `r_Audit_axis`
//!
//! * **Per-evaluation commitment** (in proof):
//!     - `e_axis` limbs at the sumcheck random point `r_sc` (one
//!       open per limb, `E::DEGREE` opens per axis)
//!     - For each axis: `e_axis` limbs at `r_RS_axis` and `r_WS_axis`
//!
//! Init leaf claims need no opens because the verifier computes them
//! independently from the cell-index identity polynomial and the
//! `ẽq(c, r_axis)` closed form (Spartan §7.2.3 opt 4).
//!
//! Random-point extraction. The prove-side sub-phase functions
//! (`prove_eval_sumcheck`, `prove_product_circuit`) consume their
//! sumcheck challenges from the shared transcript but do not return
//! them. To recover the challenges without refactoring those APIs,
//! we exploit the fact that `BytesHashTranscript` is `Clone`: at
//! each sub-phase boundary we save a clone of the transcript, then
//! after the prover advances the main transcript, replay the
//! verifier on the clone to extract the challenges. The verifier
//! consumes the same bytes the prover wrote in the same order, so
//! the clone ends up in the same state as the main transcript and
//! the challenges it derives are exactly the ones the prover used.

use arith::{ExtensionField, FFTField, Field, SimdField};
use gkr_engine::Transcript;
use serdes::{ExpSerde, SerdeError, SerdeResult};

use super::combined_point::combined_eval_point;
use super::commit::{SparseConstituentSlot, SparseLayout};
use super::eval_commit::{eval_combined_point, EvalConstituent, SparseEvalSlot};
use super::eval_sumcheck::verify_eval_sumcheck;
use super::full_open::{sparse_open_skeleton, SparseFullOpening, SparseFullOpeningScratch};
use super::multiset_open::PerAxisMultisetProof;
use super::product_argument::verify_product_circuit;
use super::types::{SparseArity, SparseMleScratchPad};
use super::whir_glue::{whir_open_at_combined_point, WhirOpenWithClaim};
use crate::whir::types::WhirCommitment;

/// Per-axis WHIR opens of the sparse-MLE constituents at the
/// per-axis multiset random points produced by Phase 1c's product-
/// circuit GKR.
#[derive(Debug, Clone, Default)]
pub struct PerAxisConstituentOpens<E: ExtensionField> {
    /// `row_axis(r_RS_axis)` — setup-commit open
    pub row_at_rs: WhirOpenWithClaim<E>,
    /// `read_ts_axis(r_RS_axis)` — setup-commit open
    pub read_ts_at_rs: WhirOpenWithClaim<E>,
    /// `e_axis(r_RS_axis)` per limb — eval-commit opens, length
    /// `E::DEGREE`
    pub e_at_rs: Vec<WhirOpenWithClaim<E>>,

    /// `row_axis(r_WS_axis)` — setup-commit open
    pub row_at_ws: WhirOpenWithClaim<E>,
    /// `read_ts_axis(r_WS_axis)` — setup-commit open
    pub read_ts_at_ws: WhirOpenWithClaim<E>,
    /// `e_axis(r_WS_axis)` per limb — eval-commit opens
    pub e_at_ws: Vec<WhirOpenWithClaim<E>>,

    /// `audit_ts_axis(r_Audit_axis)` — setup-commit open
    pub audit_ts_at_audit: WhirOpenWithClaim<E>,
}

impl<E: ExtensionField> ExpSerde for PerAxisConstituentOpens<E> {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.row_at_rs.serialize_into(&mut writer)?;
        self.read_ts_at_rs.serialize_into(&mut writer)?;
        serialize_vec(&self.e_at_rs, &mut writer)?;
        self.row_at_ws.serialize_into(&mut writer)?;
        self.read_ts_at_ws.serialize_into(&mut writer)?;
        serialize_vec(&self.e_at_ws, &mut writer)?;
        self.audit_ts_at_audit.serialize_into(&mut writer)?;
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let row_at_rs = WhirOpenWithClaim::<E>::deserialize_from(&mut reader)?;
        let read_ts_at_rs = WhirOpenWithClaim::<E>::deserialize_from(&mut reader)?;
        let e_at_rs = deserialize_vec(&mut reader)?;
        let row_at_ws = WhirOpenWithClaim::<E>::deserialize_from(&mut reader)?;
        let read_ts_at_ws = WhirOpenWithClaim::<E>::deserialize_from(&mut reader)?;
        let e_at_ws = deserialize_vec(&mut reader)?;
        let audit_ts_at_audit = WhirOpenWithClaim::<E>::deserialize_from(&mut reader)?;
        Ok(Self {
            row_at_rs,
            read_ts_at_rs,
            e_at_rs,
            row_at_ws,
            read_ts_at_ws,
            e_at_ws,
            audit_ts_at_audit,
        })
    }
}

/// Aggregate of all sparse-MLE constituent opens — both the eval-
/// claim final factor evaluations and the per-axis multiset leaf
/// claims.
#[derive(Debug, Clone, Default)]
pub struct SparseConstituentOpens<E: ExtensionField> {
    /// `val(r_sc)` — single setup-commit open at the eval-claim
    /// sumcheck random point.
    pub val_at_sumcheck: WhirOpenWithClaim<E>,
    /// `e_z(r_sc)` per limb — eval-commit opens at the sumcheck
    /// random point.
    pub ez_limbs_at_sumcheck: Vec<WhirOpenWithClaim<E>>,
    /// `e_x(r_sc)` per limb.
    pub ex_limbs_at_sumcheck: Vec<WhirOpenWithClaim<E>>,
    /// `e_y(r_sc)` per limb — `Some` only for arity Three.
    pub ey_limbs_at_sumcheck: Option<Vec<WhirOpenWithClaim<E>>>,

    pub axis_z_opens: PerAxisConstituentOpens<E>,
    pub axis_x_opens: PerAxisConstituentOpens<E>,
    pub axis_y_opens: Option<PerAxisConstituentOpens<E>>,
}

impl<E: ExtensionField> ExpSerde for SparseConstituentOpens<E> {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.val_at_sumcheck.serialize_into(&mut writer)?;
        serialize_vec(&self.ez_limbs_at_sumcheck, &mut writer)?;
        serialize_vec(&self.ex_limbs_at_sumcheck, &mut writer)?;
        let has_ey: u8 = if self.ey_limbs_at_sumcheck.is_some() {
            1
        } else {
            0
        };
        has_ey.serialize_into(&mut writer)?;
        if let Some(ey) = &self.ey_limbs_at_sumcheck {
            serialize_vec(ey, &mut writer)?;
        }
        self.axis_z_opens.serialize_into(&mut writer)?;
        self.axis_x_opens.serialize_into(&mut writer)?;
        let has_y_axis: u8 = if self.axis_y_opens.is_some() { 1 } else { 0 };
        has_y_axis.serialize_into(&mut writer)?;
        if let Some(axis_y) = &self.axis_y_opens {
            axis_y.serialize_into(&mut writer)?;
        }
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let val_at_sumcheck = WhirOpenWithClaim::<E>::deserialize_from(&mut reader)?;
        let ez_limbs_at_sumcheck = deserialize_vec(&mut reader)?;
        let ex_limbs_at_sumcheck = deserialize_vec(&mut reader)?;
        let has_ey = u8::deserialize_from(&mut reader)?;
        let ey_limbs_at_sumcheck = match has_ey {
            0 => None,
            1 => Some(deserialize_vec(&mut reader)?),
            _ => return Err(SerdeError::DeserializeError),
        };
        let axis_z_opens = PerAxisConstituentOpens::<E>::deserialize_from(&mut reader)?;
        let axis_x_opens = PerAxisConstituentOpens::<E>::deserialize_from(&mut reader)?;
        let has_y_axis = u8::deserialize_from(&mut reader)?;
        let axis_y_opens = match has_y_axis {
            0 => None,
            1 => Some(PerAxisConstituentOpens::<E>::deserialize_from(&mut reader)?),
            _ => return Err(SerdeError::DeserializeError),
        };
        Ok(Self {
            val_at_sumcheck,
            ez_limbs_at_sumcheck,
            ex_limbs_at_sumcheck,
            ey_limbs_at_sumcheck,
            axis_z_opens,
            axis_x_opens,
            axis_y_opens,
        })
    }
}

fn serialize_vec<E: ExtensionField, W: std::io::Write>(
    items: &[WhirOpenWithClaim<E>],
    mut writer: W,
) -> SerdeResult<()> {
    (items.len() as u64).serialize_into(&mut writer)?;
    for item in items {
        item.serialize_into(&mut writer)?;
    }
    Ok(())
}

fn deserialize_vec<E: ExtensionField, R: std::io::Read>(
    mut reader: R,
) -> SerdeResult<Vec<WhirOpenWithClaim<E>>> {
    const MAX_LIMBS: usize = 64;
    let n = u64::deserialize_from(&mut reader)? as usize;
    if n > MAX_LIMBS {
        return Err(SerdeError::DeserializeError);
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(WhirOpenWithClaim::<E>::deserialize_from(&mut reader)?);
    }
    Ok(out)
}

/// Output of [`sparse_open_full`]: the full sparse-MLE opening
/// (skeleton + WHIR opens of all constituents).
#[derive(Debug, Clone, Default)]
pub struct SparseMle3FullOpening<E: ExtensionField> {
    pub skeleton: SparseFullOpening<E>,
    pub constituent_opens: SparseConstituentOpens<E>,
}

impl<E: ExtensionField> ExpSerde for SparseMle3FullOpening<E> {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.skeleton.serialize_into(&mut writer)?;
        self.constituent_opens.serialize_into(&mut writer)?;
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let skeleton = SparseFullOpening::<E>::deserialize_from(&mut reader)?;
        let constituent_opens = SparseConstituentOpens::<E>::deserialize_from(&mut reader)?;
        Ok(Self {
            skeleton,
            constituent_opens,
        })
    }
}

/// Run the prover side of the full sparse-MLE WHIR open phase.
///
/// Wraps [`sparse_open_skeleton`] and additionally issues the WHIR
/// opens for every constituent the verifier needs. The result is a
/// [`SparseMle3FullOpening`] suitable for transmission to the
/// verifier; the prover scratch returned alongside is the same one
/// produced by `sparse_open_skeleton` and is no longer needed once
/// this function returns (the WHIR opens have already been
/// computed).
///
/// Random-point extraction. After running each prover sub-phase the
/// function clones the transcript at the sub-phase boundary and
/// replays the matching verifier on the clone to extract the
/// challenge sequence the prover consumed. The clone ends up in the
/// same state as the main transcript so the next sub-phase boundary
/// can fork from it again. This is sound because Fiat-Shamir
/// transcripts are deterministic functions of the bytes appended.
///
/// # Panics
/// Panics on the same input invariants as [`sparse_open_skeleton`].
/// Additionally panics if any of the prover-built sub-proofs does
/// not pass its own verifier on the clone (which would indicate a
/// programming bug in the prover or sumcheck implementation, not a
/// protocol-level rejection).
#[allow(clippy::too_many_arguments)]
pub fn sparse_open_full<F, E, T>(
    setup_commitment: &WhirCommitment,
    setup_evals: &[F],
    setup_codeword: &[F],
    setup_tree: &tree::Tree,
    setup_layout: &SparseLayout,
    scratch: &SparseMleScratchPad<F>,
    claimed_eval: E,
    z: &[E],
    x: &[E],
    y: &[E],
    transcript: &mut T,
) -> (SparseMle3FullOpening<E>, SparseFullOpeningScratch<F, E>)
where
    F: FFTField + SimdField<Scalar = F>,
    E: ExtensionField<BaseField = F>
        + FFTField
        + SimdField<Scalar = E>
        + std::ops::Add<F, Output = E>
        + std::ops::Mul<F, Output = E>,
    T: Transcript + Clone,
{
    // The setup commitment is just a witness used by tests below;
    // it is not consumed by the open path. We thread it through so
    // callers can pass it without having to keep both in lock-step.
    let _ = setup_commitment;

    // ----- Phase 1d-4c skeleton: eval-claim sumcheck + eval commit
    // ----- + per-axis multiset arguments. -------------------------

    // Save the transcript state at the START so we can replay every
    // sub-phase verifier from a known fork point. The skeleton
    // appends:
    //   1. eval-claim sumcheck rounds and final factor evals
    //   2. per-eval commitment Merkle root
    //   3. (γ_1, γ_2) samples + four product-circuit transcripts
    //      per axis
    let pre_skeleton = transcript.clone();

    let (opening, open_scratch) =
        sparse_open_skeleton::<F, E>(scratch, claimed_eval, z, x, y, transcript);

    // ----- Random-point extraction via verifier replay. -----------

    let mut replay = pre_skeleton;

    // 1. Extract eval-claim sumcheck challenges
    let eval_claim_extract = verify_eval_sumcheck::<E>(
        scratch.arity,
        scratch.log_nnz,
        opening.evalclaim.claimed_eval,
        &opening.evalclaim.eval_sumcheck,
        &mut replay,
    )
    .expect("prover-built eval-claim sumcheck must verify on the replay clone");
    let r_sc = eval_claim_extract.challenges;

    // 2. Append the per-eval commitment root (mirror what
    // sparse_open_skeleton did)
    replay.append_u8_slice(opening.eval_commitment.root.as_bytes());

    // 3. Sample (γ_1, γ_2) — discarded; we just need the side
    // effect of the transcript advancing in lock-step with the
    // prover.
    let _gamma_1: E = replay.generate_field_element();
    let _gamma_2: E = replay.generate_field_element();

    // 4. Extract per-axis product-circuit random points
    let m_z = 1usize << scratch.n_z;
    let m_x = 1usize << scratch.n_x;
    let m_y = 1usize << scratch.n_y;
    let log_m_z = scratch.n_z;
    let log_m_x = scratch.n_x;
    let log_m_y = scratch.n_y;
    let log_nnz = scratch.log_nnz;

    let axis_z_random =
        extract_axis_random_points::<E>(&opening.multiset.axis_z, log_m_z, log_nnz, &mut replay);
    let axis_x_random =
        extract_axis_random_points::<E>(&opening.multiset.axis_x, log_m_x, log_nnz, &mut replay);
    let axis_y_random = if scratch.arity == SparseArity::Three {
        let axis_y = opening
            .multiset
            .axis_y
            .as_ref()
            .expect("multiset.axis_y present for arity Three");
        Some(extract_axis_random_points::<E>(
            axis_y,
            log_m_y,
            log_nnz,
            &mut replay,
        ))
    } else {
        None
    };

    let _ = (m_z, m_x, m_y); // referenced for documentation; not used numerically here

    // ----- WHIR opens of every constituent the verifier will need.

    // 1. Eval-claim final factor evaluations
    let val_at_sumcheck = open_setup_constituent::<F, E, T>(
        setup_evals,
        setup_codeword,
        setup_tree,
        setup_layout,
        SparseConstituentSlot::Val,
        &r_sc,
        transcript,
    );

    let ez_limbs_at_sumcheck = open_eval_constituent_limbs::<F, E, T>(
        &open_scratch,
        EvalConstituent::Ez,
        &r_sc,
        transcript,
    );
    let ex_limbs_at_sumcheck = open_eval_constituent_limbs::<F, E, T>(
        &open_scratch,
        EvalConstituent::Ex,
        &r_sc,
        transcript,
    );
    let ey_limbs_at_sumcheck = if scratch.arity == SparseArity::Three {
        Some(open_eval_constituent_limbs::<F, E, T>(
            &open_scratch,
            EvalConstituent::Ey,
            &r_sc,
            transcript,
        ))
    } else {
        None
    };

    // 2. Per-axis multiset opens
    let axis_z_opens = open_per_axis_constituents::<F, E, T>(
        setup_evals,
        setup_codeword,
        setup_tree,
        setup_layout,
        &open_scratch,
        AxisOpenSelectors {
            row_slot: SparseConstituentSlot::Row,
            read_ts_slot: SparseConstituentSlot::ReadTsZ,
            audit_ts_slot: SparseConstituentSlot::AuditTsZ,
            eval_constituent: EvalConstituent::Ez,
        },
        &axis_z_random,
        transcript,
    );
    let axis_x_opens = open_per_axis_constituents::<F, E, T>(
        setup_evals,
        setup_codeword,
        setup_tree,
        setup_layout,
        &open_scratch,
        AxisOpenSelectors {
            row_slot: SparseConstituentSlot::ColX,
            read_ts_slot: SparseConstituentSlot::ReadTsX,
            audit_ts_slot: SparseConstituentSlot::AuditTsX,
            eval_constituent: EvalConstituent::Ex,
        },
        &axis_x_random,
        transcript,
    );
    let axis_y_opens = if let Some(axis_y_random) = &axis_y_random {
        Some(open_per_axis_constituents::<F, E, T>(
            setup_evals,
            setup_codeword,
            setup_tree,
            setup_layout,
            &open_scratch,
            AxisOpenSelectors {
                row_slot: SparseConstituentSlot::ColY,
                read_ts_slot: SparseConstituentSlot::ReadTsY,
                audit_ts_slot: SparseConstituentSlot::AuditTsY,
                eval_constituent: EvalConstituent::Ey,
            },
            axis_y_random,
            transcript,
        ))
    } else {
        None
    };

    let constituent_opens = SparseConstituentOpens {
        val_at_sumcheck,
        ez_limbs_at_sumcheck,
        ex_limbs_at_sumcheck,
        ey_limbs_at_sumcheck,
        axis_z_opens,
        axis_x_opens,
        axis_y_opens,
    };

    let full_opening = SparseMle3FullOpening {
        skeleton: opening,
        constituent_opens,
    };

    (full_opening, open_scratch)
}

/// Random points extracted by replaying one axis's four product-
/// circuit verifiers.
struct PerAxisRandomPoints<E: Field> {
    /// `r_init`: leaf point of the Init product-circuit GKR. Not
    /// strictly needed for any WHIR open (the verifier computes
    /// the Init leaves locally), but extracting it is required to
    /// advance the transcript in lock-step with the prover.
    pub r_init: Vec<E>,
    pub r_rs: Vec<E>,
    pub r_ws: Vec<E>,
    pub r_audit: Vec<E>,
}

fn extract_axis_random_points<E>(
    proof: &PerAxisMultisetProof<E>,
    log_m_axis: usize,
    log_nnz: usize,
    transcript: &mut impl Transcript,
) -> PerAxisRandomPoints<E>
where
    E: ExtensionField,
{
    let init = verify_product_circuit::<E>(log_m_axis, proof.h_init, &proof.init_proof, transcript)
        .expect("prover-built Init product circuit must verify on replay");
    let rs = verify_product_circuit::<E>(log_nnz, proof.h_rs, &proof.rs_proof, transcript)
        .expect("prover-built RS product circuit must verify on replay");
    let ws = verify_product_circuit::<E>(log_nnz, proof.h_ws, &proof.ws_proof, transcript)
        .expect("prover-built WS product circuit must verify on replay");
    let audit =
        verify_product_circuit::<E>(log_m_axis, proof.h_audit, &proof.audit_proof, transcript)
            .expect("prover-built Audit product circuit must verify on replay");
    PerAxisRandomPoints {
        r_init: init.leaf_point,
        r_rs: rs.leaf_point,
        r_ws: ws.leaf_point,
        r_audit: audit.leaf_point,
    }
}

/// Open a single setup-commit constituent at a sub-point.
///
/// Constructs the combined eval point via [`combined_eval_point`],
/// padding the sub-point with `E::ZERO` to the layout's `μ` if
/// needed (the constituent's natural length is bounded by `2^μ` and
/// the populated entries occupy the low part of the slot).
fn open_setup_constituent<F, E, T>(
    setup_evals: &[F],
    setup_codeword: &[F],
    setup_tree: &tree::Tree,
    setup_layout: &SparseLayout,
    constituent: SparseConstituentSlot,
    sub_point: &[E],
    transcript: &mut T,
) -> WhirOpenWithClaim<E>
where
    F: FFTField + SimdField<Scalar = F>,
    E: ExtensionField<BaseField = F>
        + FFTField
        + SimdField<Scalar = E>
        + std::ops::Add<F, Output = E>
        + std::ops::Mul<F, Output = E>,
    T: Transcript,
{
    let padded_sub: Vec<E> = pad_to(sub_point, setup_layout.mu);
    let combined_point = combined_eval_point(setup_layout, constituent, &padded_sub);
    whir_open_at_combined_point::<F, E>(
        setup_evals,
        setup_codeword,
        setup_tree,
        setup_layout.total_vars,
        &combined_point,
        transcript,
    )
}

/// Open every limb of one extension-field constituent in the per-
/// evaluation commitment at a sub-point. Returns one
/// [`WhirOpenWithClaim`] per limb (length `E::DEGREE`).
fn open_eval_constituent_limbs<F, E, T>(
    open_scratch: &SparseFullOpeningScratch<F, E>,
    constituent: EvalConstituent,
    sub_point: &[E],
    transcript: &mut T,
) -> Vec<WhirOpenWithClaim<E>>
where
    F: FFTField + SimdField<Scalar = F>,
    E: ExtensionField<BaseField = F>
        + FFTField
        + SimdField<Scalar = E>
        + std::ops::Add<F, Output = E>
        + std::ops::Mul<F, Output = E>,
    T: Transcript,
{
    let layout = open_scratch.eval_scratch.layout;
    debug_assert_eq!(sub_point.len(), layout.mu_eval);
    let mut opens = Vec::with_capacity(layout.degree);
    for limb in 0..layout.degree {
        let slot = SparseEvalSlot { constituent, limb };
        let combined_point = eval_combined_point(&layout, slot, sub_point);
        let open = whir_open_at_combined_point::<F, E>(
            &open_scratch.eval_scratch.combined,
            &open_scratch.eval_scratch.codeword,
            &open_scratch.eval_scratch.tree,
            layout.total_vars,
            &combined_point,
            transcript,
        );
        opens.push(open);
    }
    opens
}

/// Selectors that pin down which setup constituent and which eval
/// constituent correspond to a given address axis. The naming is
/// `(row, read_ts, audit_ts, e_axis)` where `row` is the axis's
/// address vector inside the sparse polynomial.
struct AxisOpenSelectors {
    row_slot: SparseConstituentSlot,
    read_ts_slot: SparseConstituentSlot,
    audit_ts_slot: SparseConstituentSlot,
    eval_constituent: EvalConstituent,
}

#[allow(clippy::too_many_arguments)]
fn open_per_axis_constituents<F, E, T>(
    setup_evals: &[F],
    setup_codeword: &[F],
    setup_tree: &tree::Tree,
    setup_layout: &SparseLayout,
    open_scratch: &SparseFullOpeningScratch<F, E>,
    selectors: AxisOpenSelectors,
    random: &PerAxisRandomPoints<E>,
    transcript: &mut T,
) -> PerAxisConstituentOpens<E>
where
    F: FFTField + SimdField<Scalar = F>,
    E: ExtensionField<BaseField = F>
        + FFTField
        + SimdField<Scalar = E>
        + std::ops::Add<F, Output = E>
        + std::ops::Mul<F, Output = E>,
    T: Transcript,
{
    let _ = &random.r_init; // Init's leaf point is computed by the verifier; no open needed

    // RS leaf claim opens
    let row_at_rs = open_setup_constituent::<F, E, T>(
        setup_evals,
        setup_codeword,
        setup_tree,
        setup_layout,
        selectors.row_slot,
        &random.r_rs,
        transcript,
    );
    let read_ts_at_rs = open_setup_constituent::<F, E, T>(
        setup_evals,
        setup_codeword,
        setup_tree,
        setup_layout,
        selectors.read_ts_slot,
        &random.r_rs,
        transcript,
    );
    let e_at_rs = open_eval_constituent_limbs::<F, E, T>(
        open_scratch,
        selectors.eval_constituent,
        &random.r_rs,
        transcript,
    );

    // WS leaf claim opens
    let row_at_ws = open_setup_constituent::<F, E, T>(
        setup_evals,
        setup_codeword,
        setup_tree,
        setup_layout,
        selectors.row_slot,
        &random.r_ws,
        transcript,
    );
    let read_ts_at_ws = open_setup_constituent::<F, E, T>(
        setup_evals,
        setup_codeword,
        setup_tree,
        setup_layout,
        selectors.read_ts_slot,
        &random.r_ws,
        transcript,
    );
    let e_at_ws = open_eval_constituent_limbs::<F, E, T>(
        open_scratch,
        selectors.eval_constituent,
        &random.r_ws,
        transcript,
    );

    // Audit leaf claim open (audit_ts only — Init/Audit cell-index
    // and mem evaluations are computed locally by the verifier)
    let audit_ts_at_audit = open_setup_constituent::<F, E, T>(
        setup_evals,
        setup_codeword,
        setup_tree,
        setup_layout,
        selectors.audit_ts_slot,
        &random.r_audit,
        transcript,
    );

    PerAxisConstituentOpens {
        row_at_rs,
        read_ts_at_rs,
        e_at_rs,
        row_at_ws,
        read_ts_at_ws,
        e_at_ws,
        audit_ts_at_audit,
    }
}

/// Append `E::ZERO` entries to a sub-point until it reaches the
/// target length. Used to embed an axis-specific random point
/// (length `log_m_axis` or `log_nnz`) into the setup commitment's
/// `μ`-variable space.
fn pad_to<E: Field>(point: &[E], target: usize) -> Vec<E> {
    debug_assert!(point.len() <= target, "pad_to: point longer than target");
    let mut out = Vec::with_capacity(target);
    out.extend_from_slice(point);
    while out.len() < target {
        out.push(E::ZERO);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whir::sparse::commit::sparse_commit;
    use crate::whir::sparse::types::SparseMle3;
    use arith::Field;
    use goldilocks::{Goldilocks, GoldilocksExt4};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use serdes::ExpSerde;
    use transcript::BytesHashTranscript;
    type Sha2T = BytesHashTranscript<gkr_hashers::SHA256hasher>;

    fn rng_for(label: &str) -> ChaCha20Rng {
        let mut seed = [0u8; 32];
        let bytes = label.as_bytes();
        let n = bytes.len().min(32);
        seed[..n].copy_from_slice(&bytes[..n]);
        ChaCha20Rng::from_seed(seed)
    }

    fn build_two_axis(
        rng: &mut ChaCha20Rng,
        n_z: usize,
        n_x: usize,
        nnz: usize,
    ) -> SparseMle3<Goldilocks> {
        let m_z = 1usize << n_z;
        let m_x = 1usize << n_x;
        let mut row = Vec::with_capacity(nnz);
        let mut col_x = Vec::with_capacity(nnz);
        let mut val = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            row.push(rng.gen_range(0..m_z));
            col_x.push(rng.gen_range(0..m_x));
            val.push(Goldilocks::random_unsafe(&mut *rng));
        }
        SparseMle3 {
            n_z,
            n_x,
            n_y: 0,
            arity: SparseArity::Two,
            row,
            col_x,
            col_y: vec![0; nnz],
            val,
        }
    }

    fn build_three_axis(
        rng: &mut ChaCha20Rng,
        n_z: usize,
        n_x: usize,
        n_y: usize,
        nnz: usize,
    ) -> SparseMle3<Goldilocks> {
        let m_z = 1usize << n_z;
        let m_x = 1usize << n_x;
        let m_y = 1usize << n_y;
        let mut row = Vec::with_capacity(nnz);
        let mut col_x = Vec::with_capacity(nnz);
        let mut col_y = Vec::with_capacity(nnz);
        let mut val = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            row.push(rng.gen_range(0..m_z));
            col_x.push(rng.gen_range(0..m_x));
            col_y.push(rng.gen_range(0..m_y));
            val.push(Goldilocks::random_unsafe(&mut *rng));
        }
        SparseMle3 {
            n_z,
            n_x,
            n_y,
            arity: SparseArity::Three,
            row,
            col_x,
            col_y,
            val,
        }
    }

    #[test]
    fn pad_to_pads_with_zero() {
        let p: Vec<GoldilocksExt4> = vec![GoldilocksExt4::from(7u64)];
        let padded = pad_to(&p, 4);
        assert_eq!(padded.len(), 4);
        assert_eq!(padded[0], GoldilocksExt4::from(7u64));
        for i in 1..4 {
            assert_eq!(padded[i], GoldilocksExt4::ZERO);
        }
    }

    #[test]
    fn open_full_two_axis_produces_all_constituent_opens() {
        let mut rng = rng_for("open_full_two_axis");
        let poly = build_two_axis(&mut rng, 3, 3, 8);
        let (commitment, scratch, tree, codeword) = sparse_commit::<Goldilocks>(&poly).unwrap();

        // Re-derive the layout from the commitment metadata
        let layout = SparseLayout::compute(
            poly.arity,
            scratch.nnz,
            1usize << scratch.n_z,
            1usize << scratch.n_x,
            1,
        );
        // Build the combined evals from the same recipe sparse_commit uses;
        // the test doesn't have access to it directly, but we can read it
        // from the codeword by looking at the original combined vector. For
        // simplicity, re-derive it via sparse_commit and discard the second
        // tree/codeword (we trust sparse_commit to be deterministic).
        let combined_evals: Vec<Goldilocks> = {
            let mut combined = vec![Goldilocks::ZERO; layout.combined_len()];
            let val_offset = layout.slot_offset(SparseConstituentSlot::Val);
            combined[val_offset..val_offset + scratch.val.len()].copy_from_slice(&scratch.val);
            let row_offset = layout.slot_offset(SparseConstituentSlot::Row);
            for (k, &a) in scratch.row.iter().enumerate() {
                combined[row_offset + k] = Goldilocks::from(a as u32);
            }
            let col_x_offset = layout.slot_offset(SparseConstituentSlot::ColX);
            for (k, &a) in scratch.col_x.iter().enumerate() {
                combined[col_x_offset + k] = Goldilocks::from(a as u32);
            }
            let read_ts_z_offset = layout.slot_offset(SparseConstituentSlot::ReadTsZ);
            combined[read_ts_z_offset..read_ts_z_offset + scratch.ts_z.read_ts.len()]
                .copy_from_slice(&scratch.ts_z.read_ts);
            let audit_ts_z_offset = layout.slot_offset(SparseConstituentSlot::AuditTsZ);
            combined[audit_ts_z_offset..audit_ts_z_offset + scratch.ts_z.audit_ts.len()]
                .copy_from_slice(&scratch.ts_z.audit_ts);
            let read_ts_x_offset = layout.slot_offset(SparseConstituentSlot::ReadTsX);
            combined[read_ts_x_offset..read_ts_x_offset + scratch.ts_x.read_ts.len()]
                .copy_from_slice(&scratch.ts_x.read_ts);
            let audit_ts_x_offset = layout.slot_offset(SparseConstituentSlot::AuditTsX);
            combined[audit_ts_x_offset..audit_ts_x_offset + scratch.ts_x.audit_ts.len()]
                .copy_from_slice(&scratch.ts_x.audit_ts);
            combined
        };

        let z: Vec<GoldilocksExt4> = (0..3)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt4> = (0..3)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let claimed = poly.evaluate::<GoldilocksExt4>(&z, &x, &[]);

        let mut p_t = Sha2T::new();
        let (full, _open_scratch) = sparse_open_full::<Goldilocks, GoldilocksExt4, Sha2T>(
            &commitment.into_whir_commitment(),
            &combined_evals,
            &codeword,
            &tree,
            &layout,
            &scratch,
            claimed,
            &z,
            &x,
            &[],
            &mut p_t,
        );

        // Eval-claim opens
        assert_eq!(full.constituent_opens.ez_limbs_at_sumcheck.len(), 4);
        assert_eq!(full.constituent_opens.ex_limbs_at_sumcheck.len(), 4);
        assert!(full.constituent_opens.ey_limbs_at_sumcheck.is_none());

        // Per-axis opens
        let z_opens = &full.constituent_opens.axis_z_opens;
        assert_eq!(z_opens.e_at_rs.len(), 4);
        assert_eq!(z_opens.e_at_ws.len(), 4);

        let x_opens = &full.constituent_opens.axis_x_opens;
        assert_eq!(x_opens.e_at_rs.len(), 4);
        assert_eq!(x_opens.e_at_ws.len(), 4);

        assert!(full.constituent_opens.axis_y_opens.is_none());
    }

    #[test]
    fn open_full_three_axis_produces_all_constituent_opens() {
        let mut rng = rng_for("open_full_three_axis");
        let poly = build_three_axis(&mut rng, 3, 3, 3, 8);
        let (commitment, scratch, tree, codeword) = sparse_commit::<Goldilocks>(&poly).unwrap();

        let layout = SparseLayout::compute(
            poly.arity,
            scratch.nnz,
            1usize << scratch.n_z,
            1usize << scratch.n_x,
            1usize << scratch.n_y,
        );
        let combined_evals: Vec<Goldilocks> = {
            let mut combined = vec![Goldilocks::ZERO; layout.combined_len()];
            let val_offset = layout.slot_offset(SparseConstituentSlot::Val);
            combined[val_offset..val_offset + scratch.val.len()].copy_from_slice(&scratch.val);
            for (k, &a) in scratch.row.iter().enumerate() {
                combined[layout.slot_offset(SparseConstituentSlot::Row) + k] =
                    Goldilocks::from(a as u32);
            }
            for (k, &a) in scratch.col_x.iter().enumerate() {
                combined[layout.slot_offset(SparseConstituentSlot::ColX) + k] =
                    Goldilocks::from(a as u32);
            }
            for (k, &a) in scratch.col_y.iter().enumerate() {
                combined[layout.slot_offset(SparseConstituentSlot::ColY) + k] =
                    Goldilocks::from(a as u32);
            }
            for (slot, src) in [
                (SparseConstituentSlot::ReadTsZ, &scratch.ts_z.read_ts),
                (SparseConstituentSlot::AuditTsZ, &scratch.ts_z.audit_ts),
                (SparseConstituentSlot::ReadTsX, &scratch.ts_x.read_ts),
                (SparseConstituentSlot::AuditTsX, &scratch.ts_x.audit_ts),
                (
                    SparseConstituentSlot::ReadTsY,
                    &scratch.ts_y.as_ref().unwrap().read_ts,
                ),
                (
                    SparseConstituentSlot::AuditTsY,
                    &scratch.ts_y.as_ref().unwrap().audit_ts,
                ),
            ] {
                let off = layout.slot_offset(slot);
                combined[off..off + src.len()].copy_from_slice(src);
            }
            combined
        };

        let z: Vec<GoldilocksExt4> = (0..3)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt4> = (0..3)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let y: Vec<GoldilocksExt4> = (0..3)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let claimed = poly.evaluate::<GoldilocksExt4>(&z, &x, &y);

        let mut p_t = Sha2T::new();
        let (full, _open_scratch) = sparse_open_full::<Goldilocks, GoldilocksExt4, Sha2T>(
            &commitment.into_whir_commitment(),
            &combined_evals,
            &codeword,
            &tree,
            &layout,
            &scratch,
            claimed,
            &z,
            &x,
            &y,
            &mut p_t,
        );

        assert_eq!(full.constituent_opens.ez_limbs_at_sumcheck.len(), 4);
        assert_eq!(full.constituent_opens.ex_limbs_at_sumcheck.len(), 4);
        assert_eq!(
            full.constituent_opens
                .ey_limbs_at_sumcheck
                .as_ref()
                .unwrap()
                .len(),
            4
        );
        assert!(full.constituent_opens.axis_y_opens.is_some());
    }

    #[test]
    fn full_opening_serialization_round_trip() {
        let mut rng = rng_for("full_opening_serde");
        let poly = build_two_axis(&mut rng, 2, 2, 4);
        let (commitment, scratch, tree, codeword) = sparse_commit::<Goldilocks>(&poly).unwrap();
        let layout = SparseLayout::compute(
            poly.arity,
            scratch.nnz,
            1usize << scratch.n_z,
            1usize << scratch.n_x,
            1,
        );
        let combined_evals: Vec<Goldilocks> = {
            let mut combined = vec![Goldilocks::ZERO; layout.combined_len()];
            let val_offset = layout.slot_offset(SparseConstituentSlot::Val);
            combined[val_offset..val_offset + scratch.val.len()].copy_from_slice(&scratch.val);
            for (k, &a) in scratch.row.iter().enumerate() {
                combined[layout.slot_offset(SparseConstituentSlot::Row) + k] =
                    Goldilocks::from(a as u32);
            }
            for (k, &a) in scratch.col_x.iter().enumerate() {
                combined[layout.slot_offset(SparseConstituentSlot::ColX) + k] =
                    Goldilocks::from(a as u32);
            }
            for (slot, src) in [
                (SparseConstituentSlot::ReadTsZ, &scratch.ts_z.read_ts),
                (SparseConstituentSlot::AuditTsZ, &scratch.ts_z.audit_ts),
                (SparseConstituentSlot::ReadTsX, &scratch.ts_x.read_ts),
                (SparseConstituentSlot::AuditTsX, &scratch.ts_x.audit_ts),
            ] {
                let off = layout.slot_offset(slot);
                combined[off..off + src.len()].copy_from_slice(src);
            }
            combined
        };
        let z: Vec<GoldilocksExt4> = vec![
            GoldilocksExt4::random_unsafe(&mut rng),
            GoldilocksExt4::random_unsafe(&mut rng),
        ];
        let x: Vec<GoldilocksExt4> = vec![
            GoldilocksExt4::random_unsafe(&mut rng),
            GoldilocksExt4::random_unsafe(&mut rng),
        ];
        let claimed = poly.evaluate::<GoldilocksExt4>(&z, &x, &[]);

        let mut p_t = Sha2T::new();
        let (full, _) = sparse_open_full::<Goldilocks, GoldilocksExt4, Sha2T>(
            &commitment.into_whir_commitment(),
            &combined_evals,
            &codeword,
            &tree,
            &layout,
            &scratch,
            claimed,
            &z,
            &x,
            &[],
            &mut p_t,
        );
        let mut bytes = Vec::new();
        full.serialize_into(&mut bytes).unwrap();
        let decoded =
            SparseMle3FullOpening::<GoldilocksExt4>::deserialize_from(&bytes[..]).unwrap();
        assert_eq!(
            decoded.skeleton.evalclaim.claimed_eval,
            full.skeleton.evalclaim.claimed_eval
        );
        assert_eq!(
            decoded.constituent_opens.ez_limbs_at_sumcheck.len(),
            full.constituent_opens.ez_limbs_at_sumcheck.len()
        );
    }
}

/// Helper extension on `SparseMle3Commitment` to obtain a
/// `WhirCommitment` view that the underlying WHIR open / verify
/// functions consume directly. This is a thin shim — the
/// `SparseMle3Commitment` already carries the Merkle root and the
/// total variable count.
impl super::types::SparseMle3Commitment {
    /// Convert this sparse commitment to the underlying
    /// [`WhirCommitment`] view used by `whir_open` / `whir_verify`.
    /// Uses `self.batched_num_vars` directly so callers cannot
    /// accidentally override the variable count.
    #[must_use]
    pub fn into_whir_commitment(&self) -> WhirCommitment {
        WhirCommitment {
            root: self.batched_root.clone(),
            num_vars: self.batched_num_vars,
        }
    }
}
