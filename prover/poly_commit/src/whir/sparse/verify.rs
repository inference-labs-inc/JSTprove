//! Verifier for the sparse-MLE WHIR commitment.
//!
//! Phase 1d-4d-iii: replays the entire `sparse_open_full` transcript
//! against the verifying-key commitment + the per-evaluation
//! commitment, verifies every constituent WHIR open, ties the
//! reconstructed extension-field factor evaluations to the eval-
//! claim sumcheck final claims, ties the per-axis multiset leaf
//! claims to the closed-form `leaves_mle` polynomials evaluated
//! against the WHIR-opened constituents, and finally checks the
//! Spartan subset equation
//!
//! ```text
//!     H(Init) · H(WS)  =  H(RS) · H(Audit)
//! ```
//!
//! per address axis. On acceptance, the original sparse-MLE
//! evaluation claim `M(z, x, y) = v` is bound by knowledge soundness
//! to the committed sparse polynomial, and the verifier never had to
//! materialize the polynomial itself — only the verifying key (which
//! is `O(n_layers · 32B)` for an honest setup) and the proof.
//!
//! The closed-form `leaves_mle` polynomials per multiset are:
//!
//! ```text
//!     leaves_mle_Init  (r_Init)  =  γ_1²·c̃_id(r_Init)  + γ_1·ẽq(r_Init, r_axis_outer)                              − γ_2
//!     leaves_mle_RS    (r_RS  )  =  γ_1²·row_axis(r_RS) + γ_1·e_axis(r_RS )                + read_ts_axis(r_RS )    − γ_2
//!     leaves_mle_WS    (r_WS  )  =  γ_1²·row_axis(r_WS) + γ_1·e_axis(r_WS )                + read_ts_axis(r_WS ) + 1 − γ_2
//!     leaves_mle_Audit (r_Audit) =  γ_1²·c̃_id(r_Audit) + γ_1·ẽq(r_Audit, r_axis_outer) + audit_ts_axis(r_Audit) − γ_2
//! ```
//!
//! where `c̃_id(r) = Σ_i 2^i · r_i` is the cell-index identity
//! polynomial's multilinear extension and `ẽq(r, r_outer) = Π_i (r_i
//! · r_outer_i + (1−r_i)·(1−r_outer_i))` is the standard eq-MLE. The
//! Init/Audit `mẽm` factor uses the Spartan §7.2.3 opt (4) shortcut
//! `mẽm_axis(r) = ẽq(r, r_outer)` so the verifier never needs the
//! mẽm polynomial committed.

use arith::{ExtensionField, FFTField, Field, SimdField};
use gkr_engine::Transcript;

use super::combined_point::combined_eval_point;
use super::commit::{SparseConstituentSlot, SparseLayout};
use super::constituent_opens::{PerAxisConstituentOpens, SparseMle3FullOpening};
use super::eval_commit::{
    eval_combined_point, reconstruct_ext_eval, EvalConstituent, SparseEvalLayout, SparseEvalSlot,
};
use super::eval_sumcheck::verify_eval_sumcheck;
use super::multiset_open::PerAxisMultisetProof;
use super::product_argument::verify_product_circuit;
use super::types::SparseArity;
use super::whir_glue::{whir_verify_with_claim, WhirOpenWithClaim};
use crate::whir::types::WhirCommitment;

/// Errors raised by `sparse_verify_full`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseVerifyError {
    /// The eval-claim sumcheck did not pass `verify_eval_sumcheck`.
    EvalClaimSumcheckRejected,
    /// One of the per-axis product-circuit GKR proofs was rejected.
    ProductCircuitRejected {
        axis: AxisLabel,
        which: ProductCircuitWhich,
    },
    /// A WHIR open of a constituent polynomial did not pass
    /// `whir_verify_with_claim`.
    WhirOpenRejected { slot: ConstituentLabel },
    /// The reconstructed extension-field factor evaluation
    /// (`reconstruct_ext_eval` on the per-limb opens) did not match
    /// the eval-claim sumcheck's final factor evaluation.
    FactorEvalMismatch { constituent: EvalConstituent },
    /// The reconstructed `leaves_mle(r)` did not match the leaf
    /// evaluation reported by the product-circuit verifier.
    LeafMleMismatch {
        axis: AxisLabel,
        which: ProductCircuitWhich,
    },
    /// The per-axis subset equation `H(Init)·H(WS) = H(RS)·H(Audit)`
    /// did not hold.
    SubsetEquationFailed { axis: AxisLabel },
    /// `arity` and the presence/absence of `y` data disagree.
    AritySchemaMismatch,
    /// Required commitment / opening field is missing for the
    /// declared arity.
    MissingForArity { what: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisLabel {
    Z,
    X,
    Y,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductCircuitWhich {
    Init,
    Rs,
    Ws,
    Audit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstituentLabel {
    Val,
    Row(AxisLabel),
    ReadTs(AxisLabel),
    AuditTs(AxisLabel),
    EzLimb(usize),
    ExLimb(usize),
    EyLimb(usize),
    EAtRsLimb(AxisLabel, usize),
    EAtWsLimb(AxisLabel, usize),
}

impl std::fmt::Display for SparseVerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EvalClaimSumcheckRejected => write!(f, "eval-claim sumcheck rejected"),
            Self::ProductCircuitRejected { axis, which } => {
                write!(f, "product circuit rejected: axis={axis:?} which={which:?}")
            }
            Self::WhirOpenRejected { slot } => {
                write!(f, "WHIR open rejected for {slot:?}")
            }
            Self::FactorEvalMismatch { constituent } => {
                write!(f, "factor eval mismatch for {constituent:?}")
            }
            Self::LeafMleMismatch { axis, which } => {
                write!(f, "leaves_mle mismatch: axis={axis:?} which={which:?}")
            }
            Self::SubsetEquationFailed { axis } => {
                write!(f, "subset equation failed on axis {axis:?}")
            }
            Self::AritySchemaMismatch => {
                write!(f, "arity does not match opening / commitment shape")
            }
            Self::MissingForArity { what } => {
                write!(f, "missing field {what} required by arity")
            }
        }
    }
}

impl std::error::Error for SparseVerifyError {}

// Intentionally NO blanket From<WhirGlueError>. Each call site
// maps the error with the correct ConstituentLabel via
// verify_one_open's explicit .map_err. A blanket impl here
// would lose the label context.

/// Run the sparse-MLE WHIR verifier.
///
/// `setup_commitment` is the public commitment from the verifying
/// key. `setup_layout` describes how the constituents are packed
/// into the combined polynomial; both prover and verifier compute
/// it identically from `(arity, nnz, n_z, n_x, n_y)` so it can be
/// re-derived rather than transmitted.
///
/// Returns `Ok(())` on acceptance and `Err(SparseVerifyError)` on
/// any failure. The error variants are documented above.
#[allow(clippy::too_many_arguments)]
pub fn sparse_verify_full<F, E, T>(
    setup_commitment: &WhirCommitment,
    setup_layout: &SparseLayout,
    arity: SparseArity,
    log_nnz: usize,
    n_z: usize,
    n_x: usize,
    n_y: usize,
    z: &[E],
    x: &[E],
    y: &[E],
    full_opening: &SparseMle3FullOpening<E>,
    transcript: &mut T,
) -> Result<(), SparseVerifyError>
where
    F: FFTField + SimdField<Scalar = F>,
    E: ExtensionField<BaseField = F>
        + FFTField
        + SimdField<Scalar = E>
        + std::ops::Add<F, Output = E>
        + std::ops::Mul<F, Output = E>,
    T: Transcript,
{
    // ----- Schema sanity checks -------------------------------------
    if z.len() != n_z || x.len() != n_x {
        return Err(SparseVerifyError::AritySchemaMismatch);
    }
    if arity == SparseArity::Three {
        if y.len() != n_y {
            return Err(SparseVerifyError::AritySchemaMismatch);
        }
    } else if !y.is_empty() {
        return Err(SparseVerifyError::AritySchemaMismatch);
    }
    let opens = &full_opening.constituent_opens;
    if (arity == SparseArity::Three) != opens.ey_limbs_at_sumcheck.is_some() {
        return Err(SparseVerifyError::AritySchemaMismatch);
    }
    if (arity == SparseArity::Three) != opens.axis_y_opens.is_some() {
        return Err(SparseVerifyError::AritySchemaMismatch);
    }

    // ----- Step 1: replay the eval-claim sumcheck -------------------
    let eval_claim = verify_eval_sumcheck::<E>(
        arity,
        log_nnz,
        full_opening.skeleton.evalclaim.claimed_eval,
        &full_opening.skeleton.evalclaim.eval_sumcheck,
        transcript,
    )
    .ok_or(SparseVerifyError::EvalClaimSumcheckRejected)?;
    let r_sc = eval_claim.challenges;
    // final_evals layout: [val, e_z, e_x, [e_y]]
    let val_eval_claim = eval_claim.final_evals[0];
    let ez_eval_claim = eval_claim.final_evals[1];
    let ex_eval_claim = eval_claim.final_evals[2];
    let ey_eval_claim = if arity == SparseArity::Three {
        Some(eval_claim.final_evals[3])
    } else {
        None
    };

    // ----- Step 2: append per-eval commitment root + sample γ -----
    transcript.append_u8_slice(full_opening.skeleton.eval_commitment.root.as_bytes());
    let gamma_1: E = transcript.generate_field_element();
    let gamma_2: E = transcript.generate_field_element();

    // ----- Step 3: per-axis multiset random points via verifier replay
    let axis_z_random = extract_axis_random_points::<E>(
        &full_opening.skeleton.multiset.axis_z,
        n_z,
        log_nnz,
        AxisLabel::Z,
        transcript,
    )?;
    let axis_x_random = extract_axis_random_points::<E>(
        &full_opening.skeleton.multiset.axis_x,
        n_x,
        log_nnz,
        AxisLabel::X,
        transcript,
    )?;
    let axis_y_random = if arity == SparseArity::Three {
        let axis_y = full_opening.skeleton.multiset.axis_y.as_ref().ok_or(
            SparseVerifyError::MissingForArity {
                what: "multiset.axis_y",
            },
        )?;
        Some(extract_axis_random_points::<E>(
            axis_y,
            n_y,
            log_nnz,
            AxisLabel::Y,
            transcript,
        )?)
    } else {
        None
    };

    // ----- Step 4: build the eval commitment as a WhirCommitment view
    let eval_layout = SparseEvalLayout::compute(
        arity,
        full_opening.skeleton.eval_commitment.degree,
        1usize << log_nnz,
    );
    let eval_commitment = WhirCommitment {
        root: full_opening.skeleton.eval_commitment.root.clone(),
        num_vars: full_opening.skeleton.eval_commitment.batched_num_vars,
    };

    // ----- Step 5: verify every WHIR open --------------------------
    // 5a: val at r_sc
    {
        let padded = pad_to(&r_sc, setup_layout.mu);
        let combined = combined_eval_point(setup_layout, SparseConstituentSlot::Val, &padded);
        verify_one_open::<F, E, T>(
            setup_commitment,
            &combined,
            &opens.val_at_sumcheck,
            ConstituentLabel::Val,
            transcript,
        )?;
        if opens.val_at_sumcheck.claim != val_eval_claim {
            return Err(SparseVerifyError::FactorEvalMismatch {
                constituent: EvalConstituent::Ez, // placeholder; will refine
            });
        }
    }

    // 5b: e_z limbs at r_sc → reconstruct → check
    {
        let limb_claims = verify_eval_constituent_limbs::<F, E, T, _>(
            &eval_commitment,
            &eval_layout,
            EvalConstituent::Ez,
            &r_sc,
            &opens.ez_limbs_at_sumcheck,
            ConstituentLabel::EzLimb,
            transcript,
        )?;
        let ez_recon = reconstruct_ext_eval(&limb_claims);
        if ez_recon != ez_eval_claim {
            return Err(SparseVerifyError::FactorEvalMismatch {
                constituent: EvalConstituent::Ez,
            });
        }
    }

    // 5c: e_x limbs at r_sc → reconstruct → check
    {
        let limb_claims = verify_eval_constituent_limbs::<F, E, T, _>(
            &eval_commitment,
            &eval_layout,
            EvalConstituent::Ex,
            &r_sc,
            &opens.ex_limbs_at_sumcheck,
            ConstituentLabel::ExLimb,
            transcript,
        )?;
        let ex_recon = reconstruct_ext_eval(&limb_claims);
        if ex_recon != ex_eval_claim {
            return Err(SparseVerifyError::FactorEvalMismatch {
                constituent: EvalConstituent::Ex,
            });
        }
    }

    // 5d: e_y limbs at r_sc → reconstruct → check (3-axis only)
    if arity == SparseArity::Three {
        let ey_opens =
            opens
                .ey_limbs_at_sumcheck
                .as_ref()
                .ok_or(SparseVerifyError::MissingForArity {
                    what: "ey_limbs_at_sumcheck",
                })?;
        let limb_claims = verify_eval_constituent_limbs::<F, E, T, _>(
            &eval_commitment,
            &eval_layout,
            EvalConstituent::Ey,
            &r_sc,
            ey_opens,
            ConstituentLabel::EyLimb,
            transcript,
        )?;
        let ey_recon = reconstruct_ext_eval(&limb_claims);
        if Some(ey_recon) != ey_eval_claim {
            return Err(SparseVerifyError::FactorEvalMismatch {
                constituent: EvalConstituent::Ey,
            });
        }
    }

    // ----- Step 6: per-axis multiset leaf-MLE consistency ----------
    verify_per_axis::<F, E, T>(
        setup_commitment,
        setup_layout,
        &eval_commitment,
        &eval_layout,
        AxisLabel::Z,
        z,
        n_z,
        SparseConstituentSlot::Row,
        SparseConstituentSlot::ReadTsZ,
        SparseConstituentSlot::AuditTsZ,
        EvalConstituent::Ez,
        &full_opening.skeleton.multiset.axis_z,
        &opens.axis_z_opens,
        &axis_z_random,
        gamma_1,
        gamma_2,
        transcript,
    )?;

    verify_per_axis::<F, E, T>(
        setup_commitment,
        setup_layout,
        &eval_commitment,
        &eval_layout,
        AxisLabel::X,
        x,
        n_x,
        SparseConstituentSlot::ColX,
        SparseConstituentSlot::ReadTsX,
        SparseConstituentSlot::AuditTsX,
        EvalConstituent::Ex,
        &full_opening.skeleton.multiset.axis_x,
        &opens.axis_x_opens,
        &axis_x_random,
        gamma_1,
        gamma_2,
        transcript,
    )?;

    if arity == SparseArity::Three {
        let axis_y_random = axis_y_random
            .as_ref()
            .ok_or(SparseVerifyError::MissingForArity {
                what: "axis_y_random",
            })?;
        let axis_y_multiset = full_opening.skeleton.multiset.axis_y.as_ref().ok_or(
            SparseVerifyError::MissingForArity {
                what: "multiset.axis_y",
            },
        )?;
        let axis_y_opens =
            opens
                .axis_y_opens
                .as_ref()
                .ok_or(SparseVerifyError::MissingForArity {
                    what: "axis_y_opens",
                })?;
        verify_per_axis::<F, E, T>(
            setup_commitment,
            setup_layout,
            &eval_commitment,
            &eval_layout,
            AxisLabel::Y,
            y,
            n_y,
            SparseConstituentSlot::ColY,
            SparseConstituentSlot::ReadTsY,
            SparseConstituentSlot::AuditTsY,
            EvalConstituent::Ey,
            axis_y_multiset,
            axis_y_opens,
            axis_y_random,
            gamma_1,
            gamma_2,
            transcript,
        )?;
    }

    // ----- Step 7: per-axis subset equations -----------------------
    if !full_opening
        .skeleton
        .multiset
        .axis_z
        .check_subset_equation()
    {
        return Err(SparseVerifyError::SubsetEquationFailed { axis: AxisLabel::Z });
    }
    if !full_opening
        .skeleton
        .multiset
        .axis_x
        .check_subset_equation()
    {
        return Err(SparseVerifyError::SubsetEquationFailed { axis: AxisLabel::X });
    }
    if arity == SparseArity::Three {
        let axis_y = full_opening.skeleton.multiset.axis_y.as_ref().unwrap();
        if !axis_y.check_subset_equation() {
            return Err(SparseVerifyError::SubsetEquationFailed { axis: AxisLabel::Y });
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct PerAxisRandomPoints<E: Field> {
    pub r_init: Vec<E>,
    pub r_rs: Vec<E>,
    pub r_ws: Vec<E>,
    pub r_audit: Vec<E>,
    /// Leaf-MLE evaluations the product-circuit GKR verifier asserts
    /// at each `r_*` random point. The verifier must additionally
    /// check that these match the closed-form `leaves_mle` values
    /// computed from the WHIR opens of the constituent polynomials.
    pub leaf_eval_init: E,
    pub leaf_eval_rs: E,
    pub leaf_eval_ws: E,
    pub leaf_eval_audit: E,
}

fn extract_axis_random_points<E>(
    proof: &PerAxisMultisetProof<E>,
    log_m_axis: usize,
    log_nnz: usize,
    axis: AxisLabel,
    transcript: &mut impl Transcript,
) -> Result<PerAxisRandomPoints<E>, SparseVerifyError>
where
    E: ExtensionField,
{
    let init = verify_product_circuit::<E>(log_m_axis, proof.h_init, &proof.init_proof, transcript)
        .ok_or(SparseVerifyError::ProductCircuitRejected {
            axis,
            which: ProductCircuitWhich::Init,
        })?;
    let rs = verify_product_circuit::<E>(log_nnz, proof.h_rs, &proof.rs_proof, transcript).ok_or(
        SparseVerifyError::ProductCircuitRejected {
            axis,
            which: ProductCircuitWhich::Rs,
        },
    )?;
    let ws = verify_product_circuit::<E>(log_nnz, proof.h_ws, &proof.ws_proof, transcript).ok_or(
        SparseVerifyError::ProductCircuitRejected {
            axis,
            which: ProductCircuitWhich::Ws,
        },
    )?;
    let audit =
        verify_product_circuit::<E>(log_m_axis, proof.h_audit, &proof.audit_proof, transcript)
            .ok_or(SparseVerifyError::ProductCircuitRejected {
                axis,
                which: ProductCircuitWhich::Audit,
            })?;
    Ok(PerAxisRandomPoints {
        r_init: init.leaf_point,
        r_rs: rs.leaf_point,
        r_ws: ws.leaf_point,
        r_audit: audit.leaf_point,
        leaf_eval_init: init.leaf_eval,
        leaf_eval_rs: rs.leaf_eval,
        leaf_eval_ws: ws.leaf_eval,
        leaf_eval_audit: audit.leaf_eval,
    })
}

fn verify_one_open<F, E, T>(
    commitment: &WhirCommitment,
    eval_point: &[E],
    open: &WhirOpenWithClaim<E>,
    label: ConstituentLabel,
    transcript: &mut T,
) -> Result<(), SparseVerifyError>
where
    F: FFTField,
    E: ExtensionField<BaseField = F> + FFTField,
    T: Transcript,
{
    whir_verify_with_claim::<F, E>(commitment, eval_point, open, transcript)
        .map_err(|_| SparseVerifyError::WhirOpenRejected { slot: label })
}

#[allow(clippy::too_many_arguments)]
fn verify_eval_constituent_limbs<F, E, T, L>(
    eval_commitment: &WhirCommitment,
    eval_layout: &SparseEvalLayout,
    constituent: EvalConstituent,
    sub_point: &[E],
    opens: &[WhirOpenWithClaim<E>],
    label_ctor: L,
    transcript: &mut T,
) -> Result<Vec<E>, SparseVerifyError>
where
    F: FFTField + SimdField<Scalar = F>,
    E: ExtensionField<BaseField = F>
        + FFTField
        + SimdField<Scalar = E>
        + std::ops::Add<F, Output = E>
        + std::ops::Mul<F, Output = E>,
    T: Transcript,
    L: Fn(usize) -> ConstituentLabel,
{
    if opens.len() != eval_layout.degree {
        return Err(SparseVerifyError::AritySchemaMismatch);
    }
    let mut claims = Vec::with_capacity(eval_layout.degree);
    for (limb, open) in opens.iter().enumerate() {
        let slot = SparseEvalSlot { constituent, limb };
        let combined = eval_combined_point(eval_layout, slot, sub_point);
        verify_one_open::<F, E, T>(
            eval_commitment,
            &combined,
            open,
            label_ctor(limb),
            transcript,
        )?;
        claims.push(open.claim);
    }
    Ok(claims)
}

#[allow(clippy::too_many_arguments)]
fn verify_per_axis<F, E, T>(
    setup_commitment: &WhirCommitment,
    setup_layout: &SparseLayout,
    eval_commitment: &WhirCommitment,
    eval_layout: &SparseEvalLayout,
    axis: AxisLabel,
    r_axis_outer: &[E],
    n_axis: usize,
    row_slot: SparseConstituentSlot,
    read_ts_slot: SparseConstituentSlot,
    audit_ts_slot: SparseConstituentSlot,
    eval_constituent: EvalConstituent,
    _multiset: &PerAxisMultisetProof<E>,
    opens: &PerAxisConstituentOpens<E>,
    random: &PerAxisRandomPoints<E>,
    gamma_1: E,
    gamma_2: E,
    transcript: &mut T,
) -> Result<(), SparseVerifyError>
where
    F: FFTField + SimdField<Scalar = F>,
    E: ExtensionField<BaseField = F>
        + FFTField
        + SimdField<Scalar = E>
        + std::ops::Add<F, Output = E>
        + std::ops::Mul<F, Output = E>,
    T: Transcript,
{
    let gamma_1_sq = gamma_1.square();

    // ----- RS: opens row, read_ts at r_RS; e_axis limbs at r_RS ---
    {
        let padded = pad_to(&random.r_rs, setup_layout.mu);
        let row_point = combined_eval_point(setup_layout, row_slot, &padded);
        verify_one_open::<F, E, T>(
            setup_commitment,
            &row_point,
            &opens.row_at_rs,
            ConstituentLabel::Row(axis),
            transcript,
        )?;
        let read_ts_point = combined_eval_point(setup_layout, read_ts_slot, &padded);
        verify_one_open::<F, E, T>(
            setup_commitment,
            &read_ts_point,
            &opens.read_ts_at_rs,
            ConstituentLabel::ReadTs(axis),
            transcript,
        )?;
        let e_limb_claims = verify_eval_constituent_limbs::<F, E, T, _>(
            eval_commitment,
            eval_layout,
            eval_constituent,
            &random.r_rs,
            &opens.e_at_rs,
            move |limb| ConstituentLabel::EAtRsLimb(axis, limb),
            transcript,
        )?;
        let e_recon = reconstruct_ext_eval(&e_limb_claims);
        // leaves_mle_RS(r_RS) = γ₁²·row(r_RS) + γ₁·e(r_RS) + read_ts(r_RS) − γ₂
        let leaves_mle_rs =
            gamma_1_sq * opens.row_at_rs.claim + gamma_1 * e_recon + opens.read_ts_at_rs.claim
                - gamma_2;
        // ProductCircuitClaim's leaf_eval is the *leaf MLE*, which
        // is exactly leaves_mle_RS(r_RS) by construction (the
        // product circuit's leaves are h_γ(addr, val, ts) − γ_2).
        if leaves_mle_rs != random.leaf_eval_rs {
            return Err(SparseVerifyError::LeafMleMismatch {
                axis,
                which: ProductCircuitWhich::Rs,
            });
        }
    }

    // ----- WS: opens row, read_ts at r_WS; e_axis limbs at r_WS ---
    {
        let padded = pad_to(&random.r_ws, setup_layout.mu);
        let row_point = combined_eval_point(setup_layout, row_slot, &padded);
        verify_one_open::<F, E, T>(
            setup_commitment,
            &row_point,
            &opens.row_at_ws,
            ConstituentLabel::Row(axis),
            transcript,
        )?;
        let read_ts_point = combined_eval_point(setup_layout, read_ts_slot, &padded);
        verify_one_open::<F, E, T>(
            setup_commitment,
            &read_ts_point,
            &opens.read_ts_at_ws,
            ConstituentLabel::ReadTs(axis),
            transcript,
        )?;
        let e_limb_claims = verify_eval_constituent_limbs::<F, E, T, _>(
            eval_commitment,
            eval_layout,
            eval_constituent,
            &random.r_ws,
            &opens.e_at_ws,
            move |limb| ConstituentLabel::EAtWsLimb(axis, limb),
            transcript,
        )?;
        let e_recon = reconstruct_ext_eval(&e_limb_claims);
        // leaves_mle_WS(r_WS) = γ₁²·row(r_WS) + γ₁·e(r_WS) + (read_ts(r_WS) + 1) − γ₂
        let leaves_mle_ws = gamma_1_sq * opens.row_at_ws.claim
            + gamma_1 * e_recon
            + opens.read_ts_at_ws.claim
            + E::ONE
            - gamma_2;
        if leaves_mle_ws != random.leaf_eval_ws {
            return Err(SparseVerifyError::LeafMleMismatch {
                axis,
                which: ProductCircuitWhich::Ws,
            });
        }
    }

    // ----- Audit: open audit_ts at r_Audit; verifier computes mem locally
    {
        let padded = pad_to(&random.r_audit, setup_layout.mu);
        let audit_point = combined_eval_point(setup_layout, audit_ts_slot, &padded);
        verify_one_open::<F, E, T>(
            setup_commitment,
            &audit_point,
            &opens.audit_ts_at_audit,
            ConstituentLabel::AuditTs(axis),
            transcript,
        )?;
        // c̃_id(r_audit) = Σ_i 2^i · r_audit[i]
        let c_id = c_id_mle::<E>(&random.r_audit);
        // ẽq(r_audit, r_axis_outer)
        let mem = eq_mle::<E>(&random.r_audit, r_axis_outer);
        // leaves_mle_Audit(r_Audit) = γ₁²·c̃_id(r_audit) + γ₁·ẽq(r_audit, r_axis) + audit_ts(r_audit) − γ₂
        let leaves_mle_audit =
            gamma_1_sq * c_id + gamma_1 * mem + opens.audit_ts_at_audit.claim - gamma_2;
        if leaves_mle_audit != random.leaf_eval_audit {
            return Err(SparseVerifyError::LeafMleMismatch {
                axis,
                which: ProductCircuitWhich::Audit,
            });
        }
        let _ = n_axis; // referenced for documentation symmetry
    }

    // ----- Init: no opens; verifier computes everything ----------
    {
        let c_id = c_id_mle::<E>(&random.r_init);
        let mem = eq_mle::<E>(&random.r_init, r_axis_outer);
        // leaves_mle_Init(r_Init) = γ₁²·c̃_id(r_init) + γ₁·ẽq(r_init, r_axis) + 0 − γ₂
        let leaves_mle_init = gamma_1_sq * c_id + gamma_1 * mem - gamma_2;
        if leaves_mle_init != random.leaf_eval_init {
            return Err(SparseVerifyError::LeafMleMismatch {
                axis,
                which: ProductCircuitWhich::Init,
            });
        }
    }

    Ok(())
}

/// `c̃_id(r) = Σ_i 2^i · r_i` — the multilinear extension of the
/// integer-valued cell-index function `c → c` over `{0,1}^k`.
fn c_id_mle<E: Field>(r: &[E]) -> E {
    let mut acc = E::ZERO;
    let mut weight = E::ONE;
    for ri in r {
        acc += weight * *ri;
        weight = weight + weight; // doubling
    }
    acc
}

/// `ẽq(a, b) = Π_i (a_i · b_i + (1 − a_i)·(1 − b_i))` for two
/// equal-length tuples in `E`. The standard eq-polynomial MLE.
fn eq_mle<E: Field>(a: &[E], b: &[E]) -> E {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = E::ONE;
    for (ai, bi) in a.iter().zip(b.iter()) {
        acc *= *ai * *bi + (E::ONE - *ai) * (E::ONE - *bi);
    }
    acc
}

/// Pad a sub-point to a target length by appending `E::ZERO`.
fn pad_to<E: Field>(point: &[E], target: usize) -> Vec<E> {
    debug_assert!(point.len() <= target);
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
    use crate::whir::sparse::constituent_opens::sparse_open_full;
    use crate::whir::sparse::types::{SparseMle3, SparseMleScratchPad};
    use arith::Field;
    use goldilocks::{Goldilocks, GoldilocksExt4};
    use rand::{Rng, SeedableRng};
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

    /// Build the same combined dense vector that `sparse_commit`
    /// produces internally, so the test can pass it to
    /// `sparse_open_full` (which currently takes the combined evals
    /// as a parameter rather than re-deriving them from the scratch).
    fn build_combined_evals_two_axis(
        scratch: &SparseMleScratchPad<Goldilocks>,
        layout: &SparseLayout,
    ) -> Vec<Goldilocks> {
        let mut combined = vec![Goldilocks::ZERO; layout.combined_len()];
        let val_off = layout.slot_offset(SparseConstituentSlot::Val);
        combined[val_off..val_off + scratch.val.len()].copy_from_slice(&scratch.val);
        let row_off = layout.slot_offset(SparseConstituentSlot::Row);
        for (k, &a) in scratch.row.iter().enumerate() {
            combined[row_off + k] = Goldilocks::from(a as u32);
        }
        let col_x_off = layout.slot_offset(SparseConstituentSlot::ColX);
        for (k, &a) in scratch.col_x.iter().enumerate() {
            combined[col_x_off + k] = Goldilocks::from(a as u32);
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
    }

    fn build_combined_evals_three_axis(
        scratch: &SparseMleScratchPad<Goldilocks>,
        layout: &SparseLayout,
    ) -> Vec<Goldilocks> {
        let mut combined = vec![Goldilocks::ZERO; layout.combined_len()];
        let val_off = layout.slot_offset(SparseConstituentSlot::Val);
        combined[val_off..val_off + scratch.val.len()].copy_from_slice(&scratch.val);
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
    }

    #[test]
    fn end_to_end_round_trip_two_axis() {
        let mut rng = rng_for("e2e_two_axis");
        let n_z = 3;
        let n_x = 3;
        let nnz = 8;
        let poly = build_two_axis(&mut rng, n_z, n_x, nnz);

        let (commitment, scratch, tree, codeword) = sparse_commit::<Goldilocks>(&poly).unwrap();
        let layout = SparseLayout::compute(
            poly.arity,
            scratch.nnz,
            1 << scratch.n_z,
            1 << scratch.n_x,
            1,
        );
        let combined_evals = build_combined_evals_two_axis(&scratch, &layout);

        let z: Vec<GoldilocksExt4> = (0..n_z)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt4> = (0..n_x)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let claimed = poly.evaluate::<GoldilocksExt4>(&z, &x, &[]);

        let mut p_t = Sha2T::new();
        let whir_commitment = commitment.into_whir_commitment();
        let (full, _open_scratch) = sparse_open_full::<Goldilocks, GoldilocksExt4, Sha2T>(
            &whir_commitment,
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

        let mut v_t = Sha2T::new();
        sparse_verify_full::<Goldilocks, GoldilocksExt4, Sha2T>(
            &whir_commitment,
            &layout,
            poly.arity,
            scratch.log_nnz,
            n_z,
            n_x,
            0,
            &z,
            &x,
            &[],
            &full,
            &mut v_t,
        )
        .expect("verifier must accept honest opening");
    }

    #[test]
    fn end_to_end_round_trip_three_axis() {
        let mut rng = rng_for("e2e_three_axis");
        let n_z = 3;
        let n_x = 3;
        let n_y = 3;
        let nnz = 8;
        let poly = build_three_axis(&mut rng, n_z, n_x, n_y, nnz);

        let (commitment, scratch, tree, codeword) = sparse_commit::<Goldilocks>(&poly).unwrap();
        let layout = SparseLayout::compute(
            poly.arity,
            scratch.nnz,
            1 << scratch.n_z,
            1 << scratch.n_x,
            1 << scratch.n_y,
        );
        let combined_evals = build_combined_evals_three_axis(&scratch, &layout);

        let z: Vec<GoldilocksExt4> = (0..n_z)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt4> = (0..n_x)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let y: Vec<GoldilocksExt4> = (0..n_y)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let claimed = poly.evaluate::<GoldilocksExt4>(&z, &x, &y);

        let mut p_t = Sha2T::new();
        let whir_commitment = commitment.into_whir_commitment();
        let (full, _open_scratch) = sparse_open_full::<Goldilocks, GoldilocksExt4, Sha2T>(
            &whir_commitment,
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

        let mut v_t = Sha2T::new();
        sparse_verify_full::<Goldilocks, GoldilocksExt4, Sha2T>(
            &whir_commitment,
            &layout,
            poly.arity,
            scratch.log_nnz,
            n_z,
            n_x,
            n_y,
            &z,
            &x,
            &y,
            &full,
            &mut v_t,
        )
        .expect("verifier must accept honest 3-axis opening");
    }

    #[test]
    fn verify_rejects_wrong_initial_claim() {
        let mut rng = rng_for("e2e_wrong_claim");
        let n_z = 3;
        let n_x = 3;
        let nnz = 8;
        let poly = build_two_axis(&mut rng, n_z, n_x, nnz);

        let (commitment, scratch, tree, codeword) = sparse_commit::<Goldilocks>(&poly).unwrap();
        let layout = SparseLayout::compute(
            poly.arity,
            scratch.nnz,
            1 << scratch.n_z,
            1 << scratch.n_x,
            1,
        );
        let combined_evals = build_combined_evals_two_axis(&scratch, &layout);

        let z: Vec<GoldilocksExt4> = (0..n_z)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt4> = (0..n_x)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let claimed = poly.evaluate::<GoldilocksExt4>(&z, &x, &[]);
        let wrong_claimed = claimed + GoldilocksExt4::ONE;

        let mut p_t = Sha2T::new();
        let whir_commitment = commitment.into_whir_commitment();
        let (mut full, _) = sparse_open_full::<Goldilocks, GoldilocksExt4, Sha2T>(
            &whir_commitment,
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
        // Substitute the wrong claimed eval into the opening so the
        // verifier sees a value that does not match the eval-claim
        // sumcheck transcript.
        full.skeleton.evalclaim.claimed_eval = wrong_claimed;

        let mut v_t = Sha2T::new();
        let result = sparse_verify_full::<Goldilocks, GoldilocksExt4, Sha2T>(
            &whir_commitment,
            &layout,
            poly.arity,
            scratch.log_nnz,
            n_z,
            n_x,
            0,
            &z,
            &x,
            &[],
            &full,
            &mut v_t,
        );
        assert!(
            matches!(result, Err(SparseVerifyError::EvalClaimSumcheckRejected)),
            "verifier must reject when initial claim does not match the proof; got {result:?}"
        );
    }
}
