use std::io::Cursor;

use expander_compiler::expander_circuit;
use expander_compiler::frontend::{ChallengeField, CircuitField, Config};
use expander_compiler::gkr_engine::{
    ExpanderPCS, FieldEngine, GKREngine, MPIConfig, StructuredReferenceString, Transcript,
};
use expander_compiler::serdes::ExpSerde;
use gkr::holographic::combined_proof::{CombinedHolographicProof, PerLayerWiringClaims};
use gkr::holographic::setup::HolographicVerifyingKey;
use gkr::verifier::holographic_gkr::{HolographicLayerInput, holographic_gkr_verify};
use poly_commit::{expander_pcs_init, expander_pcs_init_testing_only};

use crate::runner::errors::RunError;
use crate::runner::main_runner::{auto_decompress_bytes, load_circuit_from_bytes};

/// # Errors
/// Returns `RunError` on deserialization, compilation, or serialization failure.
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

/// # Errors
/// Returns `RunError` on prove, compilation, or serialization failure.
#[allow(clippy::similar_names)]
pub fn prove_holographic_from_bytes<C>(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
) -> Result<Vec<u8>, RunError>
where
    C: Config + GKREngine,
    <<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField:
        arith::FFTField + arith::SimdField<Scalar = CircuitField<C>>,
    ChallengeField<C>: arith::ExtensionField<BaseField = CircuitField<C>>
        + arith::FFTField
        + arith::SimdField<Scalar = ChallengeField<C>>
        + std::ops::Add<CircuitField<C>, Output = ChallengeField<C>>
        + std::ops::Mul<CircuitField<C>, Output = ChallengeField<C>>,
{
    let layered = load_circuit_from_bytes::<C>(circuit_bytes)?;
    let mut expander: expander_circuit::Circuit<<C as GKREngine>::FieldConfig> =
        layered.export_to_expander_flatten();

    let witness_data = auto_decompress_bytes(witness_bytes)?;
    let witness = expander_compiler::circuit::layered::witness::Witness::<C>::deserialize_from(
        Cursor::new(&*witness_data),
    )
    .map_err(|e| RunError::Deserialize(format!("witness: {e:?}")))?;

    let setup_circuit = expander.clone();
    let (proving_key, _vk) =
        gkr::holographic::setup::<<C as GKREngine>::FieldConfig>(setup_circuit)
            .map_err(|e| RunError::Compile(format!("holographic setup: {e}")))?;

    let (simd_input, simd_public_input) = witness.to_simd();
    expander.layers[0].input_vals = simd_input;
    expander.public_input.clone_from(&simd_public_input);

    let single_process = MPIConfig::prover_new();

    let mut prover = gkr::Prover::<C>::new(single_process.clone());
    prover.prepare_mem(&expander);

    let (pcs_params, pcs_proving_key, _, mut pcs_scratch) = expander_pcs_init_testing_only::<
        <C as GKREngine>::FieldConfig,
        <C as GKREngine>::PCSConfig,
    >(
        expander.log_input_size(),
        &single_process,
    );

    let (claimed_v, proof) = prover.prove(
        &mut expander,
        &pcs_params,
        &pcs_proving_key,
        &mut pcs_scratch,
    );

    let extraction = gkr::holographic::extract_gkr_layer_challenges::<C>(
        &expander,
        &expander.public_input,
        &claimed_v,
        &proof,
        1,
    )
    .ok_or_else(|| RunError::Prove("GKR challenge extraction failed".to_string()))?;

    let eval_points =
        gkr::holographic::build_eval_points_from_challenges(&extraction, &proving_key).ok_or_else(
            || RunError::Prove("failed to build eval points from challenges".to_string()),
        )?;

    let mut holo_transcript = <C as GKREngine>::TranscriptConfig::new();
    let holographic_proof = gkr::holographic::prove::<
        <C as GKREngine>::FieldConfig,
        ChallengeField<C>,
        <C as GKREngine>::TranscriptConfig,
    >(&proving_key, &eval_points, &mut holo_transcript)
    .map_err(|e| RunError::Prove(format!("holographic prove: {e}")))?;

    let wiring_claims: Vec<PerLayerWiringClaims<ChallengeField<C>>> = extraction
        .layers
        .iter()
        .map(|ch| PerLayerWiringClaims {
            layer_index: ch.layer_index,
            eval_cst: ch.eval_cst,
            eval_add: ch.eval_add,
            eval_mul: ch.eval_mul,
            eval_uni: ch.eval_uni,
        })
        .collect();

    let combined = CombinedHolographicProof {
        gkr_proof: proof,
        claimed_v,
        holographic_proof,
        wiring_claims,
    };

    let mut bytes = Vec::new();
    combined
        .serialize_into(&mut bytes)
        .map_err(|e| RunError::Serialize(format!("combined holographic proof: {e:?}")))?;
    Ok(bytes)
}

/// # Errors
/// Returns `RunError` on verification or deserialization failure.
#[allow(clippy::similar_names, clippy::too_many_lines)]
pub fn verify_holographic_with_vk<C>(vk_bytes: &[u8], proof_bytes: &[u8]) -> Result<bool, RunError>
where
    C: Config + GKREngine,
    <<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField:
        arith::FFTField + arith::SimdField<Scalar = CircuitField<C>>,
    ChallengeField<C>: arith::ExtensionField<BaseField = CircuitField<C>>
        + arith::FFTField
        + arith::SimdField<Scalar = ChallengeField<C>>
        + std::ops::Add<CircuitField<C>, Output = ChallengeField<C>>
        + std::ops::Mul<CircuitField<C>, Output = ChallengeField<C>>,
{
    use gkr::verifier::holographic_common::{LayerShape, WiringEvals};

    let vk_data = auto_decompress_bytes(vk_bytes)?;
    let vk = HolographicVerifyingKey::deserialize_from(Cursor::new(&*vk_data))
        .map_err(|e| RunError::Deserialize(format!("holographic vk: {e:?}")))?;

    let proof_data = auto_decompress_bytes(proof_bytes)?;
    let combined =
        CombinedHolographicProof::<ChallengeField<C>>::deserialize_from(Cursor::new(&*proof_data))
            .map_err(|e| RunError::Deserialize(format!("combined holographic proof: {e:?}")))?;

    let proving_time_mpi_size = 1usize;

    let mut transcript = <C as GKREngine>::TranscriptConfig::new();
    let mut gkr_cursor = Cursor::new(&combined.gkr_proof.bytes);

    let commitment = <<C as GKREngine>::PCSConfig as ExpanderPCS<
        <C as GKREngine>::FieldConfig,
    >>::Commitment::deserialize_from(&mut gkr_cursor)
    .map_err(|e| RunError::Deserialize(format!("PCS commitment: {e:?}")))?;

    let mut commit_buf = vec![];
    commitment
        .serialize_into(&mut commit_buf)
        .map_err(|e| RunError::Serialize(format!("PCS commitment re-serialize: {e:?}")))?;
    transcript.append_commitment(&commit_buf);

    let _ = transcript.generate_field_elements::<CircuitField<C>>(vk.n_rnd_coefs);
    transcript::transcript_verifier_sync(&mut transcript, proving_time_mpi_size);

    let layer_count = vk.n_layers;
    let mut wiring_claims_by_layer: std::collections::HashMap<
        usize,
        &PerLayerWiringClaims<ChallengeField<C>>,
    > = std::collections::HashMap::new();
    for claim in &combined.wiring_claims {
        wiring_claims_by_layer.insert(claim.layer_index, claim);
    }

    let mut layers_for_gkr = Vec::with_capacity(layer_count);
    for i in (0..layer_count).rev() {
        let vk_layer = &vk.layers[i];
        let claims = wiring_claims_by_layer
            .get(&i)
            .ok_or_else(|| RunError::Verify(format!("missing wiring claims for layer {i}")))?;

        layers_for_gkr.push(HolographicLayerInput {
            shape: LayerShape {
                input_var_num: vk_layer.n_x,
                output_var_num: vk_layer.n_z,
                structure_info: circuit::StructureInfo {
                    skip_sumcheck_phase_two: vk_layer.mul.is_none(),
                },
            },
            wiring: WiringEvals {
                eval_cst: claims.eval_cst,
                eval_add: claims.eval_add,
                eval_mul: claims.eval_mul,
                eval_uni: claims.eval_uni,
            },
        });
    }

    let output_var_num = vk.layers.last().map_or(0, |l| l.n_z);

    let (gkr_verified, challenge, claim_x, claim_y) =
        holographic_gkr_verify::<<C as GKREngine>::FieldConfig>(
            proving_time_mpi_size,
            &layers_for_gkr,
            output_var_num,
            &combined.claimed_v,
            &mut transcript,
            &mut gkr_cursor,
        );

    let mut holo_transcript = <C as GKREngine>::TranscriptConfig::new();
    let holo_eval_points =
        build_holo_eval_points_from_gkr_proof::<C>(&vk, &combined, proving_time_mpi_size)?;

    let holo_verify_result = gkr::holographic::verify::<
        <C as GKREngine>::FieldConfig,
        ChallengeField<C>,
        <C as GKREngine>::TranscriptConfig,
    >(
        &vk,
        &holo_eval_points,
        &combined.holographic_proof,
        &mut holo_transcript,
    );

    let holo_verified = match holo_verify_result {
        Ok(()) => true,
        Err(e) => return Err(RunError::Verify(format!("holographic verify: {e}"))),
    };

    let (pcs_params, _, pcs_verification_key, _) =
        expander_pcs_init::<<C as GKREngine>::FieldConfig, <C as GKREngine>::PCSConfig>(
            vk.log_input_size,
            &MPIConfig::verifier_new(1),
        );

    let mut challenge_x = challenge.challenge_x();
    let mut verified = verify_pcs_opening::<C>(
        &pcs_params,
        &pcs_verification_key,
        &commitment,
        &mut challenge_x,
        &claim_x,
        &mut transcript,
        &mut gkr_cursor,
    );

    if let (Some(mut challenge_y), Some(ref cy)) = (challenge.challenge_y(), claim_y) {
        transcript::transcript_verifier_sync(&mut transcript, proving_time_mpi_size);
        verified &= verify_pcs_opening::<C>(
            &pcs_params,
            &pcs_verification_key,
            &commitment,
            &mut challenge_y,
            cy,
            &mut transcript,
            &mut gkr_cursor,
        );
    }

    Ok(gkr_verified & holo_verified & verified)
}

#[allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::items_after_statements
)]
fn build_holo_eval_points_from_gkr_proof<C>(
    vk: &HolographicVerifyingKey,
    combined: &CombinedHolographicProof<ChallengeField<C>>,
    proving_time_mpi_size: usize,
) -> Result<Vec<gkr::holographic::LayerEvalPoint<ChallengeField<C>>>, RunError>
where
    C: Config + GKREngine,
    <<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField:
        arith::FFTField + arith::SimdField<Scalar = CircuitField<C>>,
    ChallengeField<C>: arith::ExtensionField<BaseField = CircuitField<C>>
        + arith::FFTField
        + arith::SimdField<Scalar = ChallengeField<C>>
        + std::ops::Add<CircuitField<C>, Output = ChallengeField<C>>
        + std::ops::Mul<CircuitField<C>, Output = ChallengeField<C>>,
{
    use arith::Field;
    use gkr_engine::{ExpanderDualVarChallenge, ExpanderSingleVarChallenge};

    let mut transcript = <C as GKREngine>::TranscriptConfig::new();
    let mut cursor = Cursor::new(&combined.gkr_proof.bytes);

    let commitment = <<C as GKREngine>::PCSConfig as ExpanderPCS<
        <C as GKREngine>::FieldConfig,
    >>::Commitment::deserialize_from(&mut cursor)
    .map_err(|e| RunError::Deserialize(format!("PCS commitment (replay): {e:?}")))?;
    let mut commit_buf = vec![];
    commitment
        .serialize_into(&mut commit_buf)
        .map_err(|e| RunError::Serialize(format!("PCS commitment (replay): {e:?}")))?;
    transcript.append_commitment(&commit_buf);

    let _ = transcript.generate_field_elements::<CircuitField<C>>(vk.n_rnd_coefs);
    transcript::transcript_verifier_sync(&mut transcript, proving_time_mpi_size);

    let layer_num = vk.n_layers;
    let output_var_num = vk.layers.last().map_or(0, |l| l.n_z);

    let mut challenge: ExpanderDualVarChallenge<<C as GKREngine>::FieldConfig> =
        ExpanderSingleVarChallenge::sample_from_transcript(
            &mut transcript,
            output_var_num,
            proving_time_mpi_size,
        )
        .into();

    transcript.lock_proof();
    transcript.append_field_element(&combined.claimed_v);
    transcript.unlock_proof();

    let max_var = vk
        .layers
        .iter()
        .map(|l| l.n_x.max(l.n_z))
        .max()
        .unwrap_or(0);
    let mut sp = sumcheck::VerifierScratchPad::<<C as GKREngine>::FieldConfig>::new_with_num_vars(
        max_var,
        proving_time_mpi_size,
    );

    type CF<C> = ChallengeField<C>;

    let mut alpha: Option<CF<C>> = None;
    let mut current_v0 = combined.claimed_v;
    let mut current_v1: Option<CF<C>> = None;

    let mut per_layer_points = Vec::with_capacity(layer_num);

    for i_rev in 0..layer_num {
        let i = layer_num - 1 - i_rev;
        let vk_layer = &vk.layers[i];
        let is_output_layer = i_rev == 0;

        let rz_0 = challenge.rz_0.clone();

        if is_output_layer {
            polynomials::EqPolynomial::<CF<C>>::eq_eval_at(
                &challenge.rz_0,
                &CF::<C>::ONE,
                &mut sp.eq_evals_at_rz0,
                &mut sp.eq_evals_first_part,
                &mut sp.eq_evals_second_part,
            );
        } else {
            let output_len = 1 << challenge.rz_0.len();
            sp.eq_evals_at_rz0[..output_len].copy_from_slice(&sp.eq_evals_at_rx[..output_len]);
            if alpha.is_some() && challenge.rz_1.is_some() {
                let a = alpha.unwrap();
                for j in 0..(1usize << vk_layer.n_z) {
                    sp.eq_evals_at_rz0[j] += a * sp.eq_evals_at_ry[j];
                }
            }
        }
        polynomials::EqPolynomial::<CF<C>>::eq_eval_at(
            &challenge.r_simd,
            &CF::<C>::ONE,
            &mut sp.eq_evals_at_r_simd,
            &mut sp.eq_evals_first_part,
            &mut sp.eq_evals_second_part,
        );
        polynomials::EqPolynomial::<CF<C>>::eq_eval_at(
            &challenge.r_mpi,
            &CF::<C>::ONE,
            &mut sp.eq_evals_at_r_mpi,
            &mut sp.eq_evals_first_part,
            &mut sp.eq_evals_second_part,
        );
        sp.r_simd.clone_from(&challenge.r_simd);
        sp.r_mpi.clone_from(&challenge.r_mpi);

        let var_num = vk_layer.n_x;
        let simd_var_num = <<C as GKREngine>::FieldConfig as FieldEngine>::get_field_pack_size()
            .trailing_zeros() as usize;

        let mut sum = current_v0;
        if let Some(v1) = current_v1 {
            if let Some(a) = alpha {
                sum += v1 * a;
            }
        }

        let claims = combined
            .wiring_claims
            .iter()
            .find(|c| c.layer_index == i)
            .ok_or_else(|| RunError::Verify(format!("missing claims for layer {i} in replay")))?;
        sum -= claims.eval_cst;

        let mut rx = vec![];
        let mut ry = None;
        let mut r_simd_xy = vec![];
        let mut r_mpi_xy = vec![];

        let mut verified = true;
        for _ in 0..var_num {
            verified &= gkr::verify_sumcheck_step::<<C as GKREngine>::FieldConfig>(
                &mut cursor,
                sumcheck::SUMCHECK_GKR_DEGREE,
                &mut transcript,
                &mut sum,
                &mut rx,
                &sp,
            );
        }
        sumcheck::GKRVerifierHelper::<<C as GKREngine>::FieldConfig>::set_rx(&rx, &mut sp);

        for _ in 0..simd_var_num {
            verified &= gkr::verify_sumcheck_step::<<C as GKREngine>::FieldConfig>(
                &mut cursor,
                sumcheck::SUMCHECK_GKR_SIMD_MPI_DEGREE,
                &mut transcript,
                &mut sum,
                &mut r_simd_xy,
                &sp,
            );
        }
        sumcheck::GKRVerifierHelper::<<C as GKREngine>::FieldConfig>::set_r_simd_xy(
            &r_simd_xy, &mut sp,
        );

        for _ in 0..proving_time_mpi_size.trailing_zeros() {
            verified &= gkr::verify_sumcheck_step::<<C as GKREngine>::FieldConfig>(
                &mut cursor,
                sumcheck::SUMCHECK_GKR_SIMD_MPI_DEGREE,
                &mut transcript,
                &mut sum,
                &mut r_mpi_xy,
                &sp,
            );
        }
        sumcheck::GKRVerifierHelper::<<C as GKREngine>::FieldConfig>::set_r_mpi_xy(
            &r_mpi_xy, &mut sp,
        );
        if !verified {
            return Err(RunError::Verify(
                "sumcheck step failed during eval-point extraction".to_string(),
            ));
        }

        let claim_x = CF::<C>::deserialize_from(&mut cursor)
            .map_err(|e| RunError::Deserialize(format!("vx_claim: {e:?}")))?;
        transcript.append_field_element(&claim_x);

        let skip_phase_two = vk_layer.mul.is_none();
        let claim_y = if skip_phase_two {
            None
        } else {
            ry = Some(vec![]);
            for _ in 0..var_num {
                verified &= gkr::verify_sumcheck_step::<<C as GKREngine>::FieldConfig>(
                    &mut cursor,
                    sumcheck::SUMCHECK_GKR_DEGREE,
                    &mut transcript,
                    &mut sum,
                    ry.as_mut().unwrap(),
                    &sp,
                );
            }
            if !verified {
                return Err(RunError::Verify(
                    "phase-two sumcheck step failed during eval-point extraction".to_string(),
                ));
            }
            sumcheck::GKRVerifierHelper::<<C as GKREngine>::FieldConfig>::set_ry(
                ry.as_ref().unwrap(),
                &mut sp,
            );

            let vy = CF::<C>::deserialize_from(&mut cursor)
                .map_err(|e| RunError::Deserialize(format!("vy_claim: {e:?}")))?;
            transcript.append_field_element(&vy);
            Some(vy)
        };

        let mul_z = rz_0.clone();
        let mul_x = rx.clone();
        let mul_y = ry.clone().unwrap_or_default();
        let add_z = rz_0.clone();
        let add_x = rx.clone();
        let rz_0_copy = rz_0;

        let eq_simd_mpi = sp.eq_r_simd_r_simd_xy * sp.eq_r_mpi_r_mpi_xy;
        let eq_simd_mpi_inv = eq_simd_mpi
            .inv()
            .ok_or_else(|| RunError::Verify("eq_simd * eq_mpi is zero".to_string()))?;
        let mul_claim = claims.eval_mul * eq_simd_mpi_inv;
        let add_claim = claims.eval_add * eq_simd_mpi_inv;

        let uni_z = rz_0_copy.clone();
        let uni_x = rx.clone();
        let uni_claim = claims.eval_uni * eq_simd_mpi_inv;
        let cst_z = rz_0_copy;
        let cst_claim = claims.eval_cst * eq_simd_mpi_inv;

        per_layer_points.push(gkr::holographic::LayerEvalPoint {
            layer_index: i,
            mul_z,
            mul_x,
            mul_y,
            add_z,
            add_x,
            uni_z,
            uni_x,
            cst_z,
            mul_claim,
            add_claim,
            uni_claim,
            cst_claim,
        });

        challenge = ExpanderDualVarChallenge::new(rx, ry, r_simd_xy, r_mpi_xy);
        current_v0 = claim_x;
        current_v1 = claim_y;

        alpha = if challenge.rz_1.is_some() {
            Some(transcript.generate_field_element::<CF<C>>())
        } else {
            None
        };
    }

    per_layer_points.sort_by_key(|p| p.layer_index);
    Ok(per_layer_points)
}

#[allow(clippy::too_many_arguments)]
fn verify_pcs_opening<C>(
    pcs_params: &<<C as GKREngine>::PCSConfig as ExpanderPCS<<C as GKREngine>::FieldConfig>>::Params,
    pcs_verification_key: &<<<C as GKREngine>::PCSConfig as ExpanderPCS<
        <C as GKREngine>::FieldConfig,
    >>::SRS as StructuredReferenceString>::VKey,
    commitment: &<<C as GKREngine>::PCSConfig as ExpanderPCS<<C as GKREngine>::FieldConfig>>::Commitment,
    challenge: &mut gkr_engine::ExpanderSingleVarChallenge<<C as GKREngine>::FieldConfig>,
    claim: &ChallengeField<C>,
    transcript: &mut impl gkr_engine::Transcript,
    proof_reader: &mut impl std::io::Read,
) -> bool
where
    C: Config + GKREngine,
{
    let Ok(opening) = <<C as GKREngine>::PCSConfig as ExpanderPCS<
        <C as GKREngine>::FieldConfig,
    >>::Opening::deserialize_from(&mut *proof_reader)
    else {
        return false;
    };

    transcript.lock_proof();
    let verified = <C as GKREngine>::PCSConfig::verify(
        pcs_params,
        pcs_verification_key,
        commitment,
        challenge,
        *claim,
        transcript,
        &opening,
    );
    transcript.unlock_proof();

    let mut buffer = vec![];
    if opening.serialize_into(&mut buffer).is_err() {
        return false;
    }
    transcript.append_u8_slice(&buffer);

    verified
}
