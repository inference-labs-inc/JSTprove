use std::io::Cursor;

use arith::Field;
use circuit::Circuit;
use gkr_engine::{
    ExpanderDualVarChallenge, ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, GKREngine,
    Proof, Transcript,
};
use serdes::ExpSerde;
use sumcheck::{GKRVerifierHelper, VerifierScratchPad};
use transcript::transcript_verifier_sync;

use super::prove::LayerEvalPoint;

pub struct PerLayerChallenge<F: FieldEngine> {
    pub layer_index: usize,
    pub rz_0: Vec<F::ChallengeField>,
    pub rz_1: Option<Vec<F::ChallengeField>>,
    pub rx: Vec<F::ChallengeField>,
    pub ry: Option<Vec<F::ChallengeField>>,
    pub r_simd_xy: Vec<F::ChallengeField>,
    pub r_mpi_xy: Vec<F::ChallengeField>,
    pub eval_cst: F::ChallengeField,
    pub eval_add: F::ChallengeField,
    pub eval_mul: F::ChallengeField,
    pub eval_uni: F::ChallengeField,
    pub eq_simd_simd_xy: F::ChallengeField,
    pub eq_mpi_mpi_xy: F::ChallengeField,
}

pub struct ChallengeExtraction<F: FieldEngine> {
    pub layers: Vec<PerLayerChallenge<F>>,
    pub final_challenge: ExpanderDualVarChallenge<F>,
    pub final_claimed_v0: F::ChallengeField,
    pub final_claimed_v1: Option<F::ChallengeField>,
}

#[allow(clippy::too_many_arguments)]
pub fn extract_gkr_layer_challenges<Cfg: GKREngine>(
    circuit: &Circuit<Cfg::FieldConfig>,
    public_input: &[<Cfg::FieldConfig as FieldEngine>::SimdCircuitField],
    claimed_v: &<Cfg::FieldConfig as FieldEngine>::ChallengeField,
    proof: &Proof,
    proving_time_mpi_size: usize,
) -> Option<ChallengeExtraction<Cfg::FieldConfig>>
where
    Cfg::FieldConfig: FieldEngine,
{
    let mut transcript = Cfg::TranscriptConfig::new();
    let mut cursor = Cursor::new(&proof.bytes);

    let commitment =
        <<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::Commitment as ExpSerde>::deserialize_from(
            &mut cursor,
        )
        .ok()?;
    let mut buffer = vec![];
    commitment.serialize_into(&mut buffer).ok()?;
    transcript.append_commitment(&buffer);

    #[cfg(feature = "grinding")]
    crate::prover::snark::grind::<Cfg>(
        &mut transcript,
        &gkr_engine::MPIConfig::verifier_new(proving_time_mpi_size as i32),
    );

    {
        let n_rnd = circuit.rnd_coefs.len();
        let _ = transcript
            .generate_field_elements::<<Cfg::FieldConfig as FieldEngine>::CircuitField>(n_rnd);
    }

    transcript_verifier_sync(&mut transcript, proving_time_mpi_size);

    let layer_num = circuit.layers.len();
    let mut sp = VerifierScratchPad::<Cfg::FieldConfig>::new(circuit, proving_time_mpi_size);

    let output_var_num = circuit.layers.last()?.output_var_num;
    let mut challenge: ExpanderDualVarChallenge<Cfg::FieldConfig> =
        ExpanderSingleVarChallenge::sample_from_transcript(
            &mut transcript,
            output_var_num,
            proving_time_mpi_size,
        )
        .into();

    let mut alpha = None;
    let mut claimed_v0 = *claimed_v;
    let mut claimed_v1: Option<<Cfg::FieldConfig as FieldEngine>::ChallengeField> = None;

    transcript.lock_proof();
    transcript.append_field_element(claimed_v);
    transcript.unlock_proof();

    let mut layers = Vec::with_capacity(layer_num);

    for i in (0..layer_num).rev() {
        let rz_0_before = challenge.rz_0.clone();
        let rz_1_before = challenge.rz_1.clone();

        let layer = &circuit.layers[i];

        let is_output_layer = i == layer_num - 1;
        GKRVerifierHelper::prepare_layer(layer, &alpha, &mut challenge, &mut sp, is_output_layer);

        let var_num = layer.input_var_num;
        let simd_var_num =
            <Cfg::FieldConfig as FieldEngine>::get_field_pack_size().trailing_zeros() as usize;
        let mut sum = claimed_v0;
        if let Some(v1) = claimed_v1 {
            if let Some(a) = alpha {
                sum += v1 * a;
            }
        }

        let eval_cst = GKRVerifierHelper::eval_cst(&layer.const_, public_input, &sp);
        sum -= eval_cst;

        let mut rx = vec![];
        let mut ry = None;
        let mut r_simd_xy = vec![];
        let mut r_mpi_xy = vec![];

        use crate::verify_sumcheck_step;
        use sumcheck::{SUMCHECK_GKR_DEGREE, SUMCHECK_GKR_SIMD_MPI_DEGREE};

        let mut verified = true;
        for _i_var in 0..var_num {
            verified &= verify_sumcheck_step::<Cfg::FieldConfig>(
                &mut cursor,
                SUMCHECK_GKR_DEGREE,
                &mut transcript,
                &mut sum,
                &mut rx,
                &sp,
            );
        }
        GKRVerifierHelper::set_rx(&rx, &mut sp);

        for _i_var in 0..simd_var_num {
            verified &= verify_sumcheck_step::<Cfg::FieldConfig>(
                &mut cursor,
                SUMCHECK_GKR_SIMD_MPI_DEGREE,
                &mut transcript,
                &mut sum,
                &mut r_simd_xy,
                &sp,
            );
        }
        GKRVerifierHelper::set_r_simd_xy(&r_simd_xy, &mut sp);

        for _i_var in 0..proving_time_mpi_size.trailing_zeros() {
            verified &= verify_sumcheck_step::<Cfg::FieldConfig>(
                &mut cursor,
                SUMCHECK_GKR_SIMD_MPI_DEGREE,
                &mut transcript,
                &mut sum,
                &mut r_mpi_xy,
                &sp,
            );
        }
        GKRVerifierHelper::set_r_mpi_xy(&r_mpi_xy, &mut sp);
        if !verified {
            return None;
        }

        let vx_claim =
            <Cfg::FieldConfig as FieldEngine>::ChallengeField::deserialize_from(&mut cursor)
                .ok()?;

        let eval_add = GKRVerifierHelper::eval_add(&layer.add, &sp);
        transcript.append_field_element(&vx_claim);

        let (vy_claim, eval_mul) = if !layer.structure_info.skip_sumcheck_phase_two {
            ry = Some(vec![]);
            for _i_var in 0..var_num {
                verified &= verify_sumcheck_step::<Cfg::FieldConfig>(
                    &mut cursor,
                    SUMCHECK_GKR_DEGREE,
                    &mut transcript,
                    &mut sum,
                    ry.as_mut().unwrap(),
                    &sp,
                );
            }
            if !verified {
                return None;
            }
            GKRVerifierHelper::set_ry(ry.as_ref().unwrap(), &mut sp);

            let vy_claim =
                <Cfg::FieldConfig as FieldEngine>::ChallengeField::deserialize_from(&mut cursor)
                    .ok()?;
            let em = GKRVerifierHelper::eval_mul(&layer.mul, &sp);
            transcript.append_field_element(&vy_claim);
            (Some(vy_claim), em)
        } else {
            (
                None,
                <Cfg::FieldConfig as FieldEngine>::ChallengeField::ZERO,
            )
        };

        challenge = ExpanderDualVarChallenge::new(
            rx.clone(),
            ry.clone(),
            r_simd_xy.clone(),
            r_mpi_xy.clone(),
        );
        claimed_v0 = vx_claim;
        claimed_v1 = vy_claim;

        alpha = if challenge.rz_1.is_some() {
            Some(
                transcript
                    .generate_field_element::<<Cfg::FieldConfig as FieldEngine>::ChallengeField>(),
            )
        } else {
            None
        };

        let eval_uni = GKRVerifierHelper::eval_pow_5(&layer.uni, &sp);

        layers.push(PerLayerChallenge {
            layer_index: i,
            rz_0: rz_0_before,
            rz_1: rz_1_before,
            rx,
            ry,
            r_simd_xy,
            r_mpi_xy,
            eval_cst,
            eval_add,
            eval_mul,
            eval_uni,
            eq_simd_simd_xy: sp.eq_r_simd_r_simd_xy,
            eq_mpi_mpi_xy: sp.eq_r_mpi_r_mpi_xy,
        });
    }

    transcript_verifier_sync(&mut transcript, proving_time_mpi_size);

    Some(ChallengeExtraction {
        layers,
        final_challenge: challenge,
        final_claimed_v0: claimed_v0,
        final_claimed_v1: claimed_v1,
    })
}

pub fn build_eval_points_from_challenges<F: FieldEngine>(
    challenges: &ChallengeExtraction<F>,
    pk: &super::setup::HolographicProvingKey<F>,
) -> Option<Vec<LayerEvalPoint<F::ChallengeField>>> {
    let mut eval_points = Vec::with_capacity(pk.layers.len());

    let mut challenge_by_layer: std::collections::HashMap<usize, &PerLayerChallenge<F>> =
        std::collections::HashMap::new();
    for ch in &challenges.layers {
        challenge_by_layer.insert(ch.layer_index, ch);
    }

    for pk_layer in &pk.layers {
        let ch = challenge_by_layer.get(&pk_layer.layer_index)?;

        let mul_z = ch.rz_0.clone();
        let mul_x = ch.rx.clone();
        let mul_y = ch.ry.clone().unwrap_or_default();
        let add_z = ch.rz_0.clone();
        let add_x = ch.rx.clone();

        let mul_claim = if let Some(ref mul_wiring) = pk_layer.mul {
            mul_wiring
                .poly
                .evaluate::<F::ChallengeField>(&mul_z, &mul_x, &mul_y)
        } else {
            F::ChallengeField::ZERO
        };

        let add_claim = if let Some(ref add_wiring) = pk_layer.add {
            add_wiring
                .poly
                .evaluate::<F::ChallengeField>(&add_z, &add_x, &[])
        } else {
            F::ChallengeField::ZERO
        };

        let uni_z = ch.rz_0.clone();
        let uni_x = ch.rx.clone();
        let uni_claim = if let Some(ref uni_wiring) = pk_layer.uni {
            uni_wiring
                .poly
                .evaluate::<F::ChallengeField>(&uni_z, &uni_x, &[])
        } else {
            F::ChallengeField::ZERO
        };

        let cst_z = ch.rz_0.clone();
        let cst_claim = if let Some(ref cst_wiring) = pk_layer.cst {
            cst_wiring
                .poly
                .evaluate::<F::ChallengeField>(&cst_z, &[], &[])
        } else {
            F::ChallengeField::ZERO
        };

        eval_points.push(LayerEvalPoint {
            layer_index: pk_layer.layer_index,
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
    }

    Some(eval_points)
}
