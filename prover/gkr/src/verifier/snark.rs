use std::{
    io::{Cursor, Read},
    marker::PhantomData,
    vec,
};

use super::gkr_square::sumcheck_verify_gkr_square_layer;
use circuit::{Circuit, RndCoefMap};
use gkr_engine::{
    ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, GKREngine, GKRScheme, MPIConfig,
    MPIEngine, Proof, StructuredReferenceString, Transcript,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use serdes::ExpSerde;
use sumcheck::{VerifierScratchPad, SUMCHECK_GKR_DEGREE, SUMCHECK_GKR_SQUARE_DEGREE};
use transcript::transcript_verifier_sync;
use utils::timer::Timer;

#[cfg(feature = "grinding")]
use crate::grind;
use crate::{
    gkr_square_verify, gkr_verify, gkr_verify_ref, parse_proof, sumcheck_verify_gkr_layer,
};

#[derive(Default)]
pub struct Verifier<Cfg: GKREngine> {
    pub mpi_config: MPIConfig,
    phantom: PhantomData<Cfg>,
}

impl<Cfg: GKREngine> Verifier<Cfg> {
    pub fn new(mpi_config: MPIConfig) -> Self {
        Self {
            mpi_config,
            phantom: PhantomData,
        }
    }

    /// Prior to GKR, we need to do the following:
    /// 1. Parse the commitment from the proof reader and use that to initialize the transcript.
    /// 2. (Optionally) grinding.
    /// 3. Fill the circuit with random coefficients.
    #[inline(always)]
    pub(crate) fn pre_gkr(
        &self,
        mut proof_reader: impl Read,
        circuit: &mut Circuit<Cfg::FieldConfig>,
        transcript: &mut Cfg::TranscriptConfig,
        proving_time_mpi_size: usize,
    ) -> Option<<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::Commitment> {
        let timer = Timer::new("pre_gkr", true);
        let commitment =
            <<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::Commitment as ExpSerde>::deserialize_from(
                &mut proof_reader,
            )
            .ok()?;
        let mut buffer = vec![];
        commitment.serialize_into(&mut buffer).ok()?;

        transcript.append_commitment(&buffer);

        #[cfg(feature = "grinding")]
        grind::<Cfg>(transcript, &self.mpi_config);

        circuit.fill_rnd_coefs(transcript);
        transcript_verifier_sync(transcript, proving_time_mpi_size);

        timer.stop();

        Some(commitment)
    }

    /// Main body of the GKR verification.
    /// We have two schemes:
    /// 1. Vanilla GKR
    /// 2. GKR square: This is a dedicated scheme for the circuit that only contains pow gates.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub(crate) fn gkr(
        &self,
        circuit: &Circuit<Cfg::FieldConfig>,
        public_input: &[<Cfg::FieldConfig as FieldEngine>::SimdCircuitField],
        claimed_v: &<Cfg::FieldConfig as FieldEngine>::ChallengeField,
        proving_time_mpi_size: usize,
        transcript: &mut Cfg::TranscriptConfig,
        mut proof_reader: impl Read,
    ) -> (
        bool,
        ExpanderSingleVarChallenge<Cfg::FieldConfig>,
        Option<ExpanderSingleVarChallenge<Cfg::FieldConfig>>,
        <Cfg::FieldConfig as FieldEngine>::ChallengeField,
        Option<<Cfg::FieldConfig as FieldEngine>::ChallengeField>,
    ) {
        let timer = Timer::new("gkr", true);
        let (verified, challenge_x, challenge_y, claim_x, claim_y) = match Cfg::SCHEME {
            GKRScheme::Vanilla => {
                let (gkr_verified, challenge, claim_x, claim_y) = gkr_verify(
                    proving_time_mpi_size,
                    circuit,
                    public_input,
                    claimed_v,
                    transcript,
                    &mut proof_reader,
                );

                (
                    gkr_verified,
                    challenge.challenge_x(),
                    challenge.challenge_y(),
                    claim_x,
                    claim_y,
                )
            }
            GKRScheme::GkrSquare => {
                let (gkr_verified, challenge_x, claim_x) = gkr_square_verify(
                    proving_time_mpi_size,
                    circuit,
                    public_input,
                    claimed_v,
                    transcript,
                    &mut proof_reader,
                );

                (gkr_verified, challenge_x, None, claim_x, None)
            }
        };
        transcript_verifier_sync(transcript, proving_time_mpi_size);

        log::info!("GKR verification: {verified}");

        timer.stop();
        (verified, challenge_x, challenge_y, claim_x, claim_y)
    }

    /// Parallel version of the GKR verification.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub(crate) fn gkr_parallel(
        &self,
        circuit: &Circuit<Cfg::FieldConfig>,
        public_input: &[<Cfg::FieldConfig as FieldEngine>::SimdCircuitField],
        claimed_v: &<Cfg::FieldConfig as FieldEngine>::ChallengeField,
        proving_time_mpi_size: usize,
        transcript: &mut Cfg::TranscriptConfig,
        mut proof_reader: impl Read,
    ) -> (
        bool,
        ExpanderSingleVarChallenge<Cfg::FieldConfig>,
        Option<ExpanderSingleVarChallenge<Cfg::FieldConfig>>,
        <Cfg::FieldConfig as FieldEngine>::ChallengeField,
        Option<<Cfg::FieldConfig as FieldEngine>::ChallengeField>,
    ) {
        let parse_proof_timer = Timer::new("parse_proof", true);
        let xy_var_degree = match Cfg::SCHEME {
            GKRScheme::Vanilla => SUMCHECK_GKR_DEGREE,
            GKRScheme::GkrSquare => SUMCHECK_GKR_SQUARE_DEGREE,
        };

        let (mut verification_units, challenge, claim_x, claim_y, parse_ok) = parse_proof(
            &mut proof_reader,
            circuit,
            proving_time_mpi_size,
            xy_var_degree,
            *claimed_v,
            transcript,
        );
        parse_proof_timer.stop();

        let gkr_parallel_timer = Timer::new("gkr_parallel", true);
        let sp = VerifierScratchPad::<Cfg::FieldConfig>::new(circuit, proving_time_mpi_size);
        let (verified, challenge_x, challenge_y, claim_x, claim_y) = match Cfg::SCHEME {
            GKRScheme::Vanilla => {
                let gkr_verified = verification_units
                    .par_iter_mut()
                    .zip(circuit.layers.par_iter())
                    .map(|(verification_unit, layer)| {
                        let mut challenge = verification_unit.claim.challenge.clone();
                        let alpha = verification_unit.claim.alpha;
                        let mut claim_x = verification_unit.claim.claim_x;
                        let mut claim_y = verification_unit.claim.claim_y;

                        let mut sp = sp.clone();
                        sumcheck_verify_gkr_layer(
                            proving_time_mpi_size,
                            layer,
                            public_input,
                            &mut challenge,
                            &mut claim_x,
                            &mut claim_y,
                            alpha,
                            &mut Cursor::new(verification_unit.proof.clone()),
                            &mut verification_unit.random_tape,
                            &mut sp,
                            false,
                            true,
                        )
                    })
                    .all(|verified| verified);

                (
                    gkr_verified,
                    challenge.challenge_x(),
                    challenge.challenge_y(),
                    claim_x,
                    claim_y,
                )
            }
            GKRScheme::GkrSquare => {
                assert!(challenge.challenge_y().is_none());
                assert!(claim_y.is_none());

                let gkr_verified = verification_units
                    .par_iter_mut()
                    .zip(circuit.layers.par_iter())
                    .map(|(verification_unit, layer)| {
                        let mut claim_x = verification_unit.claim.claim_x;
                        let mut challenge_x = verification_unit.claim.challenge.challenge_x();

                        let mut sp = sp.clone();
                        sumcheck_verify_gkr_square_layer(
                            proving_time_mpi_size,
                            layer,
                            public_input,
                            &mut challenge_x,
                            &mut claim_x,
                            &mut Cursor::new(verification_unit.proof.clone()),
                            &mut verification_unit.random_tape,
                            &mut sp,
                            false,
                            true,
                        )
                    })
                    .all(|verified| verified);

                (gkr_verified, challenge.challenge_x(), None, claim_x, None)
            }
        };
        gkr_parallel_timer.stop();
        transcript_verifier_sync(transcript, proving_time_mpi_size);

        (
            verified && parse_ok,
            challenge_x,
            challenge_y,
            claim_x,
            claim_y,
        )
    }

    /// Verify the PCS opening against the commitment and the claim from GKR.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub(crate) fn post_gkr(
        &self,
        pcs_params: &<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::Params,
        pcs_verification_key: &<<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::SRS as StructuredReferenceString>::VKey,
        commitment: &<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::Commitment,
        challenge_x: &mut ExpanderSingleVarChallenge<Cfg::FieldConfig>,
        claim_x: &<Cfg::FieldConfig as FieldEngine>::ChallengeField,
        challenge_y: &mut Option<ExpanderSingleVarChallenge<Cfg::FieldConfig>>,
        claim_y: &Option<<Cfg::FieldConfig as FieldEngine>::ChallengeField>,
        transcript: &mut impl Transcript,
        mut proof_reader: impl Read,
    ) -> bool {
        let timer = Timer::new("post_gkr", true);
        let mut verified = self.get_pcs_opening_from_proof_and_verify(
            pcs_params,
            pcs_verification_key,
            commitment,
            challenge_x,
            claim_x,
            transcript,
            &mut proof_reader,
        );

        if let (Some(challenge_y), Some(claim_y)) = (challenge_y, claim_y) {
            verified &= self.get_pcs_opening_from_proof_and_verify(
                pcs_params,
                pcs_verification_key,
                commitment,
                challenge_y,
                claim_y,
                transcript,
                &mut proof_reader,
            );
        }

        timer.stop();
        verified
    }

    /// Paritially verify the proof.
    /// Conduct the whole procedure except for pairing, if any.
    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &self,
        circuit: &mut Circuit<Cfg::FieldConfig>,
        public_input: &[<Cfg::FieldConfig as FieldEngine>::SimdCircuitField],
        claimed_v: &<Cfg::FieldConfig as FieldEngine>::ChallengeField,
        pcs_params: &<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::Params,
        pcs_verification_key: &<<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::SRS as StructuredReferenceString>::VKey,
        proof: &Proof,
    ) -> bool {
        let timer = Timer::new("snark verify", true);

        let proving_time_mpi_size = self.mpi_config.world_size();
        let mut transcript = Cfg::TranscriptConfig::new();
        let mut cursor = Cursor::new(&proof.bytes);

        let commitment =
            match self.pre_gkr(&mut cursor, circuit, &mut transcript, proving_time_mpi_size) {
                Some(c) => c,
                None => return false,
            };

        let (mut verified, mut challenge_x, mut challenge_y, claim_x, claim_y) = self.gkr(
            circuit,
            public_input,
            claimed_v,
            proving_time_mpi_size,
            &mut transcript,
            &mut cursor,
        );

        verified &= self.post_gkr(
            pcs_params,
            pcs_verification_key,
            &commitment,
            &mut challenge_x,
            &claim_x,
            &mut challenge_y,
            &claim_y,
            &mut transcript,
            &mut cursor,
        );

        timer.stop();

        verified
    }

    pub fn par_verify(
        &self,
        circuit: &mut Circuit<Cfg::FieldConfig>,
        public_input: &[<Cfg::FieldConfig as FieldEngine>::SimdCircuitField],
        claimed_v: &<Cfg::FieldConfig as FieldEngine>::ChallengeField,
        pcs_params: &<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::Params,
        pcs_verification_key: &<<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::SRS as StructuredReferenceString>::VKey,
        proof: &Proof,
    ) -> bool {
        let timer = Timer::new("snark verify", true);

        let proving_time_mpi_size = self.mpi_config.world_size();
        let mut transcript = Cfg::TranscriptConfig::new();
        let mut cursor = Cursor::new(&proof.bytes);

        let commitment =
            match self.pre_gkr(&mut cursor, circuit, &mut transcript, proving_time_mpi_size) {
                Some(c) => c,
                None => return false,
            };

        let (mut verified, mut challenge_x, mut challenge_y, claim_x, claim_y) = self.gkr_parallel(
            circuit,
            public_input,
            claimed_v,
            proving_time_mpi_size,
            &mut transcript,
            &mut cursor,
        );

        verified &= self.post_gkr(
            pcs_params,
            pcs_verification_key,
            &commitment,
            &mut challenge_x,
            &claim_x,
            &mut challenge_y,
            &claim_y,
            &mut transcript,
            &mut cursor,
        );

        timer.stop();
        verified
    }

    #[inline(always)]
    pub(crate) fn pre_gkr_ref(
        &self,
        mut proof_reader: impl Read,
        circuit: &Circuit<Cfg::FieldConfig>,
        transcript: &mut Cfg::TranscriptConfig,
        proving_time_mpi_size: usize,
    ) -> Option<(
        <Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::Commitment,
        RndCoefMap<<Cfg::FieldConfig as FieldEngine>::CircuitField>,
    )> {
        let timer = Timer::new("pre_gkr_ref", true);
        let commitment =
            <<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::Commitment as ExpSerde>::deserialize_from(
                &mut proof_reader,
            )
            .ok()?;
        let mut buffer = vec![];
        commitment.serialize_into(&mut buffer).ok()?;

        transcript.append_commitment(&buffer);

        #[cfg(feature = "grinding")]
        grind::<Cfg>(transcript, &self.mpi_config);

        let rnd_map = circuit.sample_rnd_coef_map(transcript);
        transcript_verifier_sync(transcript, proving_time_mpi_size);

        timer.stop();

        Some((commitment, rnd_map))
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub(crate) fn gkr_ref(
        &self,
        circuit: &Circuit<Cfg::FieldConfig>,
        public_input: &[<Cfg::FieldConfig as FieldEngine>::SimdCircuitField],
        claimed_v: &<Cfg::FieldConfig as FieldEngine>::ChallengeField,
        proving_time_mpi_size: usize,
        transcript: &mut Cfg::TranscriptConfig,
        mut proof_reader: impl Read,
        rnd_map: &RndCoefMap<<Cfg::FieldConfig as FieldEngine>::CircuitField>,
    ) -> (
        bool,
        ExpanderSingleVarChallenge<Cfg::FieldConfig>,
        Option<ExpanderSingleVarChallenge<Cfg::FieldConfig>>,
        <Cfg::FieldConfig as FieldEngine>::ChallengeField,
        Option<<Cfg::FieldConfig as FieldEngine>::ChallengeField>,
    ) {
        let timer = Timer::new("gkr_ref", true);
        let (verified, challenge_x, challenge_y, claim_x, claim_y) = match Cfg::SCHEME {
            GKRScheme::Vanilla => {
                let (gkr_verified, challenge, claim_x, claim_y) = gkr_verify_ref(
                    proving_time_mpi_size,
                    circuit,
                    public_input,
                    claimed_v,
                    transcript,
                    &mut proof_reader,
                    rnd_map,
                );

                (
                    gkr_verified,
                    challenge.challenge_x(),
                    challenge.challenge_y(),
                    claim_x,
                    claim_y,
                )
            }
            GKRScheme::GkrSquare => {
                let (gkr_verified, challenge_x, claim_x) = gkr_square_verify(
                    proving_time_mpi_size,
                    circuit,
                    public_input,
                    claimed_v,
                    transcript,
                    &mut proof_reader,
                );

                (gkr_verified, challenge_x, None, claim_x, None)
            }
        };
        transcript_verifier_sync(transcript, proving_time_mpi_size);

        log::info!("GKR verification (ref): {verified}");

        timer.stop();
        (verified, challenge_x, challenge_y, claim_x, claim_y)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify_ref(
        &self,
        circuit: &Circuit<Cfg::FieldConfig>,
        public_input: &[<Cfg::FieldConfig as FieldEngine>::SimdCircuitField],
        claimed_v: &<Cfg::FieldConfig as FieldEngine>::ChallengeField,
        pcs_params: &<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::Params,
        pcs_verification_key: &<<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::SRS as StructuredReferenceString>::VKey,
        proof: &Proof,
    ) -> bool {
        let timer = Timer::new("snark verify_ref", true);

        let proving_time_mpi_size = self.mpi_config.world_size();
        let mut transcript = Cfg::TranscriptConfig::new();
        let mut cursor = Cursor::new(&proof.bytes);

        let (commitment, rnd_map) =
            match self.pre_gkr_ref(&mut cursor, circuit, &mut transcript, proving_time_mpi_size) {
                Some(v) => v,
                None => return false,
            };

        let (mut verified, mut challenge_x, mut challenge_y, claim_x, claim_y) = self.gkr_ref(
            circuit,
            public_input,
            claimed_v,
            proving_time_mpi_size,
            &mut transcript,
            &mut cursor,
            &rnd_map,
        );

        verified &= self.post_gkr(
            pcs_params,
            pcs_verification_key,
            &commitment,
            &mut challenge_x,
            &claim_x,
            &mut challenge_y,
            &claim_y,
            &mut transcript,
            &mut cursor,
        );

        timer.stop();

        verified
    }
}

impl<Cfg: GKREngine> Verifier<Cfg> {
    #[allow(clippy::too_many_arguments)]
    fn get_pcs_opening_from_proof_and_verify(
        &self,
        pcs_params: &<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::Params,
        pcs_verification_key: &<<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::SRS as StructuredReferenceString>::VKey,
        commitment: &<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::Commitment,
        open_at: &mut ExpanderSingleVarChallenge<Cfg::FieldConfig>,
        v: &<Cfg::FieldConfig as FieldEngine>::ChallengeField,
        transcript: &mut impl Transcript,
        proof_reader: impl Read,
    ) -> bool {
        let opening =
            match <Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::Opening::deserialize_from(
                proof_reader,
            ) {
                Ok(o) => o,
                Err(_) => return false,
            };

        transcript.lock_proof();
        let verified = Cfg::PCSConfig::verify(
            pcs_params,
            pcs_verification_key,
            commitment,
            open_at,
            *v,
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
}
