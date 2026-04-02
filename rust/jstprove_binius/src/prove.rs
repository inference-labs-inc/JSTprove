use anyhow::Result;
use binius_core::constraint_system::ValueVec;
use binius_prover::{OptimalPackedB128, Prover};
use binius_transcript::ProverTranscript;
use binius_verifier::{
    config::StdChallenger,
    hash::{StdCompression, StdDigest},
    Verifier,
};

use crate::circuit::BiniusCircuit;

pub type StdVerifier = Verifier<StdDigest, StdCompression>;
pub type StdProver = Prover<
    OptimalPackedB128,
    binius_prover::hash::parallel_compression::ParallelCompressionAdaptor<StdCompression>,
    StdDigest,
>;

pub struct ProofArtifact {
    pub proof_bytes: Vec<u8>,
    pub public_words: Vec<binius_core::word::Word>,
}

pub fn setup(bc: &BiniusCircuit, log_inv_rate: usize) -> Result<(StdVerifier, StdProver)> {
    let cs = bc.circuit.constraint_system().clone();
    let compression = binius_prover::hash::parallel_compression::ParallelCompressionAdaptor::new(
        StdCompression::default(),
    );
    let verifier = Verifier::<StdDigest, _>::setup(cs, log_inv_rate, StdCompression::default())?;
    let prover = Prover::<OptimalPackedB128, _, StdDigest>::setup(verifier.clone(), compression)?;
    Ok((verifier, prover))
}

pub fn prove(prover: &StdProver, witness: ValueVec) -> Result<ProofArtifact> {
    let challenger = StdChallenger::default();
    let public_words = witness.public().to_vec();
    let mut prover_transcript = ProverTranscript::new(challenger);
    prover.prove(witness, &mut prover_transcript)?;
    let proof_bytes = prover_transcript.finalize();
    Ok(ProofArtifact {
        proof_bytes,
        public_words,
    })
}
