use anyhow::Result;
use binius_verifier::{config::StdChallenger, transcript::VerifierTranscript};

use crate::prove::{ProofArtifact, StdVerifier};

pub fn verify(verifier: &StdVerifier, artifact: &ProofArtifact) -> Result<()> {
    let challenger = StdChallenger::default();
    let mut verifier_transcript = VerifierTranscript::new(challenger, artifact.proof_bytes.clone());
    verifier.verify(&artifact.public_words, &mut verifier_transcript)?;
    verifier_transcript.finalize()?;
    Ok(())
}
