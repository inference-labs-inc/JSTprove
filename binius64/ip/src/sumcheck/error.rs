// Copyright 2025 Irreducible Inc.

use binius_transcript::Error as TranscriptError;

use crate::channel;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("transcript error: {0}")]
    Transcript(#[source] TranscriptError),
    #[error("verification error: {0}")]
    Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
    #[error("transcript is empty")]
    TranscriptIsEmpty,
    #[error("invalid assertion: value is not zero")]
    InvalidAssert,
}

impl From<TranscriptError> for Error {
    fn from(err: TranscriptError) -> Self {
        match err {
            TranscriptError::NotEnoughBytes => VerificationError::TranscriptIsEmpty.into(),
            _ => Error::Transcript(err),
        }
    }
}

impl From<channel::Error> for Error {
    fn from(err: channel::Error) -> Self {
        match err {
            channel::Error::ProofEmpty => VerificationError::TranscriptIsEmpty.into(),
            channel::Error::InvalidAssert => VerificationError::InvalidAssert.into(),
        }
    }
}
