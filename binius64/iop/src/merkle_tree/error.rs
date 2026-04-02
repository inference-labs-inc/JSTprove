// Copyright 2024-2025 Irreducible Inc.

use binius_utils::SerializationError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Length of the input vector is incorrect, expected {expected}")]
    IncorrectVectorLen { expected: usize },
    #[error("Index exceeds Merkle tree base size: {max}")]
    IndexOutOfRange { max: usize },
    #[error("values length must be a multiple of the batch size")]
    IncorrectBatchSize,
    #[error("The argument length must be a power of two.")]
    PowerOfTwoLengthRequired,
    #[error("Failed to serialize leaf element: {0}")]
    Serialization(SerializationError),
    #[error("The layer does not exist in the Merkle tree")]
    IncorrectLayerDepth,
    #[error("transcript error: {0}")]
    Transcript(#[from] binius_transcript::Error),
    #[error("verification failure: {0}")]
    Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
    #[error("the length of the vector does not match the committed length")]
    IncorrectVectorLength,
    #[error("the shape of the proof is incorrect")]
    IncorrectProofShape,
    #[error("the proof is invalid")]
    InvalidProof,
}
