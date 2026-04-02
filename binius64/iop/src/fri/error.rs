// Copyright 2024-2025 Irreducible Inc.

use crate::merkle_tree;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("fold arities total exceeds the number of fold rounds")]
    InvalidFoldAritySequence,
    #[error("conflicting or incorrect constructor argument: {0}")]
    InvalidArgs(String),
    #[error("cannot calculate parameters satisfying the security target")]
    ParameterError,
    #[error("Merkle tree error: {0}")]
    MerkleError(merkle_tree::Error),
    #[error("Reed-Solomon encoding error: {0}")]
    Verification(#[from] VerificationError),
    #[error("transcript error: {0}")]
    TranscriptError(#[from] binius_transcript::Error),
}

impl From<merkle_tree::Error> for Error {
    fn from(err: merkle_tree::Error) -> Self {
        match err {
            merkle_tree::Error::Verification(err) => Self::Verification(err.into()),
            _ => Self::MerkleError(err),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
    #[error("incorrect codeword folding in query round {query_round} at index {index}")]
    IncorrectFold { query_round: usize, index: usize },
    #[error("the size of the query proof is incorrect, expected {expected}")]
    IncorrectQueryProofLength { expected: usize },
    #[error(
        "the number of values in round {round} of the query proof is incorrect, expected {coset_size}"
    )]
    IncorrectQueryProofValuesLength { round: usize, coset_size: usize },
    #[error("The dimension-1 codeword must contain the same values")]
    IncorrectDegree,
    #[error("Merkle tree error: {0}")]
    MerkleError(#[from] merkle_tree::VerificationError),
}
