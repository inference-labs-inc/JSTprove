// Copyright 2025 Irreducible Inc.

#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Error indicating invalid arguments were passed to a sumcheck function.
    #[error("invalid arguments: {0}")]
    ArgumentError(String),
    #[error("multilinears do not have equal number of variables")]
    MultilinearSizeMismatch,
    #[error("bitmasks slice length does not match the evaluation point length")]
    BitmasksSizeMismatch,
    #[error("number of eval claims does not match the number of multilinears")]
    EvalClaimsNumberMismatch,
    #[error("the length of evaluation point does not match the size of the multilinears")]
    EvalPointLengthMismatch,
    #[error("batched provers should have the same number of rounds")]
    ProverRoundCountMismatch,
    #[error("expected execute() call")]
    ExpectedExecute,
    #[error("expected fold() call")]
    ExpectedFold,
    #[error("expected finish() call")]
    ExpectedFinish,
}
