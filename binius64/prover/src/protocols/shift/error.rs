// Copyright 2025 Irreducible Inc.

use crate::protocols::sumcheck::Error as SumcheckError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("sumcheck error: {0}")]
    SumcheckError(#[from] SumcheckError),
}
