// Copyright 2025 Irreducible Inc.

use crate::{
    fri,
    protocols::{basefold, intmul, shift, sumcheck},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid argument {arg}: {msg}")]
    ArgumentError { arg: String, msg: String },
    #[error("sumcheck error: {0}")]
    Sumcheck(#[from] sumcheck::Error),
    #[error("FRI error: {0}")]
    Fri(#[from] fri::Error),
    #[error("basefold error: {0}")]
    Basefold(#[from] basefold::Error),
    #[error("transcript error: {0}")]
    Transcript(#[from] binius_transcript::Error),
    #[error("integer multiplication error: {0}")]
    IntMul(#[from] intmul::Error),
    #[error("shift reduction error: {0}")]
    ShiftReduction(#[from] shift::Error),
}
