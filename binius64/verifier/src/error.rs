// Copyright 2025 Irreducible Inc.

use binius_core::ConstraintSystemError;
use binius_iop::channel::Error as IOPChannelError;
use binius_ip::channel::Error as ChannelError;

use crate::{
    fri,
    protocols::{intmul, shift, sumcheck},
    ring_switch,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("transcript error: {0}")]
    Transcript(#[from] binius_transcript::Error),
    #[error("channel error: {0}")]
    Channel(#[from] ChannelError),
    #[error("IOP channel error: {0}")]
    IOPChannel(#[from] IOPChannelError),
    #[error("FRI error: {0}")]
    FRI(#[from] fri::Error),
    #[error("ring switch error: {0}")]
    RingSwitch(#[from] ring_switch::Error),
    #[error("IntMul error: {0}")]
    IntMul(#[from] intmul::Error),
    #[error("sumcheck error: {0}")]
    Sumcheck(#[from] sumcheck::Error),
    #[error("incorrect public inputs length: expected {expected}, got {actual}")]
    IncorrectPublicInputLength { expected: usize, actual: usize },
    #[error("constraint system error: {0}")]
    ConstraintSystem(#[from] ConstraintSystemError),
    #[error("invalid proof: {0}")]
    Verification(#[from] VerificationError),
    #[error("shift reduction error: {0}")]
    ShiftReduction(#[from] shift::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
    #[error("public input check failed")]
    PublicInputCheckFailed,
    #[error("final evaluation check of sumcheck and FRI reductions failed")]
    EvaluationInconsistency,
}
