// Copyright 2025 Irreducible Inc.

use binius_ip::channel::Error as ChannelError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("transcript error")]
    Transcript(#[from] binius_transcript::Error),
    #[error("channel error")]
    Channel(#[from] ChannelError),
    #[error("sumcheck error")]
    Sumcheck(#[from] crate::protocols::sumcheck::Error),
    #[error("verification failure")]
    VerificationFailure,
}
