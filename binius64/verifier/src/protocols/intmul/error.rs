// Copyright 2025 Irreducible Inc.

use binius_ip::channel::Error as ChannelError;

use crate::protocols::{prodcheck::Error as ProdcheckError, sumcheck::Error as SumcheckError};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("transcript error")]
    Transcript(#[from] binius_transcript::Error),
    #[error("channel error")]
    Channel(#[from] ChannelError),
    #[error("sumcheck verify error")]
    SumcheckVerify(#[from] SumcheckError),
    #[error("prodcheck verify error")]
    ProdcheckVerify(#[from] ProdcheckError),
}
