// Copyright 2025 Irreducible Inc.

#![warn(rustdoc::missing_crate_level_docs)]

//! Objects used to read and write proof strings.
//!
//! A Binius proof consists of the transcript of the simulated interaction between the prover and
//! the verifier. Using the Fiat-Shamir heuristic, the prover and verifier can simulate the
//! verifier's messages, which are deterministically computed based on the sequence of prover
//! messages and calls to sample verifier challenges. The interaction consists of two parallel
//! tapes, the _transcript_ tape and the _advice_ tape. The values in the transcript tape affect
//! the Fiat-Shamir state, whereas values in the advice tape do not. **The decommitment tape must
//! only be used for values that were previously committed to in the transcript tape.** For
//! example, it is secure to write a Merkle tree root to the transcript tape, sample a random
//! index, then provide the Merkle leaf opening at that index in the advice tape.

mod error;
pub mod fiat_shamir;
mod transcript;

pub use bytes::{Buf, BufMut};
pub use error::*;
pub use transcript::*;
