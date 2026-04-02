// Copyright 2024-2025 Irreducible Inc.

//! Implementation of the Fast Reed–Solomon IOPP (FRI) over binary fields.
//!
//! FRI is an IOP of Proximity for Reed–Solomon codes. The original protocol was introduced in
//! [BBHR17], and this implementation uses a special instantiation described in [DP24] Section 3.
//!
//! This protocol implement FRI for an interleaved Reed–Solomon code, rather than a regular
//! Reed–Solomon code. Codewords in an interleaved code have the form of being a batch of
//! Reed–Solomon codewords, interleaved element-wise. For example, an interleaved codeword with a
//! batch size of 4 would have the form `a0, b0, c0, d0, a1, b1, c1, d1, ...`, where `a0, a1, ...`
//! is a Reed–Solomon codeword, `b0, b1, ...` is another Reed–Solomon codeword, and so on. The
//! batch size of the interleaved code is required to be a power of 2.
//!
//! The folding phase begins with the verifier having oracle access to an initial, purported
//! interleaved codeword. In each round the prover receives a challenge and folds the interleaved
//! codeword in half until it reaches a single codeword, mixing adjacent codewords as a linear
//! interpolation. The prover sends the folded codeword to the oracle. Then in each subsequent
//! round, the prover receives a challenge and folds the codeword in half using the FRI folding
//! procedure and may or may not send a new oracle to the verifier. The last oracle the prover
//! sends, they send entirely in the clear to the verifier, rather than sending with oracle access.
//!
//! [BBHR17]: <https://eccc.weizmann.ac.il/report/2017/134/>
//! [DP24]: <https://eprint.iacr.org/2024/504>

mod common;
mod error;
pub mod fold;
mod size_estimation;
pub mod verify;

pub use common::*;
pub use error::*;
pub use fold::FRIFoldVerifier;
pub use size_estimation::proof_size;
