// Copyright 2025 Irreducible Inc.
//! Specifies standard trait implementations and parameters.

use binius_field::{AESTowerField8b, BinaryField, BinaryField1b, BinaryField128bGhash};
use binius_hash::StdDigest;
use binius_transcript::fiat_shamir::{Challenger, HasherChallenger};
use binius_utils::checked_arithmetics::{checked_int_div, checked_log_2};

// Exports the binary fields that this system uses
pub type B1 = BinaryField1b;
pub type B128 = BinaryField128bGhash;

/// The intention of this trait is to capture the moment when a StandardChallenger type is changed.
pub trait ChallengerWithName: Challenger {
    const NAME: &'static str;
}

impl ChallengerWithName for HasherChallenger<StdDigest> {
    const NAME: &'static str = "HasherChallenger<Sha256>";
}

/// The default [`binius_transcript::fiat_shamir::Challenger`] implementation.
pub type StdChallenger = HasherChallenger<StdDigest>;

/// The protocol proves constraint systems over 64-bit words.
pub const WORD_SIZE_BITS: usize = 64;
pub const WORD_SIZE_BYTES: usize = checked_int_div(WORD_SIZE_BITS, 8);

/// log2 of [`WORD_SIZE_BITS`].
pub const LOG_WORD_SIZE_BITS: usize = checked_log_2(WORD_SIZE_BITS);
pub const LOG_WORDS_PER_ELEM: usize = checked_log_2(B128::N_BITS) - LOG_WORD_SIZE_BITS;

pub const PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES: [AESTowerField8b; 3] = [
    AESTowerField8b::new(0x2),
    AESTowerField8b::new(0x4),
    AESTowerField8b::new(0x10),
];
