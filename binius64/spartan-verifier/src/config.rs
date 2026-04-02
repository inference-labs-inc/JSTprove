// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField1b, BinaryField128bGhash};
pub use binius_verifier::{
    config::StdChallenger,
    hash::{StdCompression, StdDigest},
};

pub type B1 = BinaryField1b;
pub type B128 = BinaryField128bGhash;
