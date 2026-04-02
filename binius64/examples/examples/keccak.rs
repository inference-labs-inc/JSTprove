// Copyright 2025 Irreducible Inc.
use anyhow::Result;
use binius_examples::{Cli, circuits::keccak::KeccakExample};

fn main() -> Result<()> {
    Cli::<KeccakExample>::new("keccak")
        .about("Keccak-256 hash function circuit example")
        .run()
}
