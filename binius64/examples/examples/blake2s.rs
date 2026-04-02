// Copyright 2025 Irreducible Inc.
use anyhow::Result;
use binius_examples::{Cli, circuits::blake2s::Blake2sExample};

fn main() -> Result<()> {
    Cli::<Blake2sExample>::new("blake2s")
        .about("Blake2s hash function circuit example")
        .run()
}
