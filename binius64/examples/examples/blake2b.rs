// Copyright 2025 Irreducible Inc.
use anyhow::Result;
use binius_examples::{Cli, circuits::blake2b::Blake2bExample};

fn main() -> Result<()> {
    Cli::<Blake2bExample>::new("blake2b")
        .about("Blake2b hash function circuit example")
        .run()
}
