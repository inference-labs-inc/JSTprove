// Copyright 2025 Irreducible Inc.
use anyhow::Result;
use binius_examples::{Cli, circuits::hashsign::HashBasedSigExample};

fn main() -> Result<()> {
    Cli::<HashBasedSigExample>::new("hashsign")
        .about("Hash-based multi-signature (XMSS) verification example")
        .run()
}
