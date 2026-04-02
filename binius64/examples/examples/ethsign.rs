// Copyright 2025 Irreducible Inc.
use anyhow::Result;
use binius_examples::{Cli, circuits::ethsign::EthSignExample};

fn main() -> Result<()> {
    Cli::<EthSignExample>::new("ethsign")
        .about("Ethereum-style signing example")
        .run()
}
