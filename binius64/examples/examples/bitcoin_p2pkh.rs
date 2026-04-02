// Copyright 2025 Irreducible Inc.

use anyhow::Result;
use binius_examples::{Cli, circuits::bitcoin_p2pkh::BitcoinP2PKHExample};

fn main() -> Result<()> {
    Cli::<BitcoinP2PKHExample>::new("bitcoin_p2pkh")
		.about("Bitcoin P2PKH address validation example - proves knowledge of private key without revealing it")
		.run()
}
