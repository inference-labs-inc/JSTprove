// Copyright 2025 Irreducible Inc.

use anyhow::Result;
use binius_examples::{Cli, circuits::bitcoin_header_chain::BitcoinHeaderChainExample};

fn main() -> Result<()> {
    // disable `ureq` logging
    // unfortunately this also changes the binius logging/tracing slightly
    unsafe {
        std::env::set_var("RUST_LOG", "binius=trace,ureq=off");
    }

    Cli::<BitcoinHeaderChainExample>::new("bitcoin_headers")
        .about("Bitcoin Header Chain Example")
        .run()
}
