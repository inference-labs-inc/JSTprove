// Copyright 2025 Irreducible Inc.
use anyhow::Result;
use binius_examples::{Cli, circuits::zklogin::ZkLoginExample};

fn main() -> Result<()> {
    Cli::<ZkLoginExample>::new("zklogin")
        .about("Circuit verifying knowledge of a valid OpenID Connect login")
        .run()
}
