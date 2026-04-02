// Copyright 2025 Irreducible Inc.

use anyhow::bail;
use binius_circuits::bitcoin::header_chain::HeaderChain;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller, util::pack_bytes_into_wires_le};
use clap::Args;

use crate::ExampleCircuit;

/// Check some basic things about a Bitcoin header chain.
///
/// **IMPORTANT**: This does **NOT** prove full validity of the header chain.
/// In particular it's not checked _at all_ that the target difficulty is calculated correctly.
/// One could easily satisfy this circuit by quickly self-mining many blocks with low difficulty.
pub struct BitcoinHeaderChainExample {
    latest_digest: [Wire; 4],
    headers: Vec<[Wire; 10]>,
    header_chain_gadget: HeaderChain,
}

#[derive(Args)]
pub struct Params {
    /// Number of block headers to prove.
    #[arg(long, default_value_t = 10)]
    pub num_blocks: usize,
}

#[derive(Args)]
pub struct Instance {
    /// Height of the newest block in the proved header chain.
    /// Defaults to the currently latest block in the network.
    #[arg(long)]
    pub to_block: Option<usize>,
}

impl ExampleCircuit for BitcoinHeaderChainExample {
    type Params = Params;
    type Instance = Instance;

    fn build(params: Self::Params, builder: &mut CircuitBuilder) -> anyhow::Result<Self> {
        if params.num_blocks == 0 {
            bail!("need to prove at least one block")
        }

        let latest_digest: [Wire; 4] = std::array::from_fn(|_| builder.add_witness());
        let headers: Vec<[Wire; 10]> =
            std::iter::repeat_with(|| std::array::from_fn(|_| builder.add_witness()))
                .take(params.num_blocks)
                .collect();
        let header_chain_gadget = HeaderChain::construct_circuit(builder, &headers, latest_digest);

        Ok(Self {
            latest_digest,
            headers,
            header_chain_gadget,
        })
    }

    fn populate_witness(
        &self,
        instance: Self::Instance,
        filler: &mut WitnessFiller,
    ) -> anyhow::Result<()> {
        let (headers_value, latest_digest_value) =
            pull_headers(self.headers.len(), instance.to_block)?;
        for (header, header_value) in self.headers.iter().zip(&headers_value) {
            pack_bytes_into_wires_le(filler, header, header_value);
        }
        pack_bytes_into_wires_le(filler, &self.latest_digest, &latest_digest_value);
        let headers_value_ref: Vec<&[u8]> = headers_value.iter().map(AsRef::as_ref).collect();
        self.header_chain_gadget
            .populate_inner(filler, &headers_value_ref);

        Ok(())
    }

    fn param_summary(params: &Self::Params) -> Option<String> {
        Some(format!("{}blk", params.num_blocks))
    }
}

fn pull_headers(
    num_blocks: usize,
    to_block: Option<usize>,
) -> anyhow::Result<(Vec<Vec<u8>>, Vec<u8>)> {
    let last_height: usize = match to_block {
        Some(to_block) => to_block,
        None => ureq::get("https://mempool.space/api/blocks/tip/height")
            .call()?
            .body_mut()
            .read_to_string()?
            .trim()
            .parse()?,
    };

    let latest_digest = ureq::get(format!(
        "https://mempool.space/api/block-height/{last_height}"
    ))
    .call()?
    .body_mut()
    .read_to_string()?;
    let mut latest_digest = hex::decode(latest_digest)?;
    latest_digest.reverse();

    let first_height = (last_height + 1).checked_sub(num_blocks).unwrap();
    println!(
        "Fetching {num_blocks} block headers. First height is {first_height} and last height is {last_height} .."
    );
    let mut headers = Vec::new();
    for height in (first_height..=last_height).rev() {
        let hash = ureq::get(format!("https://mempool.space/api/block-height/{height}"))
            .call()?
            .body_mut()
            .read_to_string()?;
        println!("Fetching header with height {height} and hash {hash} ..");
        let header = ureq::get(format!("https://mempool.space/api/block/{hash}/header"))
            .call()?
            .body_mut()
            .read_to_string()?;
        let mut hash = hex::decode(hash)?;
        hash.reverse();
        let header = hex::decode(header)?;
        headers.push(header);
    }

    println!("Done.");

    Ok((headers, latest_digest))
}
