// Copyright 2025 Irreducible Inc.

use std::array;

use anyhow::Result;
use binius_circuits::{
    bignum::BigUint,
    bitcoin::p2pkh_signature::{addr_bytes_to_le_words, build_p2pkh_circuit},
};
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use bitcoin::{Network, PrivateKey, secp256k1::Secp256k1};
use bitcoin_hashes::Hash;
use clap::Args;
use rand::{RngCore, SeedableRng, rngs::StdRng};

use crate::ExampleCircuit;

/// Example circuit that proves knowledge of a Bitcoin private key corresponding to a P2PKH address.
///
/// This demonstrates a post-quantum secure Bitcoin signature scheme that maintains backwards
/// compatibility with existing Bitcoin addresses. The circuit proves knowledge of a private key
/// without revealing it, using the complete Bitcoin P2PKH address derivation:
///
/// Private Key → scalar_mul → Public Key → compress → Compressed PubKey
/// → SHA256 → Digest → swap_bytes → LE Format → RIPEMD160 → Address
pub struct BitcoinP2PKHExample {
    private_key: BigUint,
    expected_address: [Wire; 5],
}

#[derive(Args, Debug, Clone)]
pub struct Params {
    // No circuit parameters needed for this example
    // The circuit is fixed-size for secp256k1 private keys
}

#[derive(Args, Debug, Clone)]
pub struct Instance {
    /// Private key as hex string (32 bytes = 64 hex chars, without 0x prefix).
    /// If not provided, a random private key will be generated.
    #[arg(long, value_parser = parse_hex_private_key)]
    pub private_key: Option<[u8; 32]>,

    /// Expected Bitcoin P2PKH address hash as hex string (20 bytes = 40 hex chars, without 0x
    /// prefix). If not provided, the address will be computed from the private key.
    #[arg(long, value_parser = parse_hex_address)]
    pub expected_address: Option<[u8; 20]>,

    /// Seed for deterministic random generation (for reproducible results)
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

impl ExampleCircuit for BitcoinP2PKHExample {
    type Params = Params;
    type Instance = Instance;

    fn build(_params: Params, builder: &mut CircuitBuilder) -> Result<Self> {
        // Create witness for private key (4 limbs = 256 bits)
        let private_key = BigUint::new_witness(builder, 4);

        // Create witness wires for expected address (5 × 32-bit words = 160 bits = 20 bytes)
        let expected_address: [Wire; 5] = array::from_fn(|_| builder.add_witness());

        // Build the complete P2PKH circuit that proves knowledge of the private key
        build_p2pkh_circuit(builder, &private_key, expected_address);

        Ok(Self {
            private_key,
            expected_address,
        })
    }

    fn populate_witness(&self, instance: Instance, w: &mut WitnessFiller) -> Result<()> {
        // Generate or use provided private key
        let private_key_bytes = match instance.private_key {
            Some(key) => {
                tracing::info!("Using provided private key");
                key
            }
            None => {
                let mut rng = StdRng::seed_from_u64(instance.seed);
                let mut key = [0u8; 32];

                // Generate a valid secp256k1 private key (1 <= key < group_order)
                loop {
                    rng.fill_bytes(&mut key);

                    // Ensure key is not zero and is less than secp256k1 group order
                    // Group order:
                    // 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
                    if !is_zero(&key) && is_valid_secp256k1_key(&key) {
                        break;
                    }
                }

                tracing::info!(
                    "Generated private key (seed={}): {}",
                    instance.seed,
                    hex::encode(key)
                );
                key
            }
        };

        // Get expected Bitcoin address (hash160). If not provided, compute from the private key
        let expected_address_bytes = match instance.expected_address {
            Some(addr) => {
                tracing::info!("Using provided expected address: {}", hex::encode(addr));
                addr
            }
            None => {
                // Use bitcoin crate to compute the P2PKH address hash from private key
                let secp = Secp256k1::new();
                let private_key = PrivateKey::from_slice(&private_key_bytes, Network::Bitcoin)
                    .map_err(|e| anyhow::anyhow!("Invalid private key: {}", e))?;
                let public_key = private_key.public_key(&secp);
                let addr = public_key.pubkey_hash().to_byte_array();
                tracing::info!(
                    "Computed expected address from private key: {}",
                    hex::encode(addr)
                );
                addr
            }
        };

        // Convert private key bytes to little-endian 64-bit limbs
        let private_key_limbs = bytes_to_le_limbs(&private_key_bytes);

        // Convert address bytes to little-endian 32-bit words
        let address_words = addr_bytes_to_le_words(&expected_address_bytes);

        // Populate witness values
        self.private_key.populate_limbs(w, &private_key_limbs);

        for i in 0..5 {
            w[self.expected_address[i]] =
                binius_core::word::Word::from_u64(address_words[i] as u64);
        }

        tracing::info!("Successfully populated witness for Bitcoin P2PKH proof");
        Ok(())
    }
}

/// Parse hex string to 32-byte private key
fn parse_hex_private_key(s: &str) -> Result<[u8; 32], String> {
    let bytes = hex::decode(s).map_err(|e| format!("Invalid hex: {}", e))?;
    if bytes.len() != 32 {
        return Err(format!(
            "Private key must be exactly 32 bytes, got {}",
            bytes.len()
        ));
    }
    let mut key = [0u8; 32];
    key.copy_from_slice(&bytes);

    if is_zero(&key) || !is_valid_secp256k1_key(&key) {
        return Err("Invalid secp256k1 private key".to_string());
    }

    Ok(key)
}

/// Parse hex string to 20-byte address
fn parse_hex_address(s: &str) -> Result<[u8; 20], String> {
    let bytes = hex::decode(s).map_err(|e| format!("Invalid hex: {}", e))?;
    if bytes.len() != 20 {
        return Err(format!(
            "Address must be exactly 20 bytes, got {}",
            bytes.len()
        ));
    }
    let mut addr = [0u8; 20];
    addr.copy_from_slice(&bytes);
    Ok(addr)
}

/// Check if byte array is all zeros
fn is_zero(bytes: &[u8; 32]) -> bool {
    bytes.iter().all(|&b| b == 0)
}

/// Simplified check for valid secp256k1 private key
/// Real implementation would check against exact group order, but this is sufficient for examples
fn is_valid_secp256k1_key(bytes: &[u8; 32]) -> bool {
    // secp256k1 group order starts with 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE
    // So any key starting with 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF is definitely too large
    // Return false if the first 12 bytes are all 0xFF
    for i in 0..12 {
        if bytes[i] != 0xFF {
            return true;
        }
    }
    false
}

/// Convert a 32-byte big-endian private key into 4 little-endian 64-bit limbs
fn bytes_to_le_limbs(bytes_be: &[u8; 32]) -> [u64; 4] {
    // Reverse to little-endian byte order first, then chunk into u64 limbs
    let mut le = [0u8; 32];
    for i in 0..32 {
        le[i] = bytes_be[31 - i];
    }
    [
        u64::from_le_bytes(le[0..8].try_into().unwrap()),
        u64::from_le_bytes(le[8..16].try_into().unwrap()),
        u64::from_le_bytes(le[16..24].try_into().unwrap()),
        u64::from_le_bytes(le[24..32].try_into().unwrap()),
    ]
}
