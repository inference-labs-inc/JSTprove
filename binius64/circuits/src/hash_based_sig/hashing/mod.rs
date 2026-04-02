// Copyright 2025 Irreducible Inc.
//! Tweaked Keccak-256 circuits for hash-based signatures.
mod base;
mod chain;
mod message;
mod public_key;
mod tree;

pub use chain::{
    CHAIN_TWEAK, FIXED_MESSAGE_OVERHEAD, build_chain_hash, circuit_chain_hash, hash_chain_keccak,
};
pub use message::{MESSAGE_TWEAK, build_message_hash, circuit_message_hash, hash_message};
pub use public_key::{
    PUBLIC_KEY_TWEAK, build_public_key_hash, circuit_public_key_hash, hash_public_key_keccak,
};
pub use tree::{
    TREE_MESSAGE_OVERHEAD, TREE_TWEAK, build_tree_hash, circuit_tree_hash, hash_tree_node_keccak,
};
