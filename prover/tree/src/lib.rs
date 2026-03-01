#![allow(clippy::pedantic)]
//! This module defines the core components of a Merkle tree implementation.
//! It includes definitions for tree structures, nodes, leaves, and paths.

mod tree;
pub use tree::*;

mod node;
pub use node::*;

mod leaf;
pub use leaf::*;

mod path;
pub use path::*;

#[cfg(test)]
mod tests;
