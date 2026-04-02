// Copyright 2025 Irreducible Inc.

//! Circuit construction frontend for Binius64.
//!
//! This crate provides the [`CircuitBuilder`] API for constructing arithmetic circuits
//! that compile to Binius64 constraint systems. You describe your computation as a graph
//! of operations on 64-bit words, and the frontend compiles it to AND/MUL constraints.
//!
//! # Usage Flow
//!
//! Use [`CircuitBuilder`] to construct your circuit. Call methods like `add_witness()`
//! and `add_inout()` to create [`Wire`]s - handles to 64-bit values that will exist during
//! proof generation. Use operations like `band()`, `bxor()`, and `iadd_32()` to transform
//! these wires, building up your computation graph.
//!
//! When you call `build()`, the builder compiles your graph into a [`Circuit`]. This circuit
//! contains the optimized constraint system and everything needed for proof generation.
//!
//! To generate a witness, create a [`WitnessFiller`] from the circuit. Assign concrete values
//! to your input wires, then call `populate_wire_witness()` to compute all intermediate values
//! through circuit evaluation.
//!
//! Use [`CircuitStat`] to inspect metrics like constraint counts and wire usage, helpful for
//! optimization and debugging.

#![warn(rustdoc::missing_crate_level_docs)]

mod compiler;
pub mod stat;

pub mod util;

pub use compiler::{
    CircuitBuilder, Wire,
    circuit::{Circuit, WitnessFiller},
};
pub use stat::CircuitStat;
