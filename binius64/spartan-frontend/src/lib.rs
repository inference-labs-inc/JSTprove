// Copyright 2025 Irreducible Inc.

#![warn(rustdoc::missing_crate_level_docs)]

//! Frontend for building constraint systems for the Binius proof system.
//!
//! This crate provides tools for constructing and optimizing constraint systems that can be proven
//! using the Binius SNARK protocol. It enables writing circuit logic once that works for both
//! constraint generation and witness computation.
//!
//! # Overview
//!
//! A constraint system consists of multiplication constraints over binary field elements:
//! - Each constraint has the form `A * B = C`
//! - A, B, C are operands (XOR combinations of witness values)
//! - Constraints reference positions in a witness array by index
//!
//! The witness is an array of field elements that satisfies all constraints. Part of the witness
//! (public inputs/outputs) is known to the verifier, while the rest (private values) is known
//! only to the prover.
//!
//! # Workflow
//!
//! 1. **Circuit Construction**: Use [`CircuitBuilder`] to build constraints symbolically
//! 2. **Compilation**: Convert to optimized [`ConstraintSystem`] via [`compile()`]
//! 3. **Witness Generation**: Use [`WitnessGenerator`] to compute witness values
//! 4. **Validation**: Verify witness satisfies constraints (for testing)
//!
//! # Example
//!
//! ```rust
//! use binius_field::{BinaryField128bGhash as B128, Field};
//! use binius_spartan_frontend::{
//!     circuit_builder::{CircuitBuilder, ConstraintBuilder, WitnessGenerator},
//!     compiler::compile,
//! };
//!
//! // Define a simple circuit: assert that a * b = c
//! fn multiply<Builder: CircuitBuilder>(
//!     builder: &mut Builder,
//!     a: Builder::Wire,
//!     b: Builder::Wire,
//! ) -> Builder::Wire {
//!     builder.mul(a, b)
//! }
//!
//! // Build constraint system
//! let mut constraint_builder = ConstraintBuilder::new();
//! let a_wire = constraint_builder.alloc_inout();
//! let b_wire = constraint_builder.alloc_inout();
//! let c_wire = constraint_builder.alloc_inout();
//! let product = multiply(&mut constraint_builder, a_wire, b_wire);
//! constraint_builder.assert_eq(product, c_wire);
//!
//! // Compile to optimized constraint system
//! let (cs, layout) = compile(constraint_builder);
//!
//! // Generate witness with concrete values
//! let mut witness_gen = WitnessGenerator::new(&layout);
//! let a = witness_gen.write_inout(a_wire, B128::new(3));
//! let b = witness_gen.write_inout(b_wire, B128::new(5));
//! let c = witness_gen.write_inout(c_wire, B128::new(15));
//! let product = multiply(&mut witness_gen, a, b);
//! witness_gen.assert_eq(product, c);
//! let witness = witness_gen.build().unwrap();
//!
//! // Validate witness satisfies constraints
//! cs.validate(&witness);
//! ```
//!
//! # Architecture
//!
//! The crate uses a multi-phase compilation pipeline:
//!
//! 1. **IR Construction** ([`ConstraintSystemIR`]): Symbolic constraints with zero constraints
//!    (additions) and multiplication constraints, using symbolic wires
//!
//! 2. **Optimization** ([`wire_elimination`]): Eliminates unnecessary private wires by substituting
//!    zero constraints into multiplication constraints
//!
//! 3. **Finalization**: Converts symbolic wires to witness indices, producing the final
//!    [`ConstraintSystem`] and [`WitnessLayout`]
//!
//! This separation allows optimization passes to work on a flexible IR while producing an
//! efficient final constraint system that directly references witness array positions.
//!
//! [`CircuitBuilder`]: circuit_builder::CircuitBuilder
//! [`ConstraintBuilder`]: circuit_builder::ConstraintBuilder
//! [`WitnessGenerator`]: circuit_builder::WitnessGenerator
//! [`ConstraintSystemIR`]: circuit_builder::ConstraintSystemIR
//! [`ConstraintSystem`]: constraint_system::ConstraintSystem
//! [`WitnessLayout`]: constraint_system::WitnessLayout
//! [`compile()`]: compiler::compile

pub mod circuit_builder;
pub mod circuits;
pub mod compiler;
pub mod constraint_system;
pub mod wire_elimination;
