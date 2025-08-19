//! Circuit construction utilities for JSTProve, a zero-knowledge proof system
//! supporting fixed-point quantization, neural network inference, and modular
//! arithmetic over finite fields.
//!
//! This crate is structured around three core modules:
//!
//! - [`circuit_functions`]: Defines both low-level arithmetic gadgets and
//!   high-level building blocks for layers such as matmul, convolution,
//!   ReLU activation, and quantized rescaling.
//!
//! - [`runner`]: Provides CLI-based orchestration for compiling, proving, and
//!   verifying circuits, including witness generation and memory tracking.
//!
//! - [`io`]: Manages input/output serialization, especially for ONNX exports
//!   and JSON-formatted data.
//!
//! Typical usage involves calling layer constructors from [`circuit_functions`],
//! such as [`layer_matmul`] or [`layer_conv`], or invoking rescaling utilities
//! from [`utils_quantization`].
//!
//! # Feature Flags
//! This crate requires the `specialization` nightly feature.

#![feature(specialization)]

pub mod circuit_functions;
pub mod io;
pub mod runner;
