//! Circuit construction utilities for `JSTprove`, a zero-knowledge proof system
//! supporting fixed-point quantization, neural-network inference, and modular
//! arithmetic over finite fields.
//!
//! # Crate Structure
//!
//! - [`circuit_functions`]: Low-level arithmetic gadgets (LogUp, range-checks,
//!   max/min/clip, etc.) and high-level building blocks for layers such as
//!   matmul, convolution, `ReLU`, and quantized rescaling.
//!
//! - [`runner`]: CLI-oriented orchestration for compiling, proving, and
//!   verifying circuits, including witness generation and memory tracking.
//!
//! - [`io`]: Input/output helpers for serializing circuit inputs and outputs,
//!   including ONNX exports and JSON-encoded tensors.
//!
//! Typical usage involves composing layer gadgets from [`circuit_functions`],
//! then invoking the tools in [`runner`] to generate and verify proofs.
//!
//! # Feature Flags
//!
//! This crate requires the nightly feature `min_specialization`.
#![allow(
    clippy::doc_markdown,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items
)]
#![feature(min_specialization)]

pub mod api;
pub mod circuit_functions;
#[allow(
    clippy::must_use_candidate,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss
)]
pub mod cli;
#[allow(clippy::pedantic)]
pub mod expander_metadata;
pub mod io;
pub mod onnx;
pub mod proof_config;
pub mod proof_system;
pub mod runner;

pub use circuit_functions::layers::LayerKind;
pub use proof_config::{Field, ProofConfig, ProofConfigError, StampedProofConfig};
pub use proof_system::{ProofSystem, ProofSystemParseError};
pub use runner::version::{ArtifactVersion, jstprove_artifact_version};
