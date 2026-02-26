#![allow(
    clippy::pedantic,
    clippy::collapsible_if,
    clippy::should_implement_trait,
    clippy::unnecessary_filter_map
)]

pub mod compat;
pub mod graph;
pub mod ops;
pub mod parser;
pub mod quantizer;
pub mod shape_inference;

#[allow(clippy::all, clippy::pedantic)]
#[allow(non_camel_case_types)]
mod onnx_ml {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub use onnx_ml::*;
