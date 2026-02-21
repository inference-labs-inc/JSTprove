pub mod parser;
pub mod graph;
pub mod quantizer;
pub mod ops;

#[allow(clippy::all)]
#[allow(non_camel_case_types)]
mod onnx_ml {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub use onnx_ml::*;
