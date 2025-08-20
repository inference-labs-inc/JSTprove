//! Utility modules used throughout circuit construction and ONNX translation.

pub mod build_layers;
pub mod constants;
pub mod core_math;       // Bit-level and finite field math
pub mod graph_pattern_matching;
pub mod json_array;      // JSON-to-array conversions and trait lifting
pub mod onnx_model;      // ONNX layer param extraction and model shape helpers
pub mod onnx_types;      // 
pub mod quantization;    // Quantization utilities and rescaling logic
pub mod shaping;         // Shape manipulation and input partitioning
pub mod tensor_ops;      // Conversions between nested Vecs and ArrayD
