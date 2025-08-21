pub mod relu;
pub mod conv;
pub mod gemm;
pub mod maxpool;
pub mod reshape;
pub mod flatten;
pub mod constant;
pub mod layer_ops;
mod errors;
mod layer_kinds;

pub use errors::LayerError;
pub use layer_kinds::LayerKind;
