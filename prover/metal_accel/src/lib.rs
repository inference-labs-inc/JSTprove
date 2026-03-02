#![allow(clippy::all, clippy::pedantic)]

mod buffers;
mod device;
mod kernels;

pub use buffers::{MetalBufferPool, BN254_ELEM_SIZE};
pub use device::{MetalAccelerator, GPU_DISPATCH_THRESHOLD};
pub use kernels::{
    metal_accumulate_add_gates, metal_accumulate_mul_gates, metal_cross_prod_eq, metal_eq_eval_at,
    metal_fold_f, metal_fold_hg, metal_poly_eval, BN254_R,
};
