#![allow(clippy::all, clippy::pedantic)]

mod buffers;
mod device;
mod kernels;

pub use buffers::{MetalBufferPool, BN254_ELEM_SIZE};
pub use device::{MetalAccelerator, GPU_DISPATCH_THRESHOLD};
pub use kernels::{
    metal_cross_prod_eq, metal_eq_eval_at, metal_fold_all, metal_fold_f, metal_fold_hg,
    metal_poly_eval, metal_vec_add, BN254_R,
};
