#![allow(clippy::all, clippy::pedantic)]

mod buffers;
mod device;
mod kernels;
mod kernels_gl;

pub use buffers::{GoldilocksBufferPool, MetalBufferPool, BN254_ELEM_SIZE, GOLDILOCKS_ELEM_SIZE};
pub use device::{MetalAccelerator, GPU_DISPATCH_THRESHOLD};
pub use kernels::{
    metal_cross_prod_eq, metal_eq_eval_at, metal_fold_all, metal_fold_f, metal_fold_hg,
    metal_poly_eval, metal_vec_add, BN254_R,
};
pub use kernels_gl::{
    gl_metal_eq_eval_at, gl_metal_fold_all, gl_metal_poly_eval, gl_metal_vec_add, GOLDILOCKS_R,
};
