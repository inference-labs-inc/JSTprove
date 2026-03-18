#![allow(clippy::all, clippy::pedantic)]

#[cfg(target_os = "macos")]
mod buffers;
#[cfg(target_os = "macos")]
mod device;
#[cfg(target_os = "macos")]
mod kernels;
mod kernels_gl;

#[cfg(target_os = "macos")]
pub use buffers::{GoldilocksBufferPool, MetalBufferPool, BN254_ELEM_SIZE, GOLDILOCKS_ELEM_SIZE};
#[cfg(target_os = "macos")]
pub use device::{MetalAccelerator, GPU_DISPATCH_THRESHOLD};
#[cfg(target_os = "macos")]
pub use kernels::{
    metal_cross_prod_eq, metal_eq_eval_at, metal_fold_all, metal_fold_f, metal_fold_hg,
    metal_poly_eval, metal_vec_add, BN254_R,
};
pub use kernels_gl::{
    gl_metal_eq_eval_at, gl_metal_fold_all, gl_metal_poly_eval, gl_metal_vec_add, GOLDILOCKS_R,
};
