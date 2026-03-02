mod buffers;
mod device;
mod kernels;

pub use buffers::MetalBufferPool;
pub use device::MetalAccelerator;
pub use kernels::{
    metal_accumulate_add_gates, metal_accumulate_mul_gates, metal_cross_prod_eq, metal_eq_eval_at,
    metal_fold_f, metal_fold_hg, metal_poly_eval,
};
