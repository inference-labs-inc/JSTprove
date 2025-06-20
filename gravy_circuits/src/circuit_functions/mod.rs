pub mod quantization;
pub mod relu;
pub mod convolution_fn;
pub mod matrix_computation;
pub mod helper_fn;

pub mod core_operations;
pub mod layer_operations;

#[doc(inline)]
pub use core_operations::assert_is_max;

#[doc(inline)]
pub use core_operations::rescale;