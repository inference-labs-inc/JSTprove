mod core;
pub use core::analyze_model_internal;

#[cfg(feature = "python")]
mod pybindings;