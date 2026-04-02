//! Public gadget API for circuit construction.
//!
//! Re-exports all constraint-enforcing gadgets used by layers.

pub mod autotuner;
pub mod chunk_table;
pub mod euclidean_division;
pub mod function_lookup;
pub mod linear_algebra;
pub mod max_min_clip;
pub mod range_check;

pub use function_lookup::{FunctionLookupTable, function_lookup_bits, i64_to_field};
pub use max_min_clip::{
    ShiftRangeContext, constrained_clip, constrained_max, constrained_max_2, constrained_min,
    constrained_min_2, constrained_relu,
};
pub use range_check::{
    DEFAULT_LOGUP_CHUNK_BITS, LogupRangeCheckContext, constrained_reconstruct_from_bits,
    logup_range_check_pow2_unsigned, range_check_pow2_unsigned,
};
