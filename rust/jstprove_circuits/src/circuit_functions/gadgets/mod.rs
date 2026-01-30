//! Public gadget API for circuit construction.
//!
//! Re-exports all constraint-enforcing gadgets used by layers.

pub mod euclidean_algebra;
pub mod linear_algebra;
pub mod max_min_clip;
pub mod range_check;

// temporary until v2 replaces original
pub mod range_check_v2; 

pub use max_min_clip::{ShiftRangeContext, constrained_clip, constrained_max, constrained_min};
pub use range_check::{
    DEFAULT_LOGUP_CHUNK_BITS, LogupRangeCheckContext, constrained_reconstruct_from_bits,
    logup_range_check_pow2_unsigned, range_check_pow2_unsigned,
};

// temporary until v2 replaces original
pub use range_check_v2::{
    DEFAULT_LOGUP_CHUNK_BITS as DEFAULT_LOGUP_CHUNK_BITS_V2,
    LogupRangeCheckContext as LogupRangeCheckContextV2,
    logup_range_check_pow2_unsigned as logup_range_check_pow2_unsigned_v2,
};
