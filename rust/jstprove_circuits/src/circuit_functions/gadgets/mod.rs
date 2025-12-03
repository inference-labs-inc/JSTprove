pub mod max_min_clip;
pub mod range_check;

pub use range_check::{
    DEFAULT_LOGUP_CHUNK_BITS,
    LogupRangeCheckContext,
    constrained_reconstruct_from_bits,
    logup_range_check_pow2_unsigned,
    range_check_pow2_unsigned, // legacy, still okay to expose
};

pub use max_min_clip::{ShiftRangeContext, constrained_clip, constrained_max, constrained_min};
