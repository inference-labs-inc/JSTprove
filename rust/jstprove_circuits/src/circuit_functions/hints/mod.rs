//! Hint infrastructure: LogUp hints + unconstrained arithmetic helpers.

pub mod bits;
pub use bits::unconstrained_to_bits;

pub mod max_min_clip;
pub use max_min_clip::{unconstrained_clip, unconstrained_max, unconstrained_min};

/// LogUp hint registration
use circuit_std_rs::logup::{query_count_by_key_hint, query_count_hint, rangeproof_hint};
use expander_compiler::field::Field as CompilerField;
use expander_compiler::hints::registry::HintRegistry;

/// Build a HintRegistry with all LogUp-related hints registered.
/// These names MUST match the identifiers used by new_hint(...).
pub fn build_logup_hint_registry<F: CompilerField>() -> HintRegistry<F> {
    let mut registry = HintRegistry::<F>::new();

    registry.register("myhint.querycounthint", query_count_hint::<F>);
    registry.register("myhint.querycountbykeyhint", query_count_by_key_hint::<F>);
    registry.register("myhint.rangeproofhint", rangeproof_hint::<F>);

    registry
}
