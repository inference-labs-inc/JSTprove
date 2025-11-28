use circuit_std_rs::logup::{query_count_by_key_hint, query_count_hint, rangeproof_hint};
use expander_compiler::field::Field as CompilerField;
use expander_compiler::hints::registry::HintRegistry;

/// Build a HintRegistry with all LogUp-related hints registered.
pub fn build_logup_hint_registry<F: CompilerField>() -> HintRegistry<F> {
    let mut registry = HintRegistry::<F>::new();

    // The strings *must* match the ones used in new_hint(...)
    registry.register("myhint.querycounthint", query_count_hint::<F>);
    registry.register("myhint.querycountbykeyhint", query_count_by_key_hint::<F>);
    registry.register("myhint.rangeproofhint", rangeproof_hint::<F>);

    registry
}
