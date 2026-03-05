//! Hint infrastructure: LogUp hints + unconstrained arithmetic helpers.

use ethnum::U256;
use expander_compiler::field::FieldArith;

pub mod bits;
pub use bits::unconstrained_to_bits;

pub mod exp;
pub use exp::exp_hint;

pub mod sigmoid;
pub use sigmoid::sigmoid_hint;

pub mod softmax;
pub use softmax::softmax_hint;

pub mod layer_norm;
pub use layer_norm::layer_norm_hint;

pub mod max_min_clip;
pub use max_min_clip::{unconstrained_clip, unconstrained_max, unconstrained_min};

/// LogUp hint registration
use circuit_std_rs::logup::{query_count_by_key_hint, query_count_hint, rangeproof_hint};
use expander_compiler::field::Field as CompilerField;
use expander_compiler::hints::registry::HintRegistry;

#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
pub fn field_to_i64<F: FieldArith>(x: F) -> i64 {
    let p_half = F::MODULUS / 2;
    let xu = x.to_u256();
    if xu > p_half {
        let neg_magnitude = F::MODULUS - xu;
        let max_i64 = U256::from(i64::MAX as u64);
        if neg_magnitude > max_i64 {
            i64::MIN
        } else {
            -(neg_magnitude.as_u64() as i64)
        }
    } else {
        let max_i64 = U256::from(i64::MAX as u64);
        if xu > max_i64 {
            i64::MAX
        } else {
            xu.as_u64() as i64
        }
    }
}

/// Build a HintRegistry with all LogUp-related hints registered.
/// These names MUST match the identifiers used by new_hint(...).
pub fn build_logup_hint_registry<F: CompilerField>() -> HintRegistry<F> {
    let mut registry = HintRegistry::<F>::new();

    registry.register("myhint.querycounthint", query_count_hint::<F>);
    registry.register("myhint.querycountbykeyhint", query_count_by_key_hint::<F>);
    registry.register("myhint.rangeproofhint", rangeproof_hint::<F>);
    registry.register(exp::EXP_HINT_KEY, exp_hint::<F>);
    registry.register(sigmoid::SIGMOID_HINT_KEY, sigmoid_hint::<F>);
    registry.register(softmax::SOFTMAX_HINT_KEY, softmax_hint::<F>);
    registry.register(layer_norm::LAYER_NORM_HINT_KEY, layer_norm_hint::<F>);

    registry
}
