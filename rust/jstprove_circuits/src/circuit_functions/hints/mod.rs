//! Hint infrastructure: LogUp hints + unconstrained arithmetic helpers.

use ethnum::U256;
use expander_compiler::field::FieldArith;

pub mod bits;
pub use bits::unconstrained_to_bits;

pub mod exp;
pub use exp::{compute_exp_quantized, exp_hint};

pub mod gelu;
pub use gelu::{compute_gelu_quantized, gelu_hint};

pub mod sigmoid;
pub use sigmoid::{compute_sigmoid_quantized, sigmoid_hint};

pub mod softmax;
pub use softmax::softmax_hint;

pub mod softmax_verified;
pub use softmax_verified::softmax_verified_hint;

pub mod layer_norm;
pub use layer_norm::layer_norm_hint;

pub mod layer_norm_verified;
pub use layer_norm_verified::layer_norm_verified_hint;

pub mod max_min_clip;
pub use max_min_clip::{unconstrained_clip, unconstrained_max, unconstrained_min};

pub mod gridsample;
pub use gridsample::gridsample_hint;

pub mod resize;
pub use resize::resize_hint;

pub mod topk;
pub use topk::topk_hint;

pub mod hardswish;
pub use hardswish::hardswish_hint;

pub mod global_averagepool;
pub use global_averagepool::global_averagepool_hint;

pub mod instance_norm;
pub use instance_norm::instance_norm_hint;

pub mod log;
pub use log::log_hint;

pub mod reduce_mean;
pub use reduce_mean::reduce_mean_hint;

pub mod sqrt;
pub use sqrt::sqrt_hint;

pub mod tanh;
pub use tanh::tanh_hint;

pub mod erf;
pub use erf::{compute_erf_quantized, erf_abs_hint, erf_hint};

pub mod averagepool;
pub use averagepool::averagepool_hint;

pub mod pow;
pub use pow::pow_hint;

pub mod sin;
pub use sin::sin_hint;
pub mod cos;
pub use cos::cos_hint;
pub mod reduce_max;
pub use reduce_max::reduce_max_hint;

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
    registry.register(
        softmax_verified::SOFTMAX_VERIFIED_HINT_KEY,
        softmax_verified_hint::<F>,
    );
    registry.register(layer_norm::LAYER_NORM_HINT_KEY, layer_norm_hint::<F>);
    registry.register(
        layer_norm_verified::LAYER_NORM_VERIFIED_HINT_KEY,
        layer_norm_verified_hint::<F>,
    );
    registry.register(gelu::GELU_HINT_KEY, gelu_hint::<F>);
    registry.register(gridsample::GRIDSAMPLE_HINT_KEY, gridsample_hint::<F>);
    registry.register(resize::RESIZE_HINT_KEY, resize_hint::<F>);
    registry.register(topk::TOPK_HINT_KEY, topk_hint::<F>);
    registry.register(log::LOG_HINT_KEY, log_hint::<F>);
    registry.register(reduce_mean::REDUCE_MEAN_HINT_KEY, reduce_mean_hint::<F>);
    registry.register(sqrt::SQRT_HINT_KEY, sqrt_hint::<F>);
    registry.register(tanh::TANH_HINT_KEY, tanh_hint::<F>);
    registry.register(erf::ERF_HINT_KEY, erf_hint::<F>);
    registry.register(erf::ERF_ABS_HINT_KEY, erf_abs_hint::<F>);
    registry.register(averagepool::AVERAGEPOOL_HINT_KEY, averagepool_hint::<F>);
    registry.register(pow::POW_HINT_KEY, pow_hint::<F>);
    registry.register(hardswish::HARDSWISH_HINT_KEY, hardswish_hint::<F>);
    registry.register(
        global_averagepool::GLOBAL_AVERAGEPOOL_HINT_KEY,
        global_averagepool_hint::<F>,
    );
    registry.register(
        instance_norm::INSTANCE_NORM_HINT_KEY,
        instance_norm_hint::<F>,
    );
    registry.register(sin::SIN_HINT_KEY, sin_hint::<F>);
    registry.register(cos::COS_HINT_KEY, cos_hint::<F>);
    registry.register(reduce_max::REDUCE_MAX_HINT_KEY, reduce_max_hint::<F>);

    registry
}
