use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind},
};

use crate::circuit_functions::gadgets::LogupRangeCheckContext;

use crate::circuit_functions::hints::{unconstrained_max, unconstrained_min};

/// Shift parameters for signed-range max/min gadgets.
pub struct ShiftRangeContext {
    pub(crate) shift_exponent: usize,
    pub(crate) offset: Variable,
}

impl ShiftRangeContext {
    /// # Errors
    /// Returns `LayerError` if `shift_exponent` overflows `u32` or `2^s` overflows `u64`.
    pub fn new<C: Config, Builder: RootAPI<C>>(
        api: &mut Builder,
        layer: LayerKind,
        shift_exponent: usize,
    ) -> Result<Self, LayerError> {
        let offset_: u64 = 1u64
            .checked_shl(
                u32::try_from(shift_exponent).map_err(|_| LayerError::Other {
                    layer: layer.clone(),
                    msg: format!("Shift exponent {shift_exponent} is too large for type: u32"),
                })?,
            )
            .ok_or_else(|| LayerError::InvalidParameterValue {
                layer,
                layer_name: "ShiftRangeContext".to_string(),
                param_name: "shift_exponent".to_string(),
                value: shift_exponent.to_string(),
            })?;
        let offset = api.constant(CircuitField::<C>::from_u256(ethnum::U256::from(offset_)));
        Ok(Self {
            shift_exponent,
            offset,
        })
    }
}

/// Constrained max over a slice of variables. Dispatches to `constrained_max_2`
/// when `values.len() == 2` for reduced range-check cost.
///
/// # Errors
/// Returns `CircuitError` on empty input, shift overflow, or LogUp failure.
pub fn constrained_max<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    shift_ctx: &ShiftRangeContext,
    logup_ctx: &mut LogupRangeCheckContext,
    values: &[Variable],
) -> Result<Variable, CircuitError> {
    if values.is_empty() {
        return Err(LayerError::Other {
            layer: LayerKind::Max,
            msg: "constrained_max: input slice must be nonempty".to_string(),
        }
        .into());
    }

    if values.len() == 2 {
        return constrained_max_2(api, shift_ctx, logup_ctx, values[0], values[1]);
    }

    let mut values_offset = Vec::with_capacity(values.len());
    for &x in values {
        values_offset.push(api.add(x, shift_ctx.offset));
    }

    let max_offset = unconstrained_max(api, &values_offset)?;
    let max_raw = api.sub(max_offset, shift_ctx.offset);

    let n_bits = shift_ctx.shift_exponent.checked_add(1).ok_or_else(|| {
        LayerError::InvalidParameterValue {
            layer: LayerKind::Max,
            layer_name: "ShiftRangeContext".to_string(),
            param_name: "shift_exponent".to_string(),
            value: shift_ctx.shift_exponent.to_string(),
        }
    })?;

    let mut prod = api.constant(1);

    for &x in values {
        let delta = api.sub(max_raw, x);

        logup_ctx
            .range_check::<C, Builder>(api, delta, n_bits)
            .map_err(|e| LayerError::Other {
                layer: LayerKind::Max,
                msg: format!("logup_range_check_pow2_unsigned (LogUp) failed: {e}"),
            })?;

        prod = api.mul(prod, delta);
    }

    api.assert_is_zero(prod);
    Ok(max_raw)
}

/// Boolean-selector specialization of `constrained_max` for exactly two inputs.
/// Uses one LogUp range check on `|x - a|` instead of two on individual deltas.
///
/// # Errors
/// Returns `CircuitError` on shift overflow or LogUp failure.
pub fn constrained_max_2<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    shift_ctx: &ShiftRangeContext,
    logup_ctx: &mut LogupRangeCheckContext,
    x: Variable,
    a: Variable,
) -> Result<Variable, CircuitError> {
    let n_bits = shift_ctx.shift_exponent.checked_add(1).ok_or_else(|| {
        LayerError::InvalidParameterValue {
            layer: LayerKind::Max,
            layer_name: "ShiftRangeContext".to_string(),
            param_name: "shift_exponent".to_string(),
            value: shift_ctx.shift_exponent.to_string(),
        }
    })?;

    let x_sh = api.add(x, shift_ctx.offset);
    let a_sh = api.add(a, shift_ctx.offset);
    let b = api.unconstrained_greater_eq(x_sh, a_sh);
    api.assert_is_bool(b);

    let diff = api.sub(x, a);
    let b_diff = api.mul(b, diff);
    let result = api.add(a, b_diff);

    let two_b = api.add(b, b);
    let one = api.constant(1);
    let selector = api.sub(two_b, one);
    let abs_diff = api.mul(selector, diff);

    logup_ctx
        .range_check::<C, Builder>(api, abs_diff, n_bits)
        .map_err(|e| LayerError::Other {
            layer: LayerKind::Max,
            msg: format!("constrained_max_2 LogUp range check failed: {e}"),
        })?;

    Ok(result)
}

/// Constrained ReLU: `max(x, 0)` via boolean selector with one LogUp range check.
///
/// # Errors
/// Returns `CircuitError` on shift overflow or LogUp failure.
pub fn constrained_relu<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    shift_ctx: &ShiftRangeContext,
    logup_ctx: &mut LogupRangeCheckContext,
    x: Variable,
) -> Result<Variable, CircuitError> {
    let n_bits = shift_ctx.shift_exponent.checked_add(1).ok_or_else(|| {
        LayerError::InvalidParameterValue {
            layer: LayerKind::ReLU,
            layer_name: "ShiftRangeContext".to_string(),
            param_name: "shift_exponent".to_string(),
            value: shift_ctx.shift_exponent.to_string(),
        }
    })?;

    let x_shifted = api.add(x, shift_ctx.offset);
    let b = api.unconstrained_greater_eq(x_shifted, shift_ctx.offset);
    api.assert_is_bool(b);

    let result = api.mul(b, x);

    let two_result = api.add(result, result);
    let abs_x = api.sub(two_result, x);

    logup_ctx
        .range_check::<C, Builder>(api, abs_x, n_bits)
        .map_err(|e| LayerError::Other {
            layer: LayerKind::ReLU,
            msg: format!("constrained_relu LogUp range check failed: {e}"),
        })?;

    Ok(result)
}

/// Constrained min over a slice of variables. Dispatches to `constrained_min_2`
/// when `values.len() == 2` for reduced range-check cost.
///
/// # Errors
/// Returns `CircuitError` on empty input, shift overflow, or LogUp failure.
pub fn constrained_min<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    context: &ShiftRangeContext,
    logup_ctx: &mut LogupRangeCheckContext,
    values: &[Variable],
) -> Result<Variable, CircuitError> {
    if values.is_empty() {
        return Err(LayerError::Other {
            layer: LayerKind::Min,
            msg: "constrained_min: input slice must be nonempty".to_string(),
        }
        .into());
    }

    if values.len() == 2 {
        return constrained_min_2(api, context, logup_ctx, values[0], values[1]);
    }

    let mut values_offset = Vec::with_capacity(values.len());
    for &x in values {
        values_offset.push(api.add(x, context.offset));
    }

    let min_offset = unconstrained_min(api, &values_offset)?;
    let min_raw = api.sub(min_offset, context.offset);

    let n_bits =
        context
            .shift_exponent
            .checked_add(1)
            .ok_or_else(|| LayerError::InvalidParameterValue {
                layer: LayerKind::Min,
                layer_name: "ShiftRangeContext".to_string(),
                param_name: "shift_exponent".to_string(),
                value: context.shift_exponent.to_string(),
            })?;

    let mut prod = api.constant(1);

    for &x in values {
        let delta = api.sub(x, min_raw);

        logup_ctx
            .range_check::<C, Builder>(api, delta, n_bits)
            .map_err(|e| LayerError::Other {
                layer: LayerKind::Min,
                msg: format!("logup_range_check_pow2_unsigned (LogUp) failed: {e}"),
            })?;

        prod = api.mul(prod, delta);
    }

    api.assert_is_zero(prod);
    Ok(min_raw)
}

/// Boolean-selector specialization of `constrained_min` for exactly two inputs.
/// Uses one LogUp range check on `|a - x|` instead of two on individual deltas.
///
/// # Errors
/// Returns `CircuitError` on shift overflow or LogUp failure.
pub fn constrained_min_2<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    shift_ctx: &ShiftRangeContext,
    logup_ctx: &mut LogupRangeCheckContext,
    x: Variable,
    a: Variable,
) -> Result<Variable, CircuitError> {
    let n_bits = shift_ctx.shift_exponent.checked_add(1).ok_or_else(|| {
        LayerError::InvalidParameterValue {
            layer: LayerKind::Min,
            layer_name: "ShiftRangeContext".to_string(),
            param_name: "shift_exponent".to_string(),
            value: shift_ctx.shift_exponent.to_string(),
        }
    })?;

    let x_sh = api.add(x, shift_ctx.offset);
    let a_sh = api.add(a, shift_ctx.offset);
    let b = api.unconstrained_greater_eq(a_sh, x_sh);
    api.assert_is_bool(b);

    let diff = api.sub(a, x);
    let b_diff = api.mul(b, diff);
    let result = api.sub(a, b_diff);

    let two_b = api.add(b, b);
    let one = api.constant(1);
    let selector = api.sub(two_b, one);
    let abs_diff = api.mul(selector, diff);

    logup_ctx
        .range_check::<C, Builder>(api, abs_diff, n_bits)
        .map_err(|e| LayerError::Other {
            layer: LayerKind::Min,
            msg: format!("constrained_min_2 LogUp range check failed: {e}"),
        })?;

    Ok(result)
}

/// Constrained clip: `min(max(x, lower), upper)` using specialized two-element gadgets.
///
/// # Errors
/// Returns `CircuitError` on shift overflow or LogUp failure.
pub fn constrained_clip<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    range_ctx: &ShiftRangeContext,
    logup_ctx: &mut LogupRangeCheckContext,
    x: Variable,
    lower: Option<Variable>,
    upper: Option<Variable>,
) -> Result<Variable, CircuitError> {
    if lower.is_none() && upper.is_none() {
        return Ok(x);
    }

    let remap = |e: CircuitError| -> CircuitError {
        match e {
            CircuitError::Layer(le) => CircuitError::Layer(le.with_layer(LayerKind::Clip)),
            other => other,
        }
    };

    let mut cur = if let Some(a) = lower {
        constrained_max_2(api, range_ctx, logup_ctx, x, a).map_err(remap)?
    } else {
        x
    };

    if let Some(b) = upper {
        cur = constrained_min_2(api, range_ctx, logup_ctx, cur, b).map_err(remap)?;
    }

    Ok(cur)
}
