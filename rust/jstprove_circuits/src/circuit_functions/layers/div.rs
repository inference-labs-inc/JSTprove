use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::ArrayD;

use crate::circuit_functions::gadgets::LogupRangeCheckContext;
use crate::circuit_functions::utils::onnx_model::get_optional_w_or_b;
use crate::circuit_functions::utils::tensor_ops::load_array_constants_or_get_inputs;
use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params_and_expected_shape, get_input_name},
    },
};

#[derive(Debug)]
pub struct DivLayer {
    name: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    initializer_a: Option<ArrayD<i64>>,
    initializer_b: Option<ArrayD<i64>>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for DivLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let a_name = get_input_name(&self.inputs, 0, LayerKind::Div, INPUT)?;

        let a_input = load_array_constants_or_get_inputs(
            api,
            &input,
            a_name,
            &self.initializer_a,
            LayerKind::Div,
        )?;

        let divisor_const = self.initializer_b.as_ref().ok_or_else(|| LayerError::Other {
            layer: LayerKind::Div,
            msg: "Div requires constant divisor (initializer_b)".to_string(),
        })?;

        let divisor_val = divisor_const.iter().next().copied().unwrap_or(1) as u32;
        if divisor_val == 0 {
            return Err(CircuitError::from(LayerError::Other {
                layer: LayerKind::Div,
                msg: "Division by zero".to_string(),
            }));
        }

        let mut logup_ctx = LogupRangeCheckContext::new_default();
        logup_ctx.init::<C, Builder>(api);

        let divisor_var = api.constant(divisor_val);
        let shape = a_input.shape().to_vec();

        let results: Result<Vec<Variable>, CircuitError> = if divisor_val == 1 {
            a_input.iter().cloned().map(Ok).collect()
        } else {
            let divisor_minus_one = api.constant(divisor_val - 1);
            let remainder_bound_bits = (32 - (divisor_val - 1).leading_zeros()) as usize;

            a_input
                .iter()
                .map(|dividend| {
                    let quotient = api.unconstrained_int_div(*dividend, divisor_val);
                    let remainder = api.unconstrained_mod(*dividend, divisor_val);

                    let prod = api.mul(divisor_var, quotient);
                    let reconstructed = api.add(prod, remainder);
                    api.assert_is_equal(*dividend, reconstructed);

                    let diff = api.sub(divisor_minus_one, remainder);
                    logup_ctx
                        .range_check::<C, Builder>(api, diff, remainder_bound_bits)
                        .map_err(|e| LayerError::Other {
                            layer: LayerKind::Div,
                            msg: format!("remainder bound check failed: {e}"),
                        })?;

                    Ok(quotient)
                })
                .collect()
        };

        logup_ctx.finalize::<C, Builder>(api);

        let out = ArrayD::from_shape_vec(shape, results?).map_err(|e| LayerError::Other {
            layer: LayerKind::Div,
            msg: format!("Failed to reshape result: {e}"),
        })?;

        Ok((self.outputs.clone(), out))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let (_, _) = extract_params_and_expected_shape(layer_context, layer).map_err(|e| {
            LayerError::Other {
                layer: LayerKind::Div,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            }
        })?;

        let initializer_a = get_optional_w_or_b(layer_context, &layer.inputs[0])?;
        let initializer_b = get_optional_w_or_b(layer_context, &layer.inputs[1])?;

        Ok(Box::new(Self {
            name: layer.name.clone(),
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            initializer_a,
            initializer_b,
        }))
    }
}

#[cfg(test)]
mod soundness_tests {
    use crate::circuit_functions::gadgets::LogupRangeCheckContext;
    use crate::circuit_functions::hints::build_logup_hint_registry;
    use expander_compiler::frontend::*;

    const DIVISOR: u32 = 3;

    declare_circuit!(DivBy3Circuit {
        dividend: Variable,
        expected_quotient: PublicVariable
    });

    impl Define<BN254Config> for DivBy3Circuit<Variable> {
        fn define<Builder: RootAPI<BN254Config>>(&self, api: &mut Builder) {
            let divisor_val = DIVISOR;

            let mut logup_ctx = LogupRangeCheckContext::new_default();
            logup_ctx.init::<BN254Config, Builder>(api);

            let divisor_var = api.constant(divisor_val);
            let divisor_minus_one = api.constant(divisor_val - 1);
            let remainder_bound_bits = (32 - (divisor_val - 1).leading_zeros()) as usize;

            let q = api.unconstrained_int_div(self.dividend, divisor_val);
            let r = api.unconstrained_mod(self.dividend, divisor_val);

            let prod = api.mul(divisor_var, q);
            let reconstructed = api.add(prod, r);
            api.assert_is_equal(self.dividend, reconstructed);

            let diff = api.sub(divisor_minus_one, r);
            logup_ctx
                .range_check::<BN254Config, Builder>(api, diff, remainder_bound_bits)
                .expect("range check failed");

            logup_ctx.finalize::<BN254Config, Builder>(api);
            api.assert_is_equal(q, self.expected_quotient);
        }
    }

    type F = CircuitField<BN254Config>;

    #[test]
    fn test_div_valid_10_by_3() {
        let mut hint_registry = build_logup_hint_registry::<F>();

        let compile_result: CompileResult<BN254Config> =
            compile(&DivBy3Circuit::default(), CompileOptions::default())
                .expect("Compilation failed");

        let assignment = DivBy3Circuit::<F> {
            dividend: F::from(10u32),
            expected_quotient: F::from(3u32),
        };

        let witness = compile_result
            .witness_solver
            .solve_witness_with_hints(&assignment, &mut hint_registry)
            .expect("Witness solving failed");

        let output = compile_result.layered_circuit.run(&witness);
        assert!(output.iter().all(|&x| x), "10 / 3 = 3 should pass");
    }

    #[test]
    fn test_div_wrong_quotient_rejected() {
        let mut hint_registry = build_logup_hint_registry::<F>();

        let compile_result: CompileResult<BN254Config> =
            compile(&DivBy3Circuit::default(), CompileOptions::default())
                .expect("Compilation failed");

        let assignment = DivBy3Circuit::<F> {
            dividend: F::from(10u32),
            expected_quotient: F::from(2u32),
        };

        let witness = compile_result
            .witness_solver
            .solve_witness_with_hints(&assignment, &mut hint_registry)
            .expect("Witness solving failed");

        let output = compile_result.layered_circuit.run(&witness);
        assert!(
            !output.iter().all(|&x| x),
            "Claiming 10 / 3 = 2 should be rejected"
        );
    }

    const DIVISOR_7: u32 = 7;

    declare_circuit!(DivBy7Circuit {
        dividend: Variable,
        expected_quotient: PublicVariable
    });

    impl Define<BN254Config> for DivBy7Circuit<Variable> {
        fn define<Builder: RootAPI<BN254Config>>(&self, api: &mut Builder) {
            let divisor_val = DIVISOR_7;

            let mut logup_ctx = LogupRangeCheckContext::new_default();
            logup_ctx.init::<BN254Config, Builder>(api);

            let divisor_var = api.constant(divisor_val);
            let divisor_minus_one = api.constant(divisor_val - 1);
            let remainder_bound_bits = (32 - (divisor_val - 1).leading_zeros()) as usize;

            let q = api.unconstrained_int_div(self.dividend, divisor_val);
            let r = api.unconstrained_mod(self.dividend, divisor_val);

            let prod = api.mul(divisor_var, q);
            let reconstructed = api.add(prod, r);
            api.assert_is_equal(self.dividend, reconstructed);

            let diff = api.sub(divisor_minus_one, r);
            logup_ctx
                .range_check::<BN254Config, Builder>(api, diff, remainder_bound_bits)
                .expect("range check failed");

            logup_ctx.finalize::<BN254Config, Builder>(api);
            api.assert_is_equal(q, self.expected_quotient);
        }
    }

    #[test]
    fn test_div_valid_100_by_7() {
        let mut hint_registry = build_logup_hint_registry::<F>();

        let compile_result: CompileResult<BN254Config> =
            compile(&DivBy7Circuit::default(), CompileOptions::default())
                .expect("Compilation failed");

        let assignment = DivBy7Circuit::<F> {
            dividend: F::from(100u32),
            expected_quotient: F::from(14u32),
        };

        let witness = compile_result
            .witness_solver
            .solve_witness_with_hints(&assignment, &mut hint_registry)
            .expect("Witness solving failed");

        let output = compile_result.layered_circuit.run(&witness);
        assert!(output.iter().all(|&x| x), "100 / 7 = 14 should pass");
    }

    #[test]
    fn test_div_exact_division_21_by_7() {
        let mut hint_registry = build_logup_hint_registry::<F>();

        let compile_result: CompileResult<BN254Config> =
            compile(&DivBy7Circuit::default(), CompileOptions::default())
                .expect("Compilation failed");

        let assignment = DivBy7Circuit::<F> {
            dividend: F::from(21u32),
            expected_quotient: F::from(3u32),
        };

        let witness = compile_result
            .witness_solver
            .solve_witness_with_hints(&assignment, &mut hint_registry)
            .expect("Witness solving failed");

        let output = compile_result.layered_circuit.run(&witness);
        assert!(output.iter().all(|&x| x), "21 / 7 = 3 should pass");
    }
}
