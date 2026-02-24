use std::ops::Neg;

use ndarray::{Array1, ArrayD, ArrayView1, Axis, Ix1, IxDyn, concatenate};

use expander_compiler::frontend::{
    BN254Config, CircuitField, Config, Define, FieldArith, RootAPI, Variable, declare_circuit,
};

use crate::circuit_functions::CircuitError;
use crate::circuit_functions::hints::build_logup_hint_registry;
use crate::circuit_functions::utils::ArrayConversionError;
use crate::circuit_functions::utils::build_layers::build_layers;
use crate::circuit_functions::utils::onnx_model::{CircuitParams, InputData, OutputData, WANDB};
use crate::circuit_functions::utils::shaping::get_inputs;
use crate::circuit_functions::utils::tensor_ops::{
    convert_val_to_field_element, get_nd_circuit_inputs,
};
use crate::io::io_reader::onnx_context::OnnxContext;
use crate::io::io_reader::{FileReader, IOReader};
use crate::runner::errors::RunError;
use crate::runner::main_runner::{
    ConfigurableCircuit, load_circuit_from_bytes, load_witness_solver_from_bytes, prove_from_bytes,
    serialize_witness, solve_and_validate_witness, verify_from_bytes, witness_from_request,
};
use crate::runner::schema::{WitnessBundle, WitnessRequest};
use crate::runner::verify_extract::{VerifiedOutput, verify_and_extract_from_bytes};

declare_circuit!(Circuit {
    input_arr: [PublicVariable],
    outputs: [PublicVariable],
    dummy: [Variable; 2],
    scale_base: [PublicVariable; 1],
    scale_exponent: [PublicVariable; 1],
});

/// Panics if OnnxContext is not populated. Callers must call
/// `OnnxContext::set_all` (or the individual setters) before any
/// compilation or evaluation path that invokes `define`/`try_define`.
impl<C: Config> Define<C> for Circuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        if let Err(e) = self.try_define(api) {
            panic!("Circuit definition failed: {e}");
        }
    }
}

impl Circuit<Variable> {
    fn try_define<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &mut Builder,
    ) -> Result<(), CircuitError> {
        let params = OnnxContext::get_params()?;
        let architecture = OnnxContext::get_architecture()?;
        let w_and_b = if params.weights_as_inputs {
            WANDB { w_and_b: vec![] }
        } else {
            OnnxContext::get_wandb()?
        };

        if architecture.architecture.is_empty() {
            return Err(CircuitError::EmptyArchitecture);
        }

        let mut out = get_inputs(&self.input_arr, &params.inputs)?;

        let layers = build_layers::<C, Builder>(&params, &architecture, &w_and_b)?;

        for layer in &layers {
            let (keys, value) = layer.apply(api, &out)?;
            for key in keys {
                out.insert(key, value.clone());
            }
        }

        if params.outputs.is_empty() {
            return Err(CircuitError::Other(
                "circuit has no declared outputs".into(),
            ));
        }

        let flatten_shape: Vec<usize> = vec![params.effective_output_dims()];
        let mut flat_outputs: Vec<Array1<Variable>> = Vec::new();

        for output_info in &params.outputs {
            let output_name = &output_info.name;

            let output = out.get(output_name).ok_or_else(|| {
                CircuitError::Other(format!("Missing output '{output_name}' in map"))
            })?;

            flat_outputs.push(Array1::from_iter(output.iter().copied()));
        }

        let combined_output = concatenate(
            Axis(0),
            &flat_outputs
                .iter()
                .map(ndarray::ArrayBase::view)
                .collect::<Vec<ArrayView1<Variable>>>(),
        )
        .map_err(|e| CircuitError::Other(format!("Concatenation error: {e}")))?;

        let combined_output = combined_output
            .into_shape_with_order(IxDyn(&flatten_shape))
            .map_err(ArrayConversionError::ShapeError)?;

        if combined_output.len() != self.outputs.len() {
            return Err(CircuitError::Other(format!(
                "output length mismatch: circuit has {} outputs but computed {} values (total_output_dims={})",
                self.outputs.len(),
                combined_output.len(),
                params.total_output_dims()
            )));
        }
        for (&out, &combined) in self.outputs.iter().zip(combined_output.iter()) {
            api.assert_is_equal(out, combined);
        }

        // Constant placeholders required by the circuit framework to anchor
        // the constraint system when no other fixed-value wires are present.
        api.assert_is_equal(self.dummy[0], 1);
        api.assert_is_equal(self.dummy[1], 1);

        api.assert_is_equal(self.scale_base[0], params.scale_base);
        api.assert_is_equal(self.scale_exponent[0], params.scale_exponent);

        Ok(())
    }
}

impl ConfigurableCircuit for Circuit<Variable> {
    fn configure(&mut self) -> Result<(), RunError> {
        let params = OnnxContext::get_params()?;

        self.outputs = vec![Variable::default(); params.effective_output_dims()];
        self.input_arr = vec![Variable::default(); params.effective_input_dims()];

        Ok(())
    }
}

fn init_circuit_fields<C: Config>(
    assignment: &mut Circuit<CircuitField<C>>,
    params: &CircuitParams,
) {
    assignment.dummy[0] = CircuitField::<C>::from(1);
    assignment.dummy[1] = CircuitField::<C>::from(1);
    assignment.scale_base[0] = CircuitField::<C>::from(params.scale_base);
    assignment.scale_exponent[0] = CircuitField::<C>::from(params.scale_exponent);
}

pub fn apply_input_data<C: Config>(
    data: &InputData,
    mut assignment: Circuit<CircuitField<C>>,
    params: &CircuitParams,
) -> Result<Circuit<CircuitField<C>>, RunError> {
    init_circuit_fields::<C>(&mut assignment, params);

    let input_dims: &[usize] = &[params.effective_input_dims()];

    let arr: ArrayD<CircuitField<C>> = get_nd_circuit_inputs::<C>(&data.input, input_dims)
        .map_err(|e| RunError::Json(format!("Invalid input shape: {e}")))?;

    let flat: Vec<CircuitField<C>> = arr
        .into_dimensionality::<Ix1>()
        .map_err(|_| RunError::Json("Expected a 1-D input array".into()))?
        .to_vec();

    assignment.input_arr = flat;

    Ok(assignment)
}

pub fn apply_output_data<C: Config>(
    data: &OutputData,
    mut assignment: Circuit<CircuitField<C>>,
    params: &CircuitParams,
) -> Result<Circuit<CircuitField<C>>, RunError> {
    init_circuit_fields::<C>(&mut assignment, params);

    let output_dims: &[usize] = &[params.effective_output_dims()];

    let arr: ArrayD<CircuitField<C>> = get_nd_circuit_inputs::<C>(&data.output, output_dims)
        .map_err(|e| RunError::Json(format!("Invalid output shape: {e}")))?;

    let flat: Vec<CircuitField<C>> = arr
        .into_dimensionality::<Ix1>()
        .map_err(|_| RunError::Json("Expected a 1-D output array".into()))?
        .to_vec();

    assignment.outputs = flat;
    Ok(assignment)
}

fn apply_values_common<C: Config>(
    input: serde_json::Value,
    output: serde_json::Value,
    assignment: Circuit<CircuitField<C>>,
    params: &CircuitParams,
) -> Result<Circuit<CircuitField<C>>, RunError> {
    let input_data: InputData =
        serde_json::from_value(input).map_err(|e| RunError::Json(format!("{e}")))?;
    let output_data: OutputData =
        serde_json::from_value(output).map_err(|e| RunError::Json(format!("{e}")))?;
    let assignment = apply_input_data::<C>(&input_data, assignment, params)?;
    apply_output_data::<C>(&output_data, assignment, params)
}

impl<C: Config> IOReader<Circuit<CircuitField<C>>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        assignment: Circuit<CircuitField<C>>,
    ) -> Result<Circuit<CircuitField<C>>, RunError> {
        let data: InputData =
            <FileReader as IOReader<Circuit<_>, C>>::read_data_from_json::<InputData>(file_path)?;
        let params = OnnxContext::get_params()?;
        apply_input_data::<C>(&data, assignment, &params)
    }

    fn read_outputs(
        &mut self,
        file_path: &str,
        assignment: Circuit<CircuitField<C>>,
    ) -> Result<Circuit<CircuitField<C>>, RunError> {
        let data: OutputData =
            <FileReader as IOReader<Circuit<_>, C>>::read_data_from_json::<OutputData>(file_path)?;
        let params = OnnxContext::get_params()?;
        apply_output_data::<C>(&data, assignment, &params)
    }

    fn apply_values(
        &mut self,
        input: serde_json::Value,
        output: serde_json::Value,
        assignment: Circuit<CircuitField<C>>,
    ) -> Result<Circuit<CircuitField<C>>, RunError> {
        let params = OnnxContext::get_params()?;
        apply_values_common::<C>(input, output, assignment, &params)
    }

    fn get_path(&self) -> &str {
        &self.path
    }
}

/// In-memory IOReader that only supports [`IOReader::apply_values`].
///
/// File-based methods ([`IOReader::read_inputs`], [`IOReader::read_outputs`])
/// return [`RunError::Unsupported`]. [`IOReader::get_path`] returns an empty string.
///
/// When `params` is `Some`, the instance uses per-invocation circuit parameters
/// instead of reading from `OnnxContext` globals, enabling concurrent witness
/// generation for different circuits.
pub struct ValueReader {
    pub params: Option<CircuitParams>,
}

impl<C: Config> IOReader<Circuit<CircuitField<C>>, C> for ValueReader {
    fn read_inputs(
        &mut self,
        _file_path: &str,
        _assignment: Circuit<CircuitField<C>>,
    ) -> Result<Circuit<CircuitField<C>>, RunError> {
        Err(RunError::Unsupported(
            "ValueReader does not support file-based read_inputs".into(),
        ))
    }

    fn read_outputs(
        &mut self,
        _file_path: &str,
        _assignment: Circuit<CircuitField<C>>,
    ) -> Result<Circuit<CircuitField<C>>, RunError> {
        Err(RunError::Unsupported(
            "ValueReader does not support file-based read_outputs".into(),
        ))
    }

    fn apply_values(
        &mut self,
        input: serde_json::Value,
        output: serde_json::Value,
        assignment: Circuit<CircuitField<C>>,
    ) -> Result<Circuit<CircuitField<C>>, RunError> {
        let owned;
        let params = match self.params {
            Some(ref p) => p,
            None => {
                owned = OnnxContext::get_params()?;
                &owned
            }
        };
        apply_values_common::<C>(input, output, assignment, params)
    }

    fn get_path(&self) -> &str {
        ""
    }
}

pub fn witness_bn254(req: &WitnessRequest, compress: bool) -> Result<WitnessBundle, RunError> {
    let mut reader = ValueReader {
        params: req.metadata.clone(),
    };
    witness_from_request::<BN254Config, ValueReader, Circuit<CircuitField<BN254Config>>>(
        req,
        &mut reader,
        compress,
    )
}

pub fn witness_bn254_from_f64(
    circuit_bytes: &[u8],
    solver_bytes: &[u8],
    params: &CircuitParams,
    activations: &[f64],
    initializers: &[(Vec<f64>, Vec<usize>)],
    compress: bool,
) -> Result<WitnessBundle, RunError> {
    witness_from_f64::<BN254Config>(
        circuit_bytes,
        solver_bytes,
        params,
        activations,
        initializers,
        compress,
    )
}

fn quantize_f64_to_field<C: Config>(val: f64, scale: f64) -> CircuitField<C> {
    let scaled = val * scale;
    let truncated = scaled as i64;
    convert_val_to_field_element::<C>(truncated)
}

fn witness_from_f64<C: Config>(
    circuit_bytes: &[u8],
    solver_bytes: &[u8],
    params: &CircuitParams,
    activations: &[f64],
    initializers: &[(Vec<f64>, Vec<usize>)],
    compress: bool,
) -> Result<WitnessBundle, RunError> {
    let alpha = (params.scale_base as f64).powi(params.scale_exponent as i32);
    let alpha_sq = alpha * alpha;

    let num_activation_entries = params.inputs.len() - initializers.len();
    let expected_activation_elems: usize = params.inputs[..num_activation_entries]
        .iter()
        .map(|io| io.shape.iter().product::<usize>())
        .sum();
    if activations.len() != expected_activation_elems {
        return Err(RunError::Witness(format!(
            "activation length mismatch: expected {expected_activation_elems}, got {}",
            activations.len()
        )));
    }

    let mut input_arr: Vec<CircuitField<C>> = Vec::with_capacity(params.effective_input_dims());

    for &v in activations {
        input_arr.push(quantize_f64_to_field::<C>(v, alpha));
    }

    for (idx, (values, _shape)) in initializers.iter().enumerate() {
        let io = &params.inputs[num_activation_entries + idx];
        let scale = if io.shape.len() == 1 { alpha_sq } else { alpha };
        for &v in values {
            input_arr.push(quantize_f64_to_field::<C>(v, scale));
        }
    }

    if input_arr.len() != params.effective_input_dims() {
        return Err(RunError::Witness(format!(
            "total input length mismatch: expected {}, got {}",
            params.effective_input_dims(),
            input_arr.len()
        )));
    }

    let layered_circuit = load_circuit_from_bytes::<C>(circuit_bytes)?;

    let num_outputs = params.effective_output_dims();
    let private_inputs = vec![CircuitField::<C>::from(1u32), CircuitField::<C>::from(1u32)];
    let mut public_inputs: Vec<CircuitField<C>> =
        Vec::with_capacity(input_arr.len() + num_outputs + 2);
    public_inputs.extend_from_slice(&input_arr);
    public_inputs.extend(std::iter::repeat_n(CircuitField::<C>::zero(), num_outputs));
    public_inputs.push(CircuitField::<C>::from(params.scale_base));
    public_inputs.push(CircuitField::<C>::from(params.scale_exponent));

    let constraint_values = layered_circuit.eval_constraint_values(private_inputs, &public_inputs);

    if constraint_values.len() < num_outputs {
        return Err(RunError::Witness(format!(
            "constraint values length {}: expected at least {num_outputs}",
            constraint_values.len()
        )));
    }

    let computed_outputs: Vec<CircuitField<C>> = constraint_values[..num_outputs]
        .iter()
        .map(|v| v.neg())
        .collect();

    let mut assignment = Circuit::<CircuitField<C>>::default();
    assignment.input_arr = input_arr;
    assignment.outputs = computed_outputs;
    assignment.dummy[0] = CircuitField::<C>::from(1u32);
    assignment.dummy[1] = CircuitField::<C>::from(1u32);
    assignment.scale_base[0] = CircuitField::<C>::from(params.scale_base);
    assignment.scale_exponent[0] = CircuitField::<C>::from(params.scale_exponent);

    let witness_solver = load_witness_solver_from_bytes::<C>(solver_bytes)?;
    let hint_registry = build_logup_hint_registry::<CircuitField<C>>();

    let witness = solve_and_validate_witness(
        &witness_solver,
        &layered_circuit,
        &hint_registry,
        &assignment,
    )?;

    let witness_bytes = serialize_witness::<C>(&witness, compress)?;

    Ok(WitnessBundle {
        witness: witness_bytes,
        output_data: None,
    })
}

pub fn prove_bn254(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    compress: bool,
) -> Result<Vec<u8>, RunError> {
    prove_from_bytes::<BN254Config>(circuit_bytes, witness_bytes, compress)
}

pub fn verify_bn254(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
) -> Result<bool, RunError> {
    verify_from_bytes::<BN254Config>(circuit_bytes, witness_bytes, proof_bytes)
}

pub fn compile_bn254(
    circuit_path: &str,
    compress: bool,
    metadata: Option<CircuitParams>,
) -> Result<(), RunError> {
    crate::runner::main_runner::run_compile_and_serialize::<BN254Config, Circuit<Variable>>(
        circuit_path,
        compress,
        metadata,
    )
}

pub fn verify_and_extract_bn254(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
    num_inputs: usize,
    expected_inputs: Option<&[f64]>,
) -> Result<VerifiedOutput, RunError> {
    verify_and_extract_from_bytes::<BN254Config>(
        circuit_bytes,
        witness_bytes,
        proof_bytes,
        num_inputs,
        expected_inputs,
    )
}
