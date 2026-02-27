use ndarray::{Array1, ArrayD, ArrayView1, Axis, Ix1, IxDyn, concatenate};

use std::io::Cursor;

use arith::SimdField;
use expander_compiler::expander_binary::executor;
use expander_compiler::expander_circuit;
use expander_compiler::frontend::{
    BN254Config, ChallengeField, CircuitField, Config, Define, FieldArith, RootAPI, Variable,
    declare_circuit,
};
use expander_compiler::gkr_engine::{GKREngine, MPIConfig};
use expander_compiler::serdes::ExpSerde;
use jstprove_direct_builder::DirectBuilder;

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
    serialize_witness, verify_from_bytes, witness_from_request,
};
use crate::runner::schema::{WitnessBundle, WitnessRequest};
use crate::runner::verify_extract::{
    ExtractedOutput, VerifiedOutput, extract_outputs_from_witness, verify_and_extract_from_bytes,
};

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

        api.set_outputs(combined_output.iter().copied().collect());

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

/// # Errors
/// Returns `RunError` on invalid input shape or JSON parsing failure.
pub fn apply_input_data<C: Config>(
    data: &InputData,
    mut assignment: Circuit<CircuitField<C>>,
    params: &CircuitParams,
) -> Result<Circuit<CircuitField<C>>, RunError> {
    init_circuit_fields::<C>(&mut assignment, params);

    let expected_size = params.effective_input_dims();
    let input_dims: &[usize] = &[expected_size];

    let arr: ArrayD<CircuitField<C>> = get_nd_circuit_inputs::<C>(&data.input, input_dims)
        .map_err(|e| {
            let shapes: Vec<_> = params
                .inputs
                .iter()
                .filter(|io| !io.shape.is_empty())
                .map(|io| format!("{}: {:?}", io.name, io.shape))
                .collect();
            RunError::Json(format!(
                "input size mismatch: model expects {} elements (shapes: [{}]) — {e}",
                expected_size,
                shapes.join(", "),
            ))
        })?;

    let flat: Vec<CircuitField<C>> = arr
        .into_dimensionality::<Ix1>()
        .map_err(|_| RunError::Json("Expected a 1-D input array".into()))?
        .to_vec();

    assignment.input_arr = flat;

    Ok(assignment)
}

/// # Errors
/// Returns `RunError` on invalid output shape or JSON parsing failure.
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
    input: rmpv::Value,
    output: rmpv::Value,
    assignment: Circuit<CircuitField<C>>,
    params: &CircuitParams,
) -> Result<Circuit<CircuitField<C>>, RunError> {
    let input_data: InputData =
        rmpv::ext::from_value(input).map_err(|e| RunError::Json(format!("{e}")))?;
    let output_data: OutputData =
        rmpv::ext::from_value(output).map_err(|e| RunError::Json(format!("{e}")))?;
    let assignment = apply_input_data::<C>(&input_data, assignment, params)?;
    apply_output_data::<C>(&output_data, assignment, params)
}

impl<C: Config> IOReader<Circuit<CircuitField<C>>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        assignment: Circuit<CircuitField<C>>,
    ) -> Result<Circuit<CircuitField<C>>, RunError> {
        let data: InputData = <FileReader as IOReader<Circuit<_>, C>>::read_data_from_msgpack::<
            InputData,
        >(file_path)?;
        let params = OnnxContext::get_params()?;
        apply_input_data::<C>(&data, assignment, &params)
    }

    fn read_outputs(
        &mut self,
        file_path: &str,
        assignment: Circuit<CircuitField<C>>,
    ) -> Result<Circuit<CircuitField<C>>, RunError> {
        let data: OutputData = <FileReader as IOReader<Circuit<_>, C>>::read_data_from_msgpack::<
            OutputData,
        >(file_path)?;
        let params = OnnxContext::get_params()?;
        apply_output_data::<C>(&data, assignment, &params)
    }

    fn apply_values(
        &mut self,
        input: rmpv::Value,
        output: rmpv::Value,
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
        input: rmpv::Value,
        output: rmpv::Value,
        assignment: Circuit<CircuitField<C>>,
    ) -> Result<Circuit<CircuitField<C>>, RunError> {
        let owned;
        let params = if let Some(ref p) = self.params {
            p
        } else {
            owned = OnnxContext::get_params()?;
            &owned
        };
        apply_values_common::<C>(input, output, assignment, params)
    }

    #[allow(clippy::unused_self, clippy::unnecessary_literal_bound)]
    fn get_path(&self) -> &str {
        ""
    }
}

/// # Errors
/// Returns `RunError` on witness generation failure.
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

/// # Errors
/// Returns `RunError` on witness generation or activation mismatch.
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

#[allow(clippy::cast_possible_truncation)]
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
    #[allow(clippy::cast_possible_wrap)]
    let alpha = (f64::from(params.scale_base)).powi(params.scale_exponent as i32);
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
    let witness_solver = load_witness_solver_from_bytes::<C>(solver_bytes)?;
    let hint_registry = build_logup_hint_registry::<CircuitField<C>>();
    let num_outputs = params.effective_output_dims();

    let mut assignment = Circuit::<CircuitField<C>> {
        input_arr: input_arr.clone(),
        outputs: vec![CircuitField::<C>::zero(); num_outputs],
        ..Default::default()
    };
    assignment.dummy[0] = CircuitField::<C>::from(1u32);
    assignment.dummy[1] = CircuitField::<C>::from(1u32);
    assignment.scale_base[0] = CircuitField::<C>::from(params.scale_base);
    assignment.scale_exponent[0] = CircuitField::<C>::from(params.scale_exponent);

    let probe_witness = witness_solver
        .solve_witness_with_hints(&assignment, &hint_registry)
        .map_err(|e| RunError::Witness(format!("probe pass: {e:?}")))?;

    let (private_inputs, public_inputs) = probe_witness
        .iter_scalar()
        .next()
        .ok_or_else(|| RunError::Witness("empty probe witness".into()))?;

    let (actual_outputs, _) =
        layered_circuit.eval_with_public_inputs(private_inputs, &public_inputs);

    if actual_outputs.len() < num_outputs {
        return Err(RunError::Witness(format!(
            "circuit actual outputs length {}: expected at least {num_outputs} (expected_num_output_zeroes={}, num_actual_outputs={})",
            actual_outputs.len(),
            layered_circuit.expected_num_output_zeroes,
            layered_circuit.num_actual_outputs,
        )));
    }

    let computed_outputs: Vec<CircuitField<C>> = actual_outputs[..num_outputs].to_vec();

    assignment.input_arr = input_arr;
    assignment.outputs = computed_outputs;

    let witness = witness_solver
        .solve_witness_with_hints(&assignment, &hint_registry)
        .map_err(|e| RunError::Witness(format!("final pass: {e:?}")))?;

    let witness_bytes = serialize_witness::<C>(&witness, compress)?;

    Ok(WitnessBundle {
        witness: witness_bytes,
        output_data: None,
        version: Some(crate::runner::version::jstprove_artifact_version()),
    })
}

/// # Errors
/// Returns `RunError` on proof generation failure.
pub fn prove_bn254(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    compress: bool,
) -> Result<Vec<u8>, RunError> {
    prove_from_bytes::<BN254Config>(circuit_bytes, witness_bytes, compress)
}

/// # Errors
/// Returns `RunError` on verification failure.
pub fn verify_bn254(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
) -> Result<bool, RunError> {
    verify_from_bytes::<BN254Config>(circuit_bytes, witness_bytes, proof_bytes)
}

/// # Errors
/// Returns `RunError` on compilation or serialization failure.
pub fn compile_bn254(
    circuit_path: &str,
    compress: bool,
    metadata: Option<CircuitParams>,
    fast_compile: bool,
) -> Result<(), RunError> {
    if fast_compile {
        let params = match metadata {
            Some(p) => p,
            None => OnnxContext::get_params()?,
        };
        compile_bn254_direct_to_path(circuit_path, compress, &params)
    } else {
        crate::runner::main_runner::run_compile_and_serialize::<BN254Config, Circuit<Variable>>(
            circuit_path,
            compress,
            metadata,
        )
    }
}

/// # Errors
/// Returns `RunError` on DirectBuilder compilation or serialization failure.
pub fn compile_bn254_direct_to_path(
    circuit_path: &str,
    compress: bool,
    params: &CircuitParams,
) -> Result<(), RunError> {
    use crate::proof_system::ProofSystem;
    use crate::runner::schema::CompiledCircuit;
    use crate::runner::version::jstprove_artifact_version;

    let n = params.effective_input_dims();
    let dummy_inputs = vec![CircuitField::<BN254Config>::zero(); n];
    let (circuit, _) = direct_build_from_fields(params, &dummy_inputs)?;

    let mut circuit_buf = Vec::new();
    circuit
        .serialize_into(&mut circuit_buf)
        .map_err(|e| RunError::Serialize(format!("direct circuit: {e:?}")))?;

    let mut params_direct = params.clone();
    params_direct.proof_system = ProofSystem::DirectBuilder;

    let bundle = CompiledCircuit {
        circuit: circuit_buf,
        witness_solver: vec![],
        metadata: Some(params_direct),
        version: Some(jstprove_artifact_version()),
    };
    crate::runner::main_runner::write_circuit_bundle(circuit_path, &bundle, compress)
}

/// # Errors
/// Returns `RunError` on witness generation failure.
pub fn witness_bn254_from_f64_direct(
    params: &CircuitParams,
    activations: &[f64],
    initializers: &[(Vec<f64>, Vec<usize>)],
    compress: bool,
) -> Result<WitnessBundle, RunError> {
    use crate::runner::version::jstprove_artifact_version;

    let (_, witness_private) = compile_and_witness_bn254_direct(params, activations, initializers)?;

    let mut buf = Vec::new();
    witness_private
        .serialize_into(&mut buf)
        .map_err(|e| RunError::Serialize(format!("direct witness: {e:?}")))?;
    let witness_bytes = jstprove_io::maybe_compress_bytes(buf, compress)
        .map_err(|e| RunError::Serialize(format!("zstd compress: {e}")))?;

    Ok(WitnessBundle {
        witness: witness_bytes,
        output_data: None,
        version: Some(jstprove_artifact_version()),
    })
}

/// # Errors
/// Returns `RunError` on proof generation failure.
pub fn prove_bn254_direct_circuit(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    compress: bool,
) -> Result<Vec<u8>, RunError> {
    use crate::runner::main_runner::auto_decompress_bytes;

    let circuit_data = auto_decompress_bytes(circuit_bytes)?;
    let mut circuit =
        expander_circuit::Circuit::<FC<BN254Config>>::deserialize_from(Cursor::new(&*circuit_data))
            .map_err(|e| RunError::Deserialize(format!("direct circuit: {e:?}")))?;

    let witness_data = auto_decompress_bytes(witness_bytes)?;
    let witness_private =
        Vec::<CircuitField<BN254Config>>::deserialize_from(Cursor::new(&*witness_data))
            .map_err(|e| RunError::Deserialize(format!("direct witness: {e:?}")))?;

    let proof = prove_bn254_direct(&mut circuit, &witness_private)?;
    jstprove_io::maybe_compress_bytes(proof, compress)
        .map_err(|e| RunError::Serialize(format!("zstd compress: {e}")))
}

/// # Errors
/// Returns `RunError` on verification failure.
pub fn verify_bn254_direct_circuit(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
) -> Result<bool, RunError> {
    use crate::runner::main_runner::auto_decompress_bytes;

    let circuit_data = auto_decompress_bytes(circuit_bytes)?;
    let mut circuit =
        expander_circuit::Circuit::<FC<BN254Config>>::deserialize_from(Cursor::new(&*circuit_data))
            .map_err(|e| RunError::Deserialize(format!("direct circuit: {e:?}")))?;

    let witness_data = auto_decompress_bytes(witness_bytes)?;
    let witness_private =
        Vec::<CircuitField<BN254Config>>::deserialize_from(Cursor::new(&*witness_data))
            .map_err(|e| RunError::Deserialize(format!("direct witness: {e:?}")))?;

    let proof_data = auto_decompress_bytes(proof_bytes)?;
    verify_bn254_direct(&mut circuit, &witness_private, &proof_data)
}

/// # Errors
/// Returns `RunError` on deserialization or extraction failure.
pub fn extract_outputs_bn254(
    witness_bytes: &[u8],
    num_model_inputs: usize,
) -> Result<ExtractedOutput, RunError> {
    extract_outputs_from_witness(witness_bytes, num_model_inputs)
}

/// # Errors
/// Returns `RunError` on verification or output extraction failure.
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

type FC<C> = <C as GKREngine>::FieldConfig;
type DirectBuildResult = (
    expander_circuit::Circuit<FC<BN254Config>>,
    Vec<CircuitField<BN254Config>>,
);

fn direct_build_from_fields(
    params: &CircuitParams,
    input_arr_vals: &[CircuitField<BN254Config>],
) -> Result<DirectBuildResult, RunError> {
    let num_outputs = params.effective_output_dims();
    let hint_registry = build_logup_hint_registry::<CircuitField<BN254Config>>();

    let build = |output_vals: &[CircuitField<BN254Config>]| {
        let mut all: Vec<CircuitField<BN254Config>> =
            Vec::with_capacity(input_arr_vals.len() + num_outputs + 4);
        all.extend_from_slice(input_arr_vals);
        all.extend_from_slice(output_vals);
        all.push(CircuitField::<BN254Config>::from(1u32));
        all.push(CircuitField::<BN254Config>::from(1u32));
        all.push(CircuitField::<BN254Config>::from(params.scale_base));
        all.push(CircuitField::<BN254Config>::from(params.scale_exponent));
        let n = input_arr_vals.len();
        let circuit_struct = Circuit {
            input_arr: (1..=n).map(Variable::from).collect(),
            outputs: (n + 1..=n + num_outputs).map(Variable::from).collect(),
            dummy: [
                Variable::from(n + num_outputs + 1),
                Variable::from(n + num_outputs + 2),
            ],
            scale_base: [Variable::from(n + num_outputs + 3)],
            scale_exponent: [Variable::from(n + num_outputs + 4)],
        };
        (all, circuit_struct)
    };

    let dummy_outputs = vec![CircuitField::<BN254Config>::zero(); num_outputs];
    let (probe_vals, probe_circuit) = build(&dummy_outputs);
    let hint_reg_probe = build_logup_hint_registry::<CircuitField<BN254Config>>();
    let mut probe_builder = DirectBuilder::<BN254Config>::new(&probe_vals, hint_reg_probe);
    probe_circuit
        .try_define::<BN254Config, _>(&mut probe_builder)
        .map_err(|e| RunError::Compile(format!("circuit definition (probe): {e}")))?;
    let computed_outputs = probe_builder.output_witness_values();

    let (final_vals, final_circuit) = build(&computed_outputs);
    let mut builder = DirectBuilder::<BN254Config>::new(&final_vals, hint_registry);
    final_circuit
        .try_define::<BN254Config, _>(&mut builder)
        .map_err(|e| RunError::Compile(format!("circuit definition: {e}")))?;
    Ok(builder.finalize())
}

#[allow(clippy::missing_errors_doc, clippy::cast_possible_wrap)]
pub fn compile_and_witness_bn254_direct(
    params: &CircuitParams,
    activations: &[f64],
    initializers: &[(Vec<f64>, Vec<usize>)],
) -> Result<DirectBuildResult, RunError> {
    let alpha = (f64::from(params.scale_base)).powi(params.scale_exponent as i32);
    let alpha_sq = alpha * alpha;
    let num_activation_entries = params
        .inputs
        .len()
        .checked_sub(initializers.len())
        .ok_or_else(|| {
            RunError::Witness(format!(
                "initializers count ({}) exceeds params.inputs count ({})",
                initializers.len(),
                params.inputs.len()
            ))
        })?;
    let expected: usize = params.inputs[..num_activation_entries]
        .iter()
        .map(|io| io.shape.iter().product::<usize>())
        .sum();
    if activations.len() != expected {
        return Err(RunError::Witness(format!(
            "activation length mismatch: expected {expected}, got {}",
            activations.len()
        )));
    }
    let mut input_arr_vals: Vec<CircuitField<BN254Config>> =
        Vec::with_capacity(params.effective_input_dims());
    for &v in activations {
        input_arr_vals.push(quantize_f64_to_field::<BN254Config>(v, alpha));
    }
    for (idx, (values, _shape)) in initializers.iter().enumerate() {
        let io = &params.inputs[num_activation_entries + idx];
        let scale = if io.shape.len() == 1 { alpha_sq } else { alpha };
        for &v in values {
            input_arr_vals.push(quantize_f64_to_field::<BN254Config>(v, scale));
        }
    }

    let expected_total = params.effective_input_dims();
    if input_arr_vals.len() != expected_total {
        return Err(RunError::Witness(format!(
            "input dimension mismatch: expected {expected_total}, got {}",
            input_arr_vals.len()
        )));
    }

    direct_build_from_fields(params, &input_arr_vals)
}

#[allow(clippy::missing_errors_doc)]
pub fn compile_and_witness_bn254_from_fields(
    params: &CircuitParams,
    input_arr_vals: &[CircuitField<BN254Config>],
) -> Result<DirectBuildResult, RunError> {
    let expected = params.effective_input_dims();
    if input_arr_vals.len() != expected {
        return Err(RunError::Witness(format!(
            "input field elements length mismatch: expected {expected}, got {}",
            input_arr_vals.len()
        )));
    }
    direct_build_from_fields(params, input_arr_vals)
}

#[allow(clippy::missing_errors_doc)]
pub fn prove_bn254_direct(
    circuit: &mut expander_circuit::Circuit<FC<BN254Config>>,
    witness_private: &[CircuitField<BN254Config>],
) -> Result<Vec<u8>, RunError> {
    type Simd = <<BN254Config as GKREngine>::FieldConfig as expander_compiler::gkr_engine::FieldEngine>::SimdCircuitField;
    let pack_size = <Simd as SimdField>::PACK_SIZE;
    let mut buf = vec![CircuitField::<BN254Config>::zero(); pack_size];
    circuit.layers[0].input_vals = witness_private
        .iter()
        .map(|&v| {
            buf.fill(v);
            Simd::pack(&buf)
        })
        .collect();
    circuit.evaluate();
    let mpi_config = MPIConfig::prover_new();
    let (claimed_v, proof) = executor::prove::<BN254Config>(circuit, mpi_config);
    executor::dump_proof_and_claimed_v(&proof, &claimed_v)
        .map_err(|e| RunError::Serialize(format!("proof: {e:?}")))
}

#[allow(clippy::missing_errors_doc)]
pub fn verify_bn254_direct(
    circuit: &mut expander_circuit::Circuit<FC<BN254Config>>,
    witness_private: &[CircuitField<BN254Config>],
    proof_bytes: &[u8],
) -> Result<bool, RunError> {
    type Simd = <<BN254Config as GKREngine>::FieldConfig as expander_compiler::gkr_engine::FieldEngine>::SimdCircuitField;
    let pack_size = <Simd as SimdField>::PACK_SIZE;
    let mut buf = vec![CircuitField::<BN254Config>::zero(); pack_size];
    circuit.layers[0].input_vals = witness_private
        .iter()
        .map(|&v| {
            buf.fill(v);
            Simd::pack(&buf)
        })
        .collect();
    let (proof, claimed_v) =
        executor::load_proof_and_claimed_v::<ChallengeField<BN254Config>>(proof_bytes)
            .map_err(|e| RunError::Deserialize(format!("proof: {e:?}")))?;
    let mpi_config = MPIConfig::verifier_new(1);
    Ok(executor::verify::<BN254Config>(
        circuit, mpi_config, &proof, &claimed_v,
    ))
}

fn read_activations_from_file(
    input_path: &str,
) -> Result<Vec<CircuitField<BN254Config>>, RunError> {
    use crate::circuit_functions::utils::onnx_model::InputData;
    use crate::circuit_functions::utils::tensor_ops::get_nd_circuit_inputs;
    use crate::io::io_reader::onnx_context::OnnxContext;

    let params = OnnxContext::get_params()?;
    let input_bytes = std::fs::read(input_path).map_err(|e| RunError::Io {
        source: e,
        path: input_path.into(),
    })?;
    let input_data: InputData = rmp_serde::from_slice(&input_bytes)
        .map_err(|e| RunError::Deserialize(format!("input msgpack: {e:?}")))?;
    let expected_size = params.effective_input_dims();
    let arr = get_nd_circuit_inputs::<BN254Config>(&input_data.input, &[expected_size])
        .map_err(|e| RunError::Witness(format!("input extraction: {e}")))?;
    Ok(arr.into_iter().collect())
}

#[allow(clippy::missing_errors_doc)]
pub fn fast_compile_prove(
    input_path: &str,
    proof_path: &str,
    compress: bool,
) -> Result<(), RunError> {
    use crate::io::io_reader::onnx_context::OnnxContext;
    use crate::runner::schema::ProofBundle;
    use crate::runner::version::jstprove_artifact_version;
    use serde::Serialize;

    let params = OnnxContext::get_params()?;
    let input_arr_vals = read_activations_from_file(input_path)?;

    eprintln!("[fast-compile] Compiling circuit with DirectBuilder (bypasses ECC IR pipeline)");
    eprintln!(
        "[fast-compile] Tradeoff: ~6x faster total, ~2x slower prove/verify vs standard compile"
    );

    let (mut circuit, witness) = compile_and_witness_bn254_from_fields(&params, &input_arr_vals)?;
    let proof = prove_bn254_direct(&mut circuit, &witness)?;
    let proof = jstprove_io::maybe_compress_bytes(proof, compress)
        .map_err(|e| RunError::Serialize(format!("zstd compress: {e}")))?;

    let resp = ProofBundle {
        proof,
        version: Some(jstprove_artifact_version()),
    };
    let file = std::fs::File::create(proof_path).map_err(|e| RunError::Io {
        source: e,
        path: proof_path.into(),
    })?;
    let mut writer = std::io::BufWriter::new(file);
    resp.serialize(&mut rmp_serde::Serializer::new(&mut writer).with_struct_map())
        .map_err(|e| RunError::Serialize(format!("proof msgpack: {e:?}")))?;
    std::io::Write::flush(&mut writer).map_err(|e| RunError::Io {
        source: e,
        path: proof_path.into(),
    })?;

    Ok(())
}

#[allow(clippy::missing_errors_doc)]
pub fn fast_compile_verify(input_path: &str, proof_path: &str) -> Result<bool, RunError> {
    use crate::io::io_reader::onnx_context::OnnxContext;
    use crate::runner::main_runner::auto_decompress_bytes;
    use crate::runner::schema::ProofBundle;

    let params = OnnxContext::get_params()?;
    let input_arr_vals = read_activations_from_file(input_path)?;

    eprintln!("[fast-compile] Compiling circuit with DirectBuilder (bypasses ECC IR pipeline)");
    eprintln!(
        "[fast-compile] Tradeoff: ~6x faster total, ~2x slower prove/verify vs standard compile"
    );

    let (mut circuit, witness) = compile_and_witness_bn254_from_fields(&params, &input_arr_vals)?;

    let proof_bundle: ProofBundle = {
        let file = std::fs::File::open(proof_path).map_err(|e| RunError::Io {
            source: e,
            path: proof_path.into(),
        })?;
        rmp_serde::decode::from_read(std::io::BufReader::new(file))
            .map_err(|e| RunError::Deserialize(format!("proof msgpack: {e:?}")))?
    };

    let proof_data = auto_decompress_bytes(&proof_bundle.proof)?;
    verify_bn254_direct(&mut circuit, &witness, &proof_data)
}
