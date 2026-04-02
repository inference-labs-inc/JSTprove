use ndarray::{Array1, ArrayD, ArrayView1, Axis, Ix1, IxDyn, concatenate};

use expander_compiler::expander_circuit;
use expander_compiler::frontend::{
    BN254Config,
    CircuitField,
    Config,
    Define,
    FieldArith,
    GoldilocksBasefoldConfig,
    GoldilocksConfig,
    GoldilocksExt2BasefoldConfig,
    GoldilocksWhirConfig,
    GoldilocksWhirPQConfig,
    RootAPI,
    Variable,
    declare_circuit, // GoldilocksWhirConfig = GoldilocksExt3x1ConfigSha2Whir
};
use expander_compiler::gkr_engine::GKREngine;

use crate::circuit_functions::CircuitError;
use crate::circuit_functions::gadgets::LogupRangeCheckContext;
use crate::circuit_functions::gadgets::autotuner;
use crate::circuit_functions::hints::build_logup_hint_registry;
use crate::circuit_functions::utils::ArrayConversionError;
use crate::circuit_functions::utils::build_layers::{build_layers, default_n_bits_for_config};
use crate::circuit_functions::utils::onnx_model::{
    Architecture, CircuitParams, InputData, OutputData, WANDB,
};
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
    verify_and_extract_with_flat, verify_and_extract_with_flat_ref,
    verify_and_extract_with_layered,
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
        let (architecture, w_and_b) = get_architecture_and_wandb(&params)?;
        self.try_define_impl::<C, Builder>(api, &params, &architecture, &w_and_b)
    }

    fn try_define_impl<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &mut Builder,
        params: &CircuitParams,
        architecture: &Architecture,
        w_and_b: &WANDB,
    ) -> Result<(), CircuitError> {
        if architecture.architecture.is_empty() {
            return Err(CircuitError::EmptyArchitecture);
        }

        let mut out = get_inputs(&self.input_arr, &params.inputs)?;

        let layers = build_layers::<C, Builder>(params, architecture, w_and_b)?;
        let total_layers = layers.len();

        let chunk_bits = params
            .logup_chunk_bits
            .unwrap_or(crate::circuit_functions::gadgets::DEFAULT_LOGUP_CHUNK_BITS);
        let mut logup_ctx = LogupRangeCheckContext::new(chunk_bits);
        logup_ctx.init::<C, Builder>(api);

        for (pos, built) in layers.iter().enumerate() {
            api.display(
                &format!(
                    "    [{:>2}/{}] {} ({})",
                    pos + 1,
                    total_layers,
                    built.name,
                    built.op_type
                ),
                0,
            );
            for (key, value) in built.layer.apply_multi(api, &mut logup_ctx, &out)? {
                out.insert(key, value);
            }
        }

        api.display("    [logup finalize]", 0);
        logup_ctx.finalize::<C, Builder>(api);

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
        api.display("    [output assertions]", 0);
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
    witness_from_f64_generic::<BN254Config>(
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

/// ONNX TensorProto element types whose values carry indices,
/// counts, or predicates and must enter the circuit at unit scale.
/// Quantising them by alpha would feed e.g. an INT64 index of 1054
/// into a gather as 1054 * 2^scale_exponent and trip every
/// downstream bounds / mux / range check.
///
/// Codes follow the ONNX TensorProto.DataType enumeration:
/// UINT8=2, INT8=3, UINT16=4, INT16=5, INT32=6, INT64=7, BOOL=9,
/// UINT32=12, UINT64=13.
fn is_integer_elem_type(t: i16) -> bool {
    matches!(t, 2 | 3 | 4 | 5 | 6 | 7 | 9 | 12 | 13)
}

/// Convert an integer-typed wire value (passed as f64 by the
/// host-side serialisation) into the i64 the circuit expects,
/// rejecting the silently-corrupting cases up front:
///   * non-finite values (NaN / inf cast to 0 / i64::MAX),
///   * non-integral values (e.g. 0.5 silently truncating to 0),
///   * values outside the f64-exact integer range
///     (|v| > 2^53 cannot be uniquely round-tripped).
/// Returning a `RunError` instead of panicking keeps the prover
/// in a recoverable state and surfaces upstream slicer bugs at
/// the boundary instead of after they have produced a witness for
/// an unintended computation.
fn f64_index_to_i64(v: f64) -> Result<i64, RunError> {
    if !v.is_finite() {
        return Err(RunError::Witness(format!(
            "integer-typed input value is not finite: {v}"
        )));
    }
    // Bit-pattern compare on f64 is exact; any rounding would
    // change at least one mantissa bit, so this catches every
    // non-integral wire value the slicer may have emitted.
    if v.round().to_bits() != v.to_bits() {
        return Err(RunError::Witness(format!(
            "integer-typed input value is not integral: {v}"
        )));
    }
    #[allow(clippy::cast_precision_loss)]
    let max_exact = (1u64 << 53) as f64;
    if v.abs() > max_exact {
        return Err(RunError::Witness(format!(
            "integer-typed input value exceeds f64-exact range (|v| > 2^53): {v}"
        )));
    }
    #[allow(clippy::cast_possible_truncation)]
    Ok(v.round() as i64)
}

/// Encode the activation portion of a circuit's input array,
/// dispatching per-input on the manifest element type so integer
/// indices/counts/booleans bypass alpha quantisation.  Used by
/// both witness_from_f64_generic and build_debug_assignment so the
/// production and debug paths agree on the encoding.
fn push_activation_inputs<C: Config>(
    inputs: &[crate::circuit_functions::utils::onnx_types::ONNXIO],
    activations: &[f64],
    alpha: f64,
    out: &mut Vec<CircuitField<C>>,
) -> Result<(), RunError> {
    let mut offset = 0usize;
    for io in inputs {
        let elems: usize = io.shape.iter().product();
        let end = offset + elems;
        if is_integer_elem_type(io.elem_type) {
            for &v in &activations[offset..end] {
                out.push(convert_val_to_field_element::<C>(f64_index_to_i64(v)?));
            }
        } else {
            for &v in &activations[offset..end] {
                out.push(quantize_f64_to_field::<C>(v, alpha));
            }
        }
        offset = end;
    }
    Ok(())
}

/// Find the index in `inputs` at which the cumulative shape
/// product equals `activation_elements`.  Used in the debug path
/// where activations and initialisers travel via different
/// channels: the user's input array is exactly the activation
/// portion of the manifest, and everything after the boundary is
/// loaded by the witness solver from the bundle.
fn activation_input_split(
    inputs: &[crate::circuit_functions::utils::onnx_types::ONNXIO],
    activation_elements: usize,
) -> Result<usize, RunError> {
    let mut covered = 0usize;
    for (i, io) in inputs.iter().enumerate() {
        if covered == activation_elements {
            return Ok(i);
        }
        let elems: usize = io.shape.iter().product();
        covered = covered
            .checked_add(elems)
            .ok_or_else(|| RunError::Witness("input shape product overflowed usize".into()))?;
        if covered > activation_elements {
            return Err(RunError::Witness(format!(
                "activation buffer length {activation_elements} does not align with manifest input boundaries"
            )));
        }
    }
    if covered == activation_elements {
        Ok(inputs.len())
    } else {
        Err(RunError::Witness(format!(
            "activation buffer length {activation_elements} exceeds total input shape sum {covered}"
        )))
    }
}

/// Encode the initialiser portion of a circuit's input array.
/// Vector-shaped (rank-1) initialisers carry alpha squared because
/// the slicer pre-scales bias-shaped initialisers by alpha; all
/// other float-typed initialisers carry alpha.  Integer-typed
/// initialisers (e.g. INT64 shape constants embedded as
/// initialisers) bypass quantisation entirely.
fn push_initializer_inputs<C: Config>(
    inputs: &[crate::circuit_functions::utils::onnx_types::ONNXIO],
    initializers: &[(Vec<f64>, Vec<usize>)],
    alpha: f64,
    alpha_sq: f64,
    out: &mut Vec<CircuitField<C>>,
) -> Result<(), RunError> {
    for (idx, (values, _shape)) in initializers.iter().enumerate() {
        let io = &inputs[idx];
        if is_integer_elem_type(io.elem_type) {
            for &v in values {
                out.push(convert_val_to_field_element::<C>(f64_index_to_i64(v)?));
            }
        } else {
            let scale = if io.shape.len() == 1 { alpha_sq } else { alpha };
            for &v in values {
                out.push(quantize_f64_to_field::<C>(v, scale));
            }
        }
    }
    Ok(())
}

/// # Errors
/// Returns `RunError` on witness generation, serialization, or activation mismatch.
pub fn witness_from_f64_generic<C: Config>(
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
    push_activation_inputs::<C>(
        &params.inputs[..num_activation_entries],
        activations,
        alpha,
        &mut input_arr,
    )?;
    push_initializer_inputs::<C>(
        &params.inputs[num_activation_entries..],
        initializers,
        alpha,
        alpha_sq,
        &mut input_arr,
    )?;

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
    let output_i64: Vec<i64> = computed_outputs
        .iter()
        .map(|v| crate::circuit_functions::hints::field_to_i64(*v))
        .collect();

    assignment.outputs = computed_outputs;

    let witness = witness_solver
        .solve_witness_with_hints(&assignment, &hint_registry)
        .map_err(|e| RunError::Witness(format!("final pass: {e:?}")))?;

    let witness_bytes = serialize_witness::<C>(&witness, compress)?;

    Ok(WitnessBundle {
        witness: witness_bytes,
        output_data: Some(output_i64),
        version: Some(crate::runner::version::jstprove_artifact_version()),
    })
}

/// Builds a quantized assignment from raw f64 activations with auto-computed outputs.
///
/// Performs a probe pass through the compiled circuit to determine the correct
/// output values, then returns a complete assignment suitable for `debug_eval`.
///
/// # Errors
/// Returns `RunError` on quantization, deserialization, or witness solving failure.
pub fn build_debug_assignment<C: Config>(
    circuit_bytes: &[u8],
    solver_bytes: &[u8],
    params: &CircuitParams,
    activations: &[f64],
) -> Result<Circuit<CircuitField<C>>, RunError> {
    #[allow(clippy::cast_possible_wrap)]
    let alpha = f64::from(params.scale_base).powi(params.scale_exponent as i32);

    // Mirror the production witness path: dispatch per-input on
    // manifest element type so integer indices, counts and BOOL
    // values bypass alpha quantisation.  Initialisers are loaded
    // separately by the witness solver from the bundle, so the
    // debug input array consists of activation entries only -- we
    // split params.inputs at the boundary where the cumulative
    // shape product equals the activation buffer length.
    let activation_entries = activation_input_split(&params.inputs, activations.len())?;
    let mut input_arr: Vec<CircuitField<C>> = Vec::with_capacity(activations.len());
    push_activation_inputs::<C>(
        &params.inputs[..activation_entries],
        activations,
        alpha,
        &mut input_arr,
    )?;

    let layered_circuit = load_circuit_from_bytes::<C>(circuit_bytes)?;
    let witness_solver = load_witness_solver_from_bytes::<C>(solver_bytes)?;
    let hint_registry = build_logup_hint_registry::<CircuitField<C>>();
    let num_outputs = params.effective_output_dims();

    let mut assignment = Circuit::<CircuitField<C>> {
        input_arr: input_arr.clone(),
        outputs: vec![CircuitField::<C>::zero(); num_outputs],
        ..Default::default()
    };
    init_circuit_fields::<C>(&mut assignment, params);

    let probe_witness = witness_solver
        .solve_witness_with_hints(&assignment, &hint_registry)
        .map_err(|e| RunError::Witness(format!("probe pass: {e:?}")))?;

    let (private_inputs, public_inputs) = probe_witness
        .iter_scalar()
        .next()
        .ok_or_else(|| RunError::Witness("empty probe witness".into()))?;

    let (actual_outputs, _) =
        layered_circuit.eval_with_public_inputs(private_inputs, &public_inputs);

    assignment.input_arr = input_arr;
    assignment.outputs = actual_outputs[..num_outputs].to_vec();

    Ok(assignment)
}

/// Generate a witness from pre-quantized integer inputs (e.g. from input.msgpack).
///
/// Unlike [`witness_from_f64_generic`], this function does NOT apply alpha scaling
/// to the inputs — they are already quantized `i64` values that map directly to
/// circuit field elements via [`crate::circuit_functions::utils::tensor_ops::convert_val_to_field_element`].
///
/// # Errors
/// Returns `RunError` on witness generation, serialization, or activation mismatch.
pub fn witness_from_prequantized<C: Config>(
    circuit_bytes: &[u8],
    solver_bytes: &[u8],
    params: &CircuitParams,
    input_data: &InputData,
    compress: bool,
) -> Result<WitnessBundle, RunError> {
    let layered_circuit = load_circuit_from_bytes::<C>(circuit_bytes)?;
    let witness_solver = load_witness_solver_from_bytes::<C>(solver_bytes)?;
    let hint_registry = build_logup_hint_registry::<CircuitField<C>>();
    let num_outputs = params.effective_output_dims();

    // Build assignment with pre-quantized inputs (no alpha re-scaling)
    let mut probe_assignment =
        apply_input_data::<C>(input_data, Circuit::<CircuitField<C>>::default(), params)?;
    // Outputs start at zero for the probe pass
    probe_assignment.outputs = vec![CircuitField::<C>::zero(); num_outputs];

    let probe_witness = witness_solver
        .solve_witness_with_hints(&probe_assignment, &hint_registry)
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
    let output_i64: Vec<i64> = computed_outputs
        .iter()
        .map(|v| crate::circuit_functions::hints::field_to_i64(*v))
        .collect();

    probe_assignment.outputs = computed_outputs;

    let witness = witness_solver
        .solve_witness_with_hints(&probe_assignment, &hint_registry)
        .map_err(|e| RunError::Witness(format!("final pass: {e:?}")))?;

    let witness_bytes = serialize_witness::<C>(&witness, compress)?;

    Ok(WitnessBundle {
        witness: witness_bytes,
        output_data: Some(output_i64),
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

/// Stamp the manifest with the proof config that is about to compile
/// it. Always returns `Some(CircuitParams)` — every compiled bundle
/// must carry an explicit, versioned proof config so downstream
/// consumers can resolve the correct prover variant without guessing.
///
/// If the caller passes `None` (e.g. low-level test fixtures that
/// bypass the high-level `api::compile` entry point), a minimal
/// placeholder `CircuitParams` is fabricated solely to carry the stamp
/// — its other fields default to zero/empty and are not meaningful for
/// real runtime use.
fn stamp_proof_config(
    metadata: Option<CircuitParams>,
    config: crate::proof_config::ProofConfig,
) -> CircuitParams {
    let mut params = metadata.unwrap_or_else(|| CircuitParams {
        scale_base: 0,
        scale_exponent: 0,
        rescale_config: std::collections::HashMap::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        freivalds_reps: 0,
        n_bits_config: std::collections::HashMap::new(),
        weights_as_inputs: false,
        proof_system: crate::proof_system::ProofSystem::Expander,
        proof_config: None,
        logup_chunk_bits: None,
        public_inputs: Vec::new(),
    });
    params.proof_config = Some(crate::proof_config::StampedProofConfig::current(config));
    params
}

/// # Errors
/// Returns `RunError` on compilation or serialization failure.
pub fn compile_bn254(
    circuit_path: &str,
    compress: bool,
    metadata: Option<CircuitParams>,
) -> Result<(), RunError> {
    if metadata.as_ref().and_then(|m| m.logup_chunk_bits).is_none() {
        autotune_chunk_bits::<BN254Config>()?;
    }
    let metadata = stamp_proof_config(metadata, crate::proof_config::ProofConfig::Bn254Raw);
    crate::runner::main_runner::run_compile_and_serialize::<BN254Config, Circuit<Variable>>(
        circuit_path,
        compress,
        Some(metadata),
    )
}

fn autotune_chunk_bits<C: Config>() -> Result<(), RunError> {
    let params = OnnxContext::get_params()?;
    let architecture = OnnxContext::get_architecture()?;
    let n_bits = default_n_bits_for_config::<C>();

    if let Some(bits) = autotuner::lookup_operator(&params, &architecture, n_bits) {
        let mut p = params;
        p.logup_chunk_bits = Some(bits);
        OnnxContext::set_params(p);
        return Ok(());
    }

    if let Some(bits) = autotuner::lookup_circuit(&params, &architecture) {
        let mut p = params;
        p.logup_chunk_bits = Some(bits);
        OnnxContext::set_params(p);
        return Ok(());
    }

    let winner = autotuner::sweep_and_select(autotuner::candidates(), |chunk_bits| {
        let mut p = params.clone();
        p.logup_chunk_bits = Some(chunk_bits);
        OnnxContext::set_params(p);
        crate::runner::main_runner::compile_total_cost::<C, Circuit<Variable>>().ok()
    });

    autotuner::store_circuit(&params, &architecture, winner);
    autotuner::store_operator(&params, &architecture, n_bits, winner);

    let mut p = params;
    p.logup_chunk_bits = Some(winner);
    OnnxContext::set_params(p);
    Ok(())
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

pub type LayeredCircuitBN254 = expander_compiler::circuit::layered::Circuit<
    BN254Config,
    expander_compiler::circuit::layered::NormalInputType,
>;

/// # Errors
/// Returns `RunError` on decompression or deserialization failure.
pub fn deserialize_circuit_bn254(circuit_bytes: &[u8]) -> Result<LayeredCircuitBN254, RunError> {
    crate::runner::main_runner::load_circuit_from_bytes::<BN254Config>(circuit_bytes)
}

/// # Errors
/// Returns `RunError` on verification or output extraction failure.
pub fn verify_and_extract_bn254_with_layered(
    layered_circuit: &LayeredCircuitBN254,
    witness_bytes: &[u8],
    proof_bytes: &[u8],
    num_inputs: usize,
    expected_inputs: Option<&[f64]>,
) -> Result<VerifiedOutput, RunError> {
    verify_and_extract_with_layered::<BN254Config>(
        layered_circuit,
        witness_bytes,
        proof_bytes,
        num_inputs,
        expected_inputs,
    )
}

type FC<C> = <C as GKREngine>::FieldConfig;

pub type FlatCircuitBN254 = expander_circuit::Circuit<FC<BN254Config>>;

#[must_use]
pub fn flatten_circuit_bn254(layered: &LayeredCircuitBN254) -> FlatCircuitBN254 {
    layered.export_to_expander_flatten()
}

/// # Errors
/// Returns `RunError` on verification or output extraction failure.
pub fn verify_and_extract_bn254_with_flat(
    circuit: &mut FlatCircuitBN254,
    witness_bytes: &[u8],
    proof_bytes: &[u8],
    num_inputs: usize,
    expected_inputs: Option<&[f64]>,
) -> Result<VerifiedOutput, RunError> {
    verify_and_extract_with_flat::<BN254Config>(
        circuit,
        witness_bytes,
        proof_bytes,
        num_inputs,
        expected_inputs,
    )
}

/// # Errors
/// Returns `RunError` on verification or output extraction failure.
pub fn verify_and_extract_bn254_with_flat_ref(
    circuit: &FlatCircuitBN254,
    witness_bytes: &[u8],
    proof_bytes: &[u8],
    num_inputs: usize,
    expected_inputs: Option<&[f64]>,
) -> Result<VerifiedOutput, RunError> {
    verify_and_extract_with_flat_ref::<BN254Config>(
        circuit,
        witness_bytes,
        proof_bytes,
        num_inputs,
        expected_inputs,
    )
}

/// # Errors
/// Returns `RunError` on compilation or serialization failure.
pub fn compile_goldilocks(
    circuit_path: &str,
    compress: bool,
    metadata: Option<CircuitParams>,
) -> Result<(), RunError> {
    let metadata = stamp_proof_config(metadata, crate::proof_config::ProofConfig::GoldilocksRaw);
    crate::runner::main_runner::run_compile_and_serialize::<GoldilocksConfig, Circuit<Variable>>(
        circuit_path,
        compress,
        Some(metadata),
    )
}

/// # Errors
/// Returns `RunError` on witness generation failure.
pub fn witness_goldilocks(req: &WitnessRequest, compress: bool) -> Result<WitnessBundle, RunError> {
    let mut reader = ValueReader {
        params: req.metadata.clone(),
    };
    witness_from_request::<GoldilocksConfig, ValueReader, Circuit<CircuitField<GoldilocksConfig>>>(
        req,
        &mut reader,
        compress,
    )
}

/// # Errors
/// Returns `RunError` on witness generation or activation mismatch.
pub fn witness_goldilocks_from_f64(
    circuit_bytes: &[u8],
    solver_bytes: &[u8],
    params: &CircuitParams,
    activations: &[f64],
    initializers: &[(Vec<f64>, Vec<usize>)],
    compress: bool,
) -> Result<WitnessBundle, RunError> {
    witness_from_f64_generic::<GoldilocksConfig>(
        circuit_bytes,
        solver_bytes,
        params,
        activations,
        initializers,
        compress,
    )
}

/// # Errors
/// Returns `RunError` on proof generation failure.
pub fn prove_goldilocks(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    compress: bool,
) -> Result<Vec<u8>, RunError> {
    prove_from_bytes::<GoldilocksConfig>(circuit_bytes, witness_bytes, compress)
}

/// # Errors
/// Returns `RunError` on verification failure.
pub fn verify_goldilocks(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
) -> Result<bool, RunError> {
    verify_from_bytes::<GoldilocksConfig>(circuit_bytes, witness_bytes, proof_bytes)
}

pub type LayeredCircuitGoldilocks = expander_compiler::circuit::layered::Circuit<
    GoldilocksConfig,
    expander_compiler::circuit::layered::NormalInputType,
>;

/// # Errors
/// Returns `RunError` on decompression or deserialization failure.
pub fn deserialize_circuit_goldilocks(
    circuit_bytes: &[u8],
) -> Result<LayeredCircuitGoldilocks, RunError> {
    crate::runner::main_runner::load_circuit_from_bytes::<GoldilocksConfig>(circuit_bytes)
}

/// # Errors
/// Returns `RunError` on verification or output extraction failure.
pub fn verify_and_extract_goldilocks(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
    num_inputs: usize,
    expected_inputs: Option<&[f64]>,
) -> Result<VerifiedOutput, RunError> {
    verify_and_extract_from_bytes::<GoldilocksConfig>(
        circuit_bytes,
        witness_bytes,
        proof_bytes,
        num_inputs,
        expected_inputs,
    )
}

/// # Errors
/// Returns `RunError` on compilation or serialization failure.
pub fn compile_goldilocks_basefold(
    circuit_path: &str,
    compress: bool,
    metadata: Option<CircuitParams>,
) -> Result<(), RunError> {
    let metadata = stamp_proof_config(
        metadata,
        crate::proof_config::ProofConfig::GoldilocksBasefold,
    );
    crate::runner::main_runner::run_compile_and_serialize::<
        GoldilocksBasefoldConfig,
        Circuit<Variable>,
    >(circuit_path, compress, Some(metadata))
}

/// # Errors
/// Returns `RunError` on witness generation failure.
pub fn witness_goldilocks_basefold_from_f64(
    circuit_bytes: &[u8],
    solver_bytes: &[u8],
    params: &CircuitParams,
    activations: &[f64],
    initializers: &[(Vec<f64>, Vec<usize>)],
    compress: bool,
) -> Result<WitnessBundle, RunError> {
    witness_from_f64_generic::<GoldilocksBasefoldConfig>(
        circuit_bytes,
        solver_bytes,
        params,
        activations,
        initializers,
        compress,
    )
}

/// # Errors
/// Returns `RunError` on proof generation failure.
pub fn prove_goldilocks_basefold(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    compress: bool,
) -> Result<Vec<u8>, RunError> {
    prove_from_bytes::<GoldilocksBasefoldConfig>(circuit_bytes, witness_bytes, compress)
}

/// # Errors
/// Returns `RunError` on verification failure.
pub fn verify_goldilocks_basefold(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
) -> Result<bool, RunError> {
    verify_from_bytes::<GoldilocksBasefoldConfig>(circuit_bytes, witness_bytes, proof_bytes)
}

/// # Errors
/// Returns `RunError` on verification or output extraction failure.
pub fn verify_and_extract_goldilocks_basefold(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
    num_inputs: usize,
    expected_inputs: Option<&[f64]>,
) -> Result<VerifiedOutput, RunError> {
    verify_and_extract_from_bytes::<GoldilocksBasefoldConfig>(
        circuit_bytes,
        witness_bytes,
        proof_bytes,
        num_inputs,
        expected_inputs,
    )
}

/// # Errors
/// Returns `RunError` on compilation or serialization failure.
pub fn compile_goldilocks_whir(
    circuit_path: &str,
    compress: bool,
    metadata: Option<CircuitParams>,
) -> Result<(), RunError> {
    let metadata = stamp_proof_config(
        metadata,
        crate::proof_config::ProofConfig::GoldilocksExt3Whir,
    );
    crate::runner::main_runner::run_compile_and_serialize::<GoldilocksWhirConfig, Circuit<Variable>>(
        circuit_path,
        compress,
        Some(metadata),
    )
}

/// # Errors
/// Returns `RunError` on witness generation failure.
pub fn witness_goldilocks_whir_from_f64(
    circuit_bytes: &[u8],
    solver_bytes: &[u8],
    params: &CircuitParams,
    activations: &[f64],
    initializers: &[(Vec<f64>, Vec<usize>)],
    compress: bool,
) -> Result<WitnessBundle, RunError> {
    witness_from_f64_generic::<GoldilocksWhirConfig>(
        circuit_bytes,
        solver_bytes,
        params,
        activations,
        initializers,
        compress,
    )
}

/// # Errors
/// Returns `RunError` on proof generation failure.
pub fn prove_goldilocks_whir(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    compress: bool,
) -> Result<Vec<u8>, RunError> {
    prove_from_bytes::<GoldilocksWhirConfig>(circuit_bytes, witness_bytes, compress)
}

/// # Errors
/// Returns `RunError` on verification failure.
pub fn verify_goldilocks_whir(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
) -> Result<bool, RunError> {
    verify_from_bytes::<GoldilocksWhirConfig>(circuit_bytes, witness_bytes, proof_bytes)
}

/// # Errors
/// Returns `RunError` on verification or output extraction failure.
pub fn verify_and_extract_goldilocks_whir(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
    num_inputs: usize,
    expected_inputs: Option<&[f64]>,
) -> Result<VerifiedOutput, RunError> {
    verify_and_extract_from_bytes::<GoldilocksWhirConfig>(
        circuit_bytes,
        witness_bytes,
        proof_bytes,
        num_inputs,
        expected_inputs,
    )
}

/// # Errors
/// Returns `RunError` on compilation or serialization failure.
pub fn compile_goldilocks_whir_pq(
    circuit_path: &str,
    compress: bool,
    metadata: Option<CircuitParams>,
) -> Result<(), RunError> {
    let metadata = stamp_proof_config(
        metadata,
        crate::proof_config::ProofConfig::GoldilocksExt4Whir,
    );
    crate::runner::main_runner::run_compile_and_serialize::<GoldilocksWhirPQConfig, Circuit<Variable>>(
        circuit_path,
        compress,
        Some(metadata),
    )
}

/// # Errors
/// Returns `RunError` on witness generation failure.
pub fn witness_goldilocks_whir_pq_from_f64(
    circuit_bytes: &[u8],
    solver_bytes: &[u8],
    params: &CircuitParams,
    activations: &[f64],
    initializers: &[(Vec<f64>, Vec<usize>)],
    compress: bool,
) -> Result<WitnessBundle, RunError> {
    witness_from_f64_generic::<GoldilocksWhirPQConfig>(
        circuit_bytes,
        solver_bytes,
        params,
        activations,
        initializers,
        compress,
    )
}

/// # Errors
/// Returns `RunError` on proof generation failure.
pub fn prove_goldilocks_whir_pq(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    compress: bool,
) -> Result<Vec<u8>, RunError> {
    prove_from_bytes::<GoldilocksWhirPQConfig>(circuit_bytes, witness_bytes, compress)
}

/// # Errors
/// Returns `RunError` on verification failure.
pub fn verify_goldilocks_whir_pq(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
) -> Result<bool, RunError> {
    verify_from_bytes::<GoldilocksWhirPQConfig>(circuit_bytes, witness_bytes, proof_bytes)
}

/// # Errors
/// Returns `RunError` on verification or output extraction failure.
pub fn verify_and_extract_goldilocks_whir_pq(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
    num_inputs: usize,
    expected_inputs: Option<&[f64]>,
) -> Result<VerifiedOutput, RunError> {
    verify_and_extract_from_bytes::<GoldilocksWhirPQConfig>(
        circuit_bytes,
        witness_bytes,
        proof_bytes,
        num_inputs,
        expected_inputs,
    )
}

/// # Errors
/// Returns `RunError` on compilation or serialization failure.
pub fn compile_goldilocks_ext2(
    circuit_path: &str,
    compress: bool,
    metadata: Option<CircuitParams>,
) -> Result<(), RunError> {
    let metadata = stamp_proof_config(
        metadata,
        crate::proof_config::ProofConfig::GoldilocksExt2Basefold,
    );
    crate::runner::main_runner::run_compile_and_serialize::<
        GoldilocksExt2BasefoldConfig,
        Circuit<Variable>,
    >(circuit_path, compress, Some(metadata))
}

/// # Errors
/// Returns `RunError` on witness generation or activation mismatch.
pub fn witness_goldilocks_ext2_from_f64(
    circuit_bytes: &[u8],
    solver_bytes: &[u8],
    params: &CircuitParams,
    activations: &[f64],
    initializers: &[(Vec<f64>, Vec<usize>)],
    compress: bool,
) -> Result<WitnessBundle, RunError> {
    witness_from_f64_generic::<GoldilocksExt2BasefoldConfig>(
        circuit_bytes,
        solver_bytes,
        params,
        activations,
        initializers,
        compress,
    )
}

/// # Errors
/// Returns `RunError` on proof generation failure.
pub fn prove_goldilocks_ext2(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    compress: bool,
) -> Result<Vec<u8>, RunError> {
    prove_from_bytes::<GoldilocksExt2BasefoldConfig>(circuit_bytes, witness_bytes, compress)
}

/// # Errors
/// Returns `RunError` on verification failure.
pub fn verify_goldilocks_ext2(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
) -> Result<bool, RunError> {
    verify_from_bytes::<GoldilocksExt2BasefoldConfig>(circuit_bytes, witness_bytes, proof_bytes)
}

/// # Errors
/// Returns `RunError` on verification or output extraction failure.
pub fn verify_and_extract_goldilocks_ext2(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
    num_inputs: usize,
    expected_inputs: Option<&[f64]>,
) -> Result<VerifiedOutput, RunError> {
    verify_and_extract_from_bytes::<GoldilocksExt2BasefoldConfig>(
        circuit_bytes,
        witness_bytes,
        proof_bytes,
        num_inputs,
        expected_inputs,
    )
}

fn get_architecture_and_wandb(
    _params: &CircuitParams,
) -> Result<(Architecture, WANDB), crate::io::io_reader::onnx_context::OnnxContextError> {
    let architecture = OnnxContext::get_architecture()?;
    let wandb = OnnxContext::get_wandb()?;
    Ok((architecture, wandb))
}
