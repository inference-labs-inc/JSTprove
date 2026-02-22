use ndarray::{Array1, ArrayD, ArrayView1, Axis, Ix1, IxDyn, concatenate};

use expander_compiler::frontend::{
    BN254Config, CircuitField, Config, Define, RootAPI, Variable, declare_circuit,
};

use crate::circuit_functions::CircuitError;
use crate::circuit_functions::utils::ArrayConversionError;
use crate::circuit_functions::utils::build_layers::build_layers;
use crate::circuit_functions::utils::onnx_model::{InputData, OutputData};
use crate::circuit_functions::utils::shaping::get_inputs;
use crate::circuit_functions::utils::tensor_ops::get_nd_circuit_inputs;
use crate::io::io_reader::onnx_context::OnnxContext;
use crate::io::io_reader::{FileReader, IOReader};
use crate::runner::errors::RunError;
use crate::runner::main_runner::{
    ConfigurableCircuit, prove_from_bytes, verify_from_bytes, witness_from_request,
};
use crate::runner::schema::{WitnessBundle, WitnessRequest};

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
            panic!("Circuit definition failed: {e:?}");
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
        let w_and_b = OnnxContext::get_wandb()?;

        if architecture.architecture.is_empty() {
            return Err(CircuitError::EmptyArchitecture);
        }

        let mut out = get_inputs(&self.input_arr, params.inputs.clone())?;

        let layers = build_layers::<C, Builder>(&params, &architecture, &w_and_b)?;

        for layer in &layers {
            let (keys, value) = layer.apply(api, &out)?;
            for key in keys {
                out.insert(key, value.clone());
            }
        }

        let flatten_shape: Vec<usize> = vec![
            params
                .outputs
                .iter()
                .map(|obj| obj.shape.iter().product::<usize>())
                .sum(),
        ];
        let mut flat_outputs: Vec<Array1<Variable>> = Vec::new();

        for output_info in &params.outputs {
            let output_name = &output_info.name;

            let output = out
                .get(output_name)
                .ok_or_else(|| {
                    CircuitError::Other(format!("Missing output '{output_name}' in map"))
                })?
                .clone();

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
                "output length mismatch: circuit has {} outputs but computed {} values",
                self.outputs.len(),
                combined_output.len()
            )));
        }
        for (j, _) in self.outputs.iter().enumerate() {
            api.assert_is_equal(self.outputs[j], combined_output[j]);
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

        let output_dims: usize = params
            .outputs
            .iter()
            .map(|obj| obj.shape.iter().product::<usize>())
            .sum();
        self.outputs = vec![Variable::default(); output_dims];

        let input_dims: usize = params
            .inputs
            .iter()
            .map(|obj| obj.shape.iter().product::<usize>())
            .sum();
        self.input_arr = vec![Variable::default(); input_dims];

        Ok(())
    }
}

pub fn apply_input_data<C: Config>(
    data: &InputData,
    mut assignment: Circuit<CircuitField<C>>,
) -> Result<Circuit<CircuitField<C>>, RunError> {
    let params = OnnxContext::get_params()?;

    let input_dims: &[usize] = &[params
        .inputs
        .iter()
        .map(|obj| obj.shape.iter().product::<usize>())
        .sum()];

    assignment.dummy[0] = CircuitField::<C>::from(1);
    assignment.dummy[1] = CircuitField::<C>::from(1);

    assignment.scale_base[0] = CircuitField::<C>::from(params.scale_base);
    assignment.scale_exponent[0] = CircuitField::<C>::from(params.scale_exponent);

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
) -> Result<Circuit<CircuitField<C>>, RunError> {
    let params = OnnxContext::get_params()?;
    let output_dims: &[usize] = &[params
        .outputs
        .iter()
        .map(|obj| obj.shape.iter().product::<usize>())
        .sum()];

    let arr: ArrayD<CircuitField<C>> = get_nd_circuit_inputs::<C>(&data.output, output_dims)
        .map_err(|e| RunError::Json(format!("Invalid output shape: {e}")))?;

    let flat: Vec<CircuitField<C>> = arr
        .into_dimensionality::<Ix1>()
        .map_err(|_| RunError::Json("Expected a 1-D output array".into()))?
        .to_vec();

    assignment.outputs = flat;
    Ok(assignment)
}

impl<C: Config> IOReader<Circuit<CircuitField<C>>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        assignment: Circuit<CircuitField<C>>,
    ) -> Result<Circuit<CircuitField<C>>, RunError> {
        let data: InputData =
            <FileReader as IOReader<Circuit<_>, C>>::read_data_from_json::<InputData>(file_path)?;
        apply_input_data::<C>(&data, assignment)
    }

    fn read_outputs(
        &mut self,
        file_path: &str,
        assignment: Circuit<CircuitField<C>>,
    ) -> Result<Circuit<CircuitField<C>>, RunError> {
        let data: OutputData =
            <FileReader as IOReader<Circuit<_>, C>>::read_data_from_json::<OutputData>(file_path)?;
        apply_output_data::<C>(&data, assignment)
    }

    fn apply_values(
        &mut self,
        input: serde_json::Value,
        output: serde_json::Value,
        assignment: Circuit<CircuitField<C>>,
    ) -> Result<Circuit<CircuitField<C>>, RunError> {
        let input_data: InputData =
            serde_json::from_value(input).map_err(|e| RunError::Json(format!("{e:?}")))?;
        let output_data: OutputData =
            serde_json::from_value(output).map_err(|e| RunError::Json(format!("{e:?}")))?;
        let assignment = apply_input_data::<C>(&input_data, assignment)?;
        apply_output_data::<C>(&output_data, assignment)
    }

    fn get_path(&self) -> &str {
        &self.path
    }
}

pub fn witness_bn254(
    req: &WitnessRequest,
    compress: bool,
) -> Result<WitnessBundle, RunError> {
    // Empty path is intentional: witness_from_request calls
    // io_reader.apply_values() with inline Value data and never
    // invokes FileReader::read_inputs or read_outputs.
    let mut reader = FileReader {
        path: String::new(),
    };
    witness_from_request::<BN254Config, FileReader, Circuit<CircuitField<BN254Config>>>(
        req,
        &mut reader,
        compress,
    )
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
    metadata: Option<crate::circuit_functions::utils::onnx_model::CircuitParams>,
) -> Result<(), RunError> {
    crate::runner::main_runner::run_compile_and_serialize::<BN254Config, Circuit<Variable>>(
        circuit_path,
        compress,
        metadata,
    )
}
