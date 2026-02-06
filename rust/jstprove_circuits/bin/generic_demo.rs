#[allow(unused_imports)]
/// Standard library imports
use core::panic;

use jstprove_circuits::circuit_functions::CircuitError;
use jstprove_circuits::circuit_functions::utils::ArrayConversionError;
use jstprove_circuits::circuit_functions::utils::build_layers::build_layers;
use jstprove_circuits::runner::errors::RunError;
/// External crate imports
use ndarray::{Array1, ArrayD, ArrayView1, Axis, Ix1, IxDyn, concatenate};

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{
    BN254Config, CircuitField, Config, Define, RootAPI, Variable, declare_circuit,
};

/// Internal crate imports
use jstprove_circuits::circuit_functions::utils::onnx_model::{
    Architecture, CircuitParams, InputData, OutputData, WANDB,
};
use jstprove_circuits::circuit_functions::utils::shaping::get_inputs;

use jstprove_circuits::circuit_functions::utils::tensor_ops::get_nd_circuit_inputs;
use jstprove_circuits::io::io_reader::{FileReader, IOReader};
use jstprove_circuits::runner::main_runner::{ConfigurableCircuit, get_arg, get_args, handle_args};

/// Your new context module
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;

declare_circuit!(Circuit {
    input_arr: [PublicVariable],
    outputs: [PublicVariable],
    dummy: [Variable; 2],
    scale_base: [PublicVariable; 1],
    scale_exponent: [PublicVariable; 1],
});

// Memorization, in a better place
impl<C: Config> Define<C> for Circuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        if let Err(e) = self.try_define(api) {
            panic!("Circuit definition failed: {e:?}");
            // or:
            // eprintln!("Circuit definition failed: {e}");
            // return;
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

        // Getting inputs
        let mut out = get_inputs(&self.input_arr, params.inputs.clone())?;

        let layers = build_layers::<C, Builder>(params, architecture, w_and_b)?;

        if architecture.architecture.is_empty() {
            return Err(CircuitError::EmptyArchitecture);
        }

        for (i, layer) in layers.iter().enumerate() {
            eprintln!("Applying Layer {:?}", &architecture.architecture[i].name);
            let result = layer.apply(api, out.clone())?;
            result.0.into_iter().for_each(|key| {
                // out.insert(key, Arc::clone(&value)); Depending on memory constraints here
                out.insert(key, result.1.clone());
            });
        }

        let flatten_shape: Vec<usize> = vec![
            params
                .outputs
                .iter()
                .map(|obj| obj.shape.iter().product::<usize>())
                .sum(),
        ];
        eprintln!("Flatten shape {flatten_shape:?}");

        let mut flat_outputs: Vec<Array1<Variable>> = Vec::new();

        for output_info in &params.outputs {
            let output_name = &output_info.name;

            let output = out
                .get(output_name)
                .ok_or_else(|| {
                    CircuitError::Other(format!("Missing output '{output_name}' in map"))
                })?
                .clone();

            // Ensure the output is contiguous and flatten it
            let output_slice = output.as_slice().ok_or_else(|| {
                CircuitError::Other(format!("Output '{output_name}' array not contiguous"))
            })?;

            flat_outputs.push(Array1::from(output_slice.to_vec()));
        }

        // Concatenate all flattened outputs into one tensor
        let combined_output = concatenate(
            Axis(0),
            &flat_outputs
                .iter()
                .map(ndarray::ArrayBase::view)
                .collect::<Vec<ArrayView1<Variable>>>(),
        )
        .map_err(|e| CircuitError::Other(format!("Concatenation error: {e}")))?;

        // Optionally reshape it to the desired final flatten_shape
        let combined_output = combined_output
            .into_shape_with_order(IxDyn(&flatten_shape))
            .map_err(ArrayConversionError::ShapeError)?;

        eprintln!("Check outputs");
        for (j, _) in self.outputs.iter().enumerate() {
            api.display("out1", self.outputs[j]);
            api.display("out2", combined_output[j]);
            api.assert_is_equal(self.outputs[j], combined_output[j]);
        }

        api.assert_is_equal(self.dummy[0], 1);
        api.assert_is_equal(self.dummy[1], 1);

        api.assert_is_equal(self.scale_base[0], params.scale_base);
        api.assert_is_equal(self.scale_exponent[0], params.scale_exponent);

        Ok(())
    }
}

impl ConfigurableCircuit for Circuit<Variable> {
    fn configure(&mut self) -> Result<(), RunError> {
        // Change input and outputs as needed
        let params = OnnxContext::get_params()?;
        // Outputs
        let output_dims: usize = params
            .outputs
            .iter()
            .map(|obj| obj.shape.iter().product::<usize>())
            .sum();
        self.outputs = vec![Variable::default(); output_dims];

        // Inputs
        let input_dims: usize = params
            .inputs
            .iter()
            .map(|obj| obj.shape.iter().product::<usize>())
            .sum();
        eprintln!("input_dims {:?}", params.inputs);
        self.input_arr = vec![Variable::default(); input_dims];
        Ok(())
    }
}

fn apply_input_data<C: Config>(
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

fn apply_output_data<C: Config>(
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

fn set_onnx_context(matches: &clap::ArgMatches) {
    let meta_file_path = get_arg(matches, "meta").unwrap();
    let meta_file = std::fs::read_to_string(&meta_file_path).expect("Failed to read metadata file");
    let params: CircuitParams = serde_json::from_str(&meta_file).expect("Invalid metadata JSON");
    OnnxContext::set_params(params)
        .map_err(|e| CircuitError::Other(e.to_string()))
        .unwrap();

    let arch_file_path = get_arg(matches, "arch").unwrap();
    let arch_file =
        std::fs::read_to_string(&arch_file_path).expect("Failed to read architecture file");
    let arch: Architecture = serde_json::from_str(&arch_file).expect("Invalid architecture JSON");
    OnnxContext::set_architecture(arch)
        .map_err(|e| CircuitError::Other(e.to_string()))
        .unwrap();

    let wandb_file_path = get_arg(matches, "wandb").unwrap();
    let wandb_file = std::fs::read_to_string(&wandb_file_path).expect("Failed to read W&B file");
    let wandb: WANDB = serde_json::from_str(&wandb_file).expect("Invalid W&B JSON");
    OnnxContext::set_wandb(wandb)
        .map_err(|e| CircuitError::Other(e.to_string()))
        .unwrap();
}

const ONNX_CONTEXT_COMMANDS: &[&str] = &[
    "run_compile_circuit",
    "run_debug_witness",
    "msgpack_compile",
];

fn main() {
    let mut file_reader = FileReader {
        path: "demo_cnn".to_owned(),
    };

    let matches = get_args();

    let cmd_type = get_arg(&matches, "type").unwrap_or_default();
    let needs_onnx_context = ONNX_CONTEXT_COMMANDS.contains(&cmd_type.as_str());

    if needs_onnx_context {
        let has_meta = matches.get_one::<String>("meta").is_some();
        let has_arch = matches.get_one::<String>("arch").is_some();
        let has_wandb = matches.get_one::<String>("wandb").is_some();
        if !has_meta || !has_arch || !has_wandb {
            eprintln!(
                "Error: command '{cmd_type}' requires ONNX context. Provide --meta, --arch, and --wandb arguments."
            );
            std::process::exit(1);
        }
    }

    if needs_onnx_context {
        set_onnx_context(&matches);
    }

    if let Err(err) =
        handle_args::<BN254Config, Circuit<Variable>, Circuit<_>, _>(&matches, &mut file_reader)
    {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}
