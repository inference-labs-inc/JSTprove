#[allow(unused_imports)]
/// Standard library imports
use core::panic;

use jstprove_circuits::circuit_functions::CircuitError;
use jstprove_circuits::circuit_functions::utils::ArrayConversionError;
use jstprove_circuits::circuit_functions::utils::build_layers::build_layers;
use jstprove_circuits::runner::errors::RunError;
/// External crate imports
use std::sync::LazyLock;

use ndarray::{ArrayD, Ix1, IxDyn};

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
use jstprove_circuits::runner::main_runner::{ConfigurableCircuit, handle_args};

type WeightsData = (Architecture, CircuitParams);

// // This reads the weights json into a string
const MATRIX_WEIGHTS_FILE: &str =
    include_str!("../../../python/models/weights/onnx_generic_circuit_weights.json");

//lazy static macro, forces this to be done at compile time (and allows for a constant of this weights variable)
// Weights will be read in
// Lazily parse the weights JSON file on first access
static WEIGHTS_INPUT: LazyLock<WeightsData> = LazyLock::new(|| {
    serde_json::from_str(MATRIX_WEIGHTS_FILE).expect("JSON was not well-formatted")
});
static WEIGHTS_INPUT2: LazyLock<WANDB> = LazyLock::new(|| {
    let path = std::path::Path::new("python/models/weights/onnx_generic_circuit_weights2.json");
    let json_str =
        std::fs::read_to_string(path).expect("Failed to read weights JSON file at runtime");
    serde_json::from_str(&json_str).expect("JSON was not well-formatted")
});
// Extract components from WEIGHTS_INPUT lazily
static ARCHITECTURE: LazyLock<Architecture> = LazyLock::new(|| WEIGHTS_INPUT.0.clone());
static W_AND_B: LazyLock<WANDB> = LazyLock::new(|| WEIGHTS_INPUT2.clone());
static CIRCUITPARAMS: LazyLock<CircuitParams> = LazyLock::new(|| WEIGHTS_INPUT.1.clone());

declare_circuit!(Circuit {
    input_arr: [PublicVariable],
    outputs: [PublicVariable],
    dummy: [Variable; 2]
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
        // Getting inputs
        let mut out = get_inputs(&self.input_arr, ARCHITECTURE.inputs.clone())?;

        // let mut out = out2.remove("input").unwrap().clone();
        let layers = build_layers::<C, Builder>(&CIRCUITPARAMS, &ARCHITECTURE, &W_AND_B)?;

        if ARCHITECTURE.architecture.is_empty() {
            return Err(CircuitError::EmptyArchitecture);
        }

        for (i, layer) in layers.iter().enumerate() {
            eprintln!("Applying Layer {:?}", &ARCHITECTURE.architecture[i].name);
            let result = layer.apply(api, out.clone())?;
            result.0.into_iter().for_each(|key| {
                // out.insert(key, Arc::clone(&value)); Depending on memory constraints here
                out.insert(key, result.1.clone());
            });
        }

        eprint!("Flatten output");
        let flatten_shape: Vec<usize> = vec![
            ARCHITECTURE
                .outputs
                .iter()
                .map(|obj| obj.shape.iter().product::<usize>())
                .product(),
        ];

        // TODO only support single output
        let output_name = ARCHITECTURE
            .outputs
            .first()
            .ok_or_else(|| CircuitError::Other("No outputs defined in ARCHITECTURE".to_string()))?
            .name
            .clone();

        let output = out
            .get(&output_name)
            .ok_or_else(|| CircuitError::Other("Missing output in map".into()))?
            .clone()
            .into_shape_with_order(IxDyn(&flatten_shape))
            .map_err(ArrayConversionError::ShapeError)?;

        let output = output
            .as_slice()
            .ok_or_else(|| CircuitError::Other("Output array not contiguous".into()))?;

        for (j, _) in self.outputs.iter().enumerate() {
            api.display("out1", self.outputs[j]);
            api.display("out2", output[j]);
            api.assert_is_equal(self.outputs[j], output[j]);
        }

        api.assert_is_equal(self.dummy[0], 1);
        api.assert_is_equal(self.dummy[1], 1);

        Ok(())
    }
}

impl ConfigurableCircuit for Circuit<Variable> {
    fn configure(&mut self) {
        // Change input and outputs as needed
        // Outputs
        let output_dims: usize = ARCHITECTURE
            .outputs
            .iter()
            .map(|obj| obj.shape.iter().product::<usize>())
            .product();
        self.outputs = vec![Variable::default(); output_dims];

        // Inputs
        let input_dims: usize = ARCHITECTURE
            .inputs
            .iter()
            .map(|obj| obj.shape.iter().product::<usize>())
            .product();
        self.input_arr = vec![Variable::default(); input_dims];
    }
}

impl<C: Config> IOReader<Circuit<CircuitField<C>>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: Circuit<CircuitField<C>>,
    ) -> Result<Circuit<CircuitField<C>>, RunError> {
        let data: InputData =
            <FileReader as IOReader<Circuit<_>, C>>::read_data_from_json::<InputData>(file_path)?;

        // compute the total number of inputs
        let input_dims: &[usize] = &[ARCHITECTURE
            .inputs
            .iter()
            .map(|obj| obj.shape.iter().product::<usize>())
            .product()];

        assignment.dummy[0] = CircuitField::<C>::from(1);
        assignment.dummy[1] = CircuitField::<C>::from(1);

        // 1) get back an ArrayD<CircuitField<C>>
        let arr: ArrayD<CircuitField<C>> = get_nd_circuit_inputs::<C>(&data.input, input_dims)
            .map_err(|e| RunError::Json(format!("Invalid input shape: {e}")))?;

        // 2) downcast to Ix1 and collect into a Vec
        let flat: Vec<CircuitField<C>> = arr
            .into_dimensionality::<Ix1>()
            .map_err(|_| RunError::Json("Expected a 1-D input array".into()))?
            .to_vec();

        assignment.input_arr = flat;
        Ok(assignment)
    }

    fn read_outputs(
        &mut self,
        file_path: &str,
        mut assignment: Circuit<CircuitField<C>>,
    ) -> Result<Circuit<CircuitField<C>>, RunError> {
        let data: OutputData =
            <FileReader as IOReader<Circuit<_>, C>>::read_data_from_json::<OutputData>(file_path)?;

        let output_dims: &[usize] = &[ARCHITECTURE
            .outputs
            .iter()
            .map(|obj| obj.shape.iter().product::<usize>())
            .product()];

        let arr: ArrayD<CircuitField<C>> = get_nd_circuit_inputs::<C>(&data.output, output_dims)
            .map_err(|e| RunError::Json(format!("Invalid output shape: {e}")))?;

        let flat: Vec<CircuitField<C>> = arr
            .into_dimensionality::<Ix1>()
            .map_err(|_| RunError::Json("Expected a 1-D output array".into()))?
            .to_vec();

        assignment.outputs = flat;
        Ok(assignment)
    }

    fn get_path(&self) -> &str {
        &self.path
    }
}

fn main() {
    let mut file_reader = FileReader {
        path: "demo_cnn".to_owned(),
    };

    if let Err(err) = handle_args::<BN254Config, Circuit<Variable>, Circuit<_>, _>(&mut file_reader)
    {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}
