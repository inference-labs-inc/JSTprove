use crate::runner::errors::RunError;
use expander_compiler::frontend::Config;
use expander_compiler::frontend::internal::DumpLoadTwoVariables;
use expander_compiler::gkr_engine::{FieldEngine, GKREngine};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::io::Read;

/// Implement `io_reader` to read inputs and outputs of the circuit.
///
/// This is primarily used for witness generation
pub trait IOReader<CircuitType, C: Config>
where
    CircuitType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    /// Reads and deserializes a JSON file into a strongly typed struct.
    ///
    /// This helper function:
    /// - Opens the file at `file_path`.
    /// - Reads its contents into a string.
    /// - Uses [`serde_json`] to deserialize the string into the target type `I`.
    /// - Converts any I/O or deserialization failures into [`RunError`].
    ///
    /// # Type Parameters
    ///
    /// - `I` – The type to deserialize into. Must implement [`DeserializeOwned`].
    ///
    /// # Arguments
    ///
    /// - `file_path` – Path to the JSON file to read.
    ///
    /// # Returns
    ///
    /// An instance of type `I` populated from the JSON contents.
    ///
    /// # Errors
    ///
    /// Returns a [`RunError`] if:
    /// - The file cannot be opened or read (`RunError::Io`).
    /// - The JSON cannot be parsed into the expected type (`RunError::Json`).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let inputs: InputData = read_data_from_json("inputs.json")?;
    /// let outputs: OutputData = read_data_from_json("outputs.json")?;
    /// ```
    fn read_data_from_json<I>(file_path: &str) -> Result<I, RunError>
    where
        I: DeserializeOwned,
    {
        // Read the JSON file into a string
        let mut file = std::fs::File::open(file_path).map_err(|e| RunError::Io {
            source: e,
            path: file_path.into(),
        })?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| RunError::Io {
                source: e,
                path: file_path.into(),
            })?;
        // Deserialize the JSON into the InputData struct
        serde_json::from_str(&contents).map_err(|e| RunError::Json(format!("{e:?}")))
    }
    /// Reads circuit inputs from a JSON file and applies them to a circuit assignment.
    ///
    /// # Arguments
    ///
    /// - `file_path` – Path to a JSON file containing input values.
    /// - `assignment` – The circuit assignment to populate.
    ///
    /// # Returns
    ///
    /// A new [`Circuit`] with its inputs populated from the JSON file.
    ///
    /// # Errors
    ///
    /// Returns a [`RunError`] if:
    /// - The file cannot be read (`RunError::Io`)
    /// - The JSON cannot be parsed (`RunError::Json`)
    /// - The input shape does not match the expected architecture
    /// - The array cannot be flattened into 1-D
    ///
    fn read_inputs(
        &mut self,
        file_path: &str,
        assignment: CircuitType,
    ) -> Result<CircuitType, RunError>;
    /// Reads circuit outputs from a JSON file and applies them to a circuit assignment.
    ///
    /// # Arguments
    ///
    /// - `file_path` – Path to a JSON file containing expected output values.
    /// - `assignment` – The circuit assignment to populate.
    ///
    /// # Returns
    ///
    /// A new [`Circuit`] with its outputs populated from the JSON file.
    ///
    /// # Errors
    ///
    /// Returns a [`RunError`] if:
    /// - The file cannot be read (`RunError::Io`)
    /// - The JSON cannot be parsed (`RunError::Json`)
    /// - The output shape does not match the expected architecture
    /// - The array cannot be flattened into 1-D
    ///
    fn read_outputs(
        &mut self,
        file_path: &str,
        assignment: CircuitType,
    ) -> Result<CircuitType, RunError>;
    /// Applies inline JSON input/output values to a circuit assignment.
    ///
    /// # Errors
    ///
    /// Returns a [`RunError`] if deserialization or assignment fails.
    fn apply_values(
        &mut self,
        _input: Value,
        _output: Value,
        _assignment: CircuitType,
    ) -> Result<CircuitType, RunError> {
        Err(RunError::Json(
            "apply_values not implemented for this IOReader".into(),
        ))
    }

    fn get_path(&self) -> &str;
}
/// To implement `IOReader` in each binary to read in inputs and outputs of the circuit as is needed on an individual circuit basis
#[derive(Clone)]
pub struct FileReader {
    pub path: String,
}

pub mod onnx_context {
    use std::cell::RefCell;

    use thiserror::Error;

    use crate::circuit_functions::utils::onnx_model::{Architecture, CircuitParams, WANDB};

    thread_local! {
        static ARCHITECTURE: RefCell<Option<Architecture>> = const { RefCell::new(None) };
        static CIRCUITPARAMS: RefCell<Option<CircuitParams>> = const { RefCell::new(None) };
        static W_AND_B: RefCell<Option<WANDB>> = const { RefCell::new(None) };
    }

    #[derive(Debug, Error)]
    pub enum OnnxContextError {
        #[error("Architecture not set")]
        ArchitectureNotSet,
        #[error("Circuit parameters not set")]
        CircuitParamsNotSet,
        #[error("Weights & Biases (WANDB) not set")]
        WandbNotSet,
    }

    pub struct OnnxContext;

    impl OnnxContext {
        pub fn set_all(architecture: Architecture, params: CircuitParams, wandb: Option<WANDB>) {
            ARCHITECTURE.with(|a| *a.borrow_mut() = Some(architecture));
            CIRCUITPARAMS.with(|p| *p.borrow_mut() = Some(params));
            W_AND_B.with(|w| *w.borrow_mut() = wandb);
        }

        pub fn set_architecture(meta: Architecture) {
            ARCHITECTURE.with(|a| *a.borrow_mut() = Some(meta));
        }

        pub fn set_params(meta: CircuitParams) {
            CIRCUITPARAMS.with(|p| *p.borrow_mut() = Some(meta));
        }

        pub fn set_wandb(meta: WANDB) {
            W_AND_B.with(|w| *w.borrow_mut() = Some(meta));
        }

        pub fn clear_params() {
            CIRCUITPARAMS.with(|p| *p.borrow_mut() = None);
        }

        pub fn get_architecture() -> Result<Architecture, OnnxContextError> {
            ARCHITECTURE.with(|a| {
                a.borrow()
                    .clone()
                    .ok_or(OnnxContextError::ArchitectureNotSet)
            })
        }

        pub fn get_params() -> Result<CircuitParams, OnnxContextError> {
            CIRCUITPARAMS.with(|p| {
                p.borrow()
                    .clone()
                    .ok_or(OnnxContextError::CircuitParamsNotSet)
            })
        }

        pub fn get_wandb() -> Result<WANDB, OnnxContextError> {
            W_AND_B.with(|w| w.borrow().clone().ok_or(OnnxContextError::WandbNotSet))
        }
    }
}
