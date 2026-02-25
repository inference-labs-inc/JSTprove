use crate::runner::errors::RunError;
use expander_compiler::frontend::Config;
use expander_compiler::frontend::internal::DumpLoadTwoVariables;
use expander_compiler::gkr_engine::{FieldEngine, GKREngine};
use rmpv::Value;
use serde::de::DeserializeOwned;
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
    /// Reads and deserializes a msgpack file into a strongly typed struct.
    ///
    /// This helper function:
    /// - Opens the file at `file_path`.
    /// - Reads its contents into bytes.
    /// - Uses [`rmp_serde`] to deserialize the bytes into the target type `I`.
    /// - Converts any I/O or deserialization failures into [`RunError`].
    ///
    /// # Type Parameters
    ///
    /// - `I` – The type to deserialize into. Must implement [`DeserializeOwned`].
    ///
    /// # Arguments
    ///
    /// - `file_path` – Path to the msgpack file to read.
    ///
    /// # Returns
    ///
    /// An instance of type `I` populated from the msgpack contents.
    ///
    /// # Errors
    ///
    /// Returns a [`RunError`] if:
    /// - The file cannot be opened or read (`RunError::Io`).
    /// - The msgpack cannot be parsed into the expected type (`RunError::Deserialize`).
    fn read_data_from_msgpack<I>(file_path: &str) -> Result<I, RunError>
    where
        I: DeserializeOwned,
    {
        let mut file = std::fs::File::open(file_path).map_err(|e| RunError::Io {
            source: e,
            path: file_path.into(),
        })?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents).map_err(|e| RunError::Io {
            source: e,
            path: file_path.into(),
        })?;
        rmp_serde::from_slice(&contents).map_err(|e| RunError::Deserialize(format!("{e:?}")))
    }
    /// Reads circuit inputs from a msgpack file and applies them to a circuit assignment.
    ///
    /// # Arguments
    ///
    /// - `file_path` – Path to a msgpack file containing input values.
    /// - `assignment` – The circuit assignment to populate.
    ///
    /// # Returns
    ///
    /// A new [`Circuit`] with its inputs populated from the msgpack file.
    ///
    /// # Errors
    ///
    /// Returns a [`RunError`] if:
    /// - The file cannot be read (`RunError::Io`)
    /// - The msgpack cannot be parsed (`RunError::Deserialize`)
    /// - The input shape does not match the expected architecture
    /// - The array cannot be flattened into 1-D
    ///
    fn read_inputs(
        &mut self,
        file_path: &str,
        assignment: CircuitType,
    ) -> Result<CircuitType, RunError>;
    /// Reads circuit outputs from a msgpack file and applies them to a circuit assignment.
    ///
    /// # Arguments
    ///
    /// - `file_path` – Path to a msgpack file containing expected output values.
    /// - `assignment` – The circuit assignment to populate.
    ///
    /// # Returns
    ///
    /// A new [`Circuit`] with its outputs populated from the msgpack file.
    ///
    /// # Errors
    ///
    /// Returns a [`RunError`] if:
    /// - The file cannot be read (`RunError::Io`)
    /// - The msgpack cannot be parsed (`RunError::Deserialize`)
    /// - The output shape does not match the expected architecture
    /// - The array cannot be flattened into 1-D
    ///
    fn read_outputs(
        &mut self,
        file_path: &str,
        assignment: CircuitType,
    ) -> Result<CircuitType, RunError>;
    /// Applies inline msgpack input/output values to a circuit assignment.
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

        /// # Errors
        /// Returns `OnnxContextError::ArchitectureNotSet` if uninitialized.
        pub fn get_architecture() -> Result<Architecture, OnnxContextError> {
            ARCHITECTURE.with(|a| {
                a.borrow()
                    .clone()
                    .ok_or(OnnxContextError::ArchitectureNotSet)
            })
        }

        /// # Errors
        /// Returns `OnnxContextError::CircuitParamsNotSet` if uninitialized.
        pub fn get_params() -> Result<CircuitParams, OnnxContextError> {
            CIRCUITPARAMS.with(|p| {
                p.borrow()
                    .clone()
                    .ok_or(OnnxContextError::CircuitParamsNotSet)
            })
        }

        /// # Errors
        /// Returns `OnnxContextError::WandbNotSet` if uninitialized.
        pub fn get_wandb() -> Result<WANDB, OnnxContextError> {
            W_AND_B.with(|w| w.borrow().clone().ok_or(OnnxContextError::WandbNotSet))
        }
    }
}
