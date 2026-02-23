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
    use std::sync::RwLock;

    use thiserror::Error;

    use crate::circuit_functions::utils::onnx_model::{Architecture, CircuitParams, WANDB};

    static ARCHITECTURE: RwLock<Option<Architecture>> = RwLock::new(None);
    static CIRCUITPARAMS: RwLock<Option<CircuitParams>> = RwLock::new(None);
    static W_AND_B: RwLock<Option<WANDB>> = RwLock::new(None);

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
        /// Acquires write locks on ARCHITECTURE, CIRCUITPARAMS, and W_AND_B
        /// (sequentially, in that order) before performing any mutations,
        /// preventing partial-update races. Note: a concurrent reader (e.g.
        /// `get_params()`) may observe stale values during lock acquisition.
        /// Use this when updating the full context (e.g. per-slice).
        pub fn set_all(architecture: Architecture, params: CircuitParams, wandb: Option<WANDB>) {
            let mut arch_guard = ARCHITECTURE.write().unwrap_or_else(|e| e.into_inner());
            let mut params_guard = CIRCUITPARAMS.write().unwrap_or_else(|e| e.into_inner());
            let mut wandb_guard = W_AND_B.write().unwrap_or_else(|e| e.into_inner());
            *arch_guard = Some(architecture);
            *params_guard = Some(params);
            *wandb_guard = wandb;
        }

        /// Overwrites the stored architecture. WARNING: concurrent partial
        /// updates via individual setters are not atomic — prefer `set_all`
        /// when updating multiple fields, or ensure external synchronization.
        pub fn set_architecture(meta: Architecture) {
            let mut guard = ARCHITECTURE.write().unwrap_or_else(|e| e.into_inner());
            *guard = Some(meta);
        }

        /// Overwrites the stored circuit parameters. WARNING: concurrent
        /// partial updates via individual setters are not atomic — prefer
        /// `set_all` or ensure external synchronization.
        pub fn set_params(meta: CircuitParams) {
            let mut guard = CIRCUITPARAMS.write().unwrap_or_else(|e| e.into_inner());
            *guard = Some(meta);
        }

        /// Overwrites the stored weights & biases. WARNING: concurrent
        /// partial updates via individual setters are not atomic — prefer
        /// `set_all` or ensure external synchronization.
        pub fn set_wandb(meta: WANDB) {
            let mut guard = W_AND_B.write().unwrap_or_else(|e| e.into_inner());
            *guard = Some(meta);
        }

        pub fn get_architecture() -> Result<Architecture, OnnxContextError> {
            let guard = ARCHITECTURE.read().unwrap_or_else(|e| e.into_inner());
            guard.clone().ok_or(OnnxContextError::ArchitectureNotSet)
        }

        pub fn get_params() -> Result<CircuitParams, OnnxContextError> {
            let guard = CIRCUITPARAMS.read().unwrap_or_else(|e| e.into_inner());
            guard.clone().ok_or(OnnxContextError::CircuitParamsNotSet)
        }

        pub fn get_wandb() -> Result<WANDB, OnnxContextError> {
            let guard = W_AND_B.read().unwrap_or_else(|e| e.into_inner());
            guard.clone().ok_or(OnnxContextError::WandbNotSet)
        }
    }
}
