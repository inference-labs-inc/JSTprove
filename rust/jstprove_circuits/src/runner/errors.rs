use std::io;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CliError {
    #[error("Missing required argument: {0}")]
    MissingArgument(&'static str),

    #[error("Unknown command: {0}")]
    UnknownCommand(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    RunError(#[from] RunError),

    #[error("Other error: {0}")]
    Other(String),
}

#[derive(Debug, Error)]
pub enum RunError {
    #[error("I/O error while accessing {path}: {source}")]
    Io {
        #[source]
        source: io::Error,
        path: String,
    },

    #[error("JSON deserialization error: {0}")]
    Json(String),

    #[error("Circuit compilation failed: {0}")]
    Compile(String),

    #[error("Serialization error: {0}")]
    Serialize(String),

    #[error("Deserialization error: {0}")]
    Deserialize(String),

    #[error("Witness generation failed: {0}")]
    Witness(String),

    #[error("Proving witness failed: {0}")]
    Prove(String),

    #[error("Verifying proof failed: {0}")]
    Verify(String),
}
