use thiserror::Error;

#[derive(Error, Debug)]
pub enum CliError {
    #[error("Missing required argument: {0}")]
    MissingArgument(&'static str),

    #[error("Unknown command: {0}")]
    UnknownCommand(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Other error: {0}")]
    Other(String),
}
