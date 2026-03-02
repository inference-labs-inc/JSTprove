use serdes::SerdeError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExpErrors {
    #[error("Unknown string `{0}` for config enum deserialize")]
    SerializationError(String),

    #[error("Unknown string `{0}` for PCS type")]
    PCSTypeError(String),

    #[error("Unknown string `{0}` for FiatShamir Hash Type")]
    FiatShamirHashTypeError(String),

    #[error("field serde error: {0:?}")]
    SerdeError(#[from] SerdeError),

    #[error("other error: {0:?}")]
    OtherError(#[from] std::io::Error),
}
