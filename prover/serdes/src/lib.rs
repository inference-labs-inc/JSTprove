#![allow(clippy::pedantic)]
pub mod error;
pub mod macros;
pub mod serdes;

pub use error::{SerdeError, SerdeResult};
pub use serdes::ExpSerde;
pub use serdes_derive::ExpSerde;
