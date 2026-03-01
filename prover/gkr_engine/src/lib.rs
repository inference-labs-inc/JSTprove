#![allow(clippy::pedantic)]
#![allow(clippy::manual_div_ceil)]
#![feature(associated_type_defaults)]

mod errors;
mod field_engine;
mod mpi_engine;
mod poly_commit;
mod scheme;
mod transcript;

pub use errors::*;
pub use field_engine::*;
pub use mpi_engine::*;
pub use poly_commit::*;
pub use scheme::*;
pub use transcript::*;

pub trait GKREngine: Send + Sync {
    type FieldConfig: FieldEngine;

    type MPIConfig: MPIEngine;

    type TranscriptConfig: Transcript;

    type PCSConfig: ExpanderPCS<Self::FieldConfig>;

    const SCHEME: GKRScheme;
}
