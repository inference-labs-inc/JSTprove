#![allow(clippy::pedantic)]

pub mod gadgets;
#[allow(
    clippy::collapsible_if,
    clippy::should_implement_trait,
    clippy::unnecessary_filter_map
)]
pub mod onnx;
pub mod padding;
pub mod runner;
pub mod util;

pub use shared_types::Fr;
