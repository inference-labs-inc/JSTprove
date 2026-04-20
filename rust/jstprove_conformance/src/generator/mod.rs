pub mod builder;
pub mod cases_m3;
pub mod cases_m4;
pub mod op_specs;
pub mod shapes;
pub mod values;

pub use builder::{shrink, TestCaseBuilder, DEFAULT_SEEDS};
pub use cases_m3::{arithmetic_cases, boolean_cases, reduction_cases, structural_cases};
pub use cases_m4::{
    norm_cases, pooling_cases, rescaling_cases, spatial_cases, topk_cases, transcendental_cases,
};
pub use op_specs::{add_spec, gemm_spec, relu_spec, OpInputSpec, TensorSpec};
pub use shapes::{broadcast_pair, edge_case_shapes, incompatible_pair, ShapeSpec};
pub use values::{ValueSpec, ALPHA, SAFE_RANGE};

/// Maximum number of test cases to generate per operator group.
///
/// When the `ci` feature is enabled, this returns a small number to keep the
/// full harness under the 5-minute CI budget.  Locally, all cases run.
pub fn default_case_count() -> usize {
    if cfg!(feature = "ci") {
        5
    } else {
        usize::MAX
    }
}
