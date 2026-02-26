#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::collapsible_if,
    clippy::doc_markdown,
    clippy::filter_map_identity,
    clippy::for_kv_map,
    clippy::if_not_else,
    clippy::implicit_hasher,
    clippy::items_after_statements,
    clippy::manual_filter_map,
    clippy::manual_let_else,
    clippy::map_entry,
    clippy::map_unwrap_or,
    clippy::match_same_arms,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::manual_memcpy,
    clippy::needless_pass_by_value,
    clippy::needless_borrow,
    clippy::needless_range_loop,
    clippy::range_plus_one,
    clippy::redundant_closure_for_method_calls,
    clippy::should_implement_trait,
    clippy::similar_names,
    clippy::single_match_else,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::trivially_copy_pass_by_ref,
    clippy::uninlined_format_args,
    clippy::unnecessary_filter_map,
    clippy::unnecessary_wraps,
    clippy::unreadable_literal,
    clippy::wildcard_imports
)]

pub mod gadgets;
pub mod onnx;
pub mod padding;
pub mod runner;
pub mod util;

pub use shared_types::Fr;
