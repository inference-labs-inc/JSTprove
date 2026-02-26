pub mod batch;
#[allow(
    clippy::manual_memcpy,
    clippy::map_entry,
    clippy::needless_borrow,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]
pub mod circuit_builder;
pub mod compile;
pub mod pipe;
pub mod prove;
pub mod verify;
#[allow(
    clippy::for_kv_map,
    clippy::manual_memcpy,
    clippy::map_entry,
    clippy::needless_range_loop
)]
pub mod witness;
