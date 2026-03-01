pub mod builtin;
pub mod registry;

pub use builtin::*;

use registry::HintCaller;

use crate::{field::Field, utils::error::Error};

pub fn safe_impl<F: Field>(
    hint_caller: &impl HintCaller<F>,
    hint_id: usize,
    inputs: &[F],
    num_outputs: usize,
) -> Result<Vec<F>, Error> {
    match BuiltinHintIds::from_usize(hint_id) {
        Some(hint_id) => Ok(impl_builtin_hint(hint_id, inputs, num_outputs)),
        None => hint_caller.call(hint_id, inputs, num_outputs),
    }
}
