// use crate::layer_handlers;

pub mod layer_ir;
pub mod layer_helpers;
mod extact_params;

// use layer_ir::LayerParams;
use tract_onnx::prelude::*;

use crate::layer_handlers::layer_ir::LayerParams;

use crate::layer_handlers::extact_params::ExtractParams;




// impl<T: ?Sized> ExtractParams for T {
//     default fn extract_params(&self) -> Option<LayerParams> {
//         None
//     }
// }

// Dispatch helper: Try downcast against known ops
// pub fn extract_layer_params(op: &dyn TypedOp) -> Option<LayerParams> {
//     if let Some(conv) = op.downcast_ref::<tract_core::ops::cnn::Conv>() {
//         return conv.extract_params();
//     }
//     if let Some(ein) = op.downcast_ref::<tract_core::ops::einsum::EinSum>() {
//         return ein.extract_params();
//     }
//     // ... Add more here

//     None
// }

macro_rules! impl_dispatch {
    ($op:expr, $( $ty:ty ),*) => {{
        let op = $op;
        $(
            if let Some(casted) = op.downcast_ref::<$ty>() {
                return casted.extract_params();
            }
        )*
        None
    }}
}

pub fn extract_layer_params(op: &dyn TypedOp) -> Option<LayerParams> {
    impl_dispatch!(op,
        tract_core::ops::cnn::Conv,
        tract_core::ops::einsum::EinSum
    )
}