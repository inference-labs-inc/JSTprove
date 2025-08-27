use crate::circuit_functions::CircuitError;
use crate::circuit_functions::layers::LayerError;
use crate::circuit_functions::layers::layer_ops::LayerOp;
use crate::circuit_functions::utils::build_layers::BuildLayerContext;
use crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry;
use crate::circuit_functions::utils::onnx_model::CircuitParams;
use crate::circuit_functions::utils::onnx_types::ONNXLayer;

use crate::circuit_functions::layers::constant::ConstantLayer;
use crate::circuit_functions::layers::conv::ConvLayer;
use crate::circuit_functions::layers::flatten::FlattenLayer;
use crate::circuit_functions::layers::gemm::GemmLayer;
use crate::circuit_functions::layers::maxpool::MaxPoolLayer;
use crate::circuit_functions::layers::relu::ReluLayer;
use crate::circuit_functions::layers::reshape::ReshapeLayer;

use expander_compiler::frontend::{Config, RootAPI};
use std::str::FromStr;

// Macro to define layers
macro_rules! define_layers {
    (
        $( $variant:ident => {
            name: $name:expr,
            builder: $builder:path $(, aliases: [ $( $alias:expr ),* ] )?
        } ),* $(,)?
    ) => {

        // --------------------------
        // LayerKind enum
        // --------------------------
        #[derive(Debug, Clone)]
        pub enum LayerKind {
            $( $variant ),*
        }

        // --------------------------
        // Display impl
        // --------------------------
        impl std::fmt::Display for LayerKind {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $( LayerKind::$variant => write!(f, $name), )*
                }
            }
        }
        impl FromStr for LayerKind {
            type Err = LayerError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $(
                        $name => Ok(LayerKind::$variant),
                        $(
                            $( $alias => Ok(LayerKind::$variant), )*
                        )?
                    )*
                    other => Err(LayerError::UnknownOp { op_type: other.to_string() }),
                }
            }
        }

        // --------------------------
        // FromStr / TryFrom impls
        // --------------------------
        impl LayerKind {
            // pub fn from_str(s: &str) -> Result<Self, LayerError> {
            //     match s {
            //         $(
            //             $name => Ok(LayerKind::$variant),
            //             $(
            //                 $( $alias => Ok(LayerKind::$variant), )*
            //             )?
            //         )*
            //         other => Err(LayerError::UnknownOp { op_type: other.to_string() }),
            //     }
            // }

            #[must_use] pub fn builder<C: Config, Builder: RootAPI<C>>(
                &self
            ) -> fn(
                &ONNXLayer,
                &CircuitParams,
                PatternRegistry,
                bool,
                usize,
                &BuildLayerContext,
            ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
                match self {
                    $(
                        LayerKind::$variant => |layer, cp, opt, is_rescale, i, ctx| {
                            $builder(layer, cp, opt, is_rescale, i, ctx)
                        },
                    )*
                }
            }
        }

        impl TryFrom<&str> for LayerKind {
            type Error = LayerError;
            fn try_from(s: &str) -> Result<Self, Self::Error> {
                LayerKind::from_str(s)
            }
        }

        impl TryFrom<String> for LayerKind {
            type Error = LayerError;
            fn try_from(s: String) -> Result<Self, Self::Error> {
                LayerKind::from_str(&s)
            }
        }

        impl TryFrom<&ONNXLayer> for LayerKind {
            type Error = LayerError;
            fn try_from(layer: &ONNXLayer) -> Result<Self, Self::Error> {
                LayerKind::from_str(layer.op_type.as_str())
            }
        }
    }
}
/*
Layer Registry
When defining new layers, make sure to activate them by placing the new layer in the registry below.
*/

define_layers! {
    Constant => { name: "Constant", builder: ConstantLayer::build },
    Conv     => { name: "Conv",     builder: ConvLayer::build },
    Flatten  => { name: "Flatten",  builder: FlattenLayer::build },
    Gemm     => { name: "Gemm",     builder: GemmLayer::build },
    MaxPool  => { name: "MaxPool",  builder: MaxPoolLayer::build },
    ReLU     => { name: "ReLU",     builder: ReluLayer::build, aliases: ["Relu"] },
    Reshape  => { name: "Reshape",  builder: ReshapeLayer::build },
}
