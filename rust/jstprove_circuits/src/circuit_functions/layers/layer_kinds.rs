use crate::circuit_functions::CircuitError;
use crate::circuit_functions::layers::LayerError;
use crate::circuit_functions::layers::batchnorm::BatchnormLayer;
use crate::circuit_functions::layers::binary_arith::BinaryArithLayer;
use crate::circuit_functions::layers::binary_compare::BinaryCompareLayer;
use crate::circuit_functions::layers::div::DivLayer;
use crate::circuit_functions::layers::layer_ops::LayerOp;
use crate::circuit_functions::layers::mul::MulLayer;
use crate::circuit_functions::utils::build_layers::BuildLayerContext;
use crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry;
use crate::circuit_functions::utils::onnx_model::CircuitParams;
use crate::circuit_functions::utils::onnx_types::ONNXLayer;

use crate::circuit_functions::layers::clip::ClipLayer;
use crate::circuit_functions::layers::constant::ConstantLayer;
use crate::circuit_functions::layers::conv::ConvLayer;
use crate::circuit_functions::layers::flatten::FlattenLayer;
use crate::circuit_functions::layers::gemm::GemmLayer;
use crate::circuit_functions::layers::maxpool::MaxPoolLayer;
use crate::circuit_functions::layers::relu::ReluLayer;
use crate::circuit_functions::layers::reshape::ReshapeLayer;
use crate::circuit_functions::layers::squeeze::SqueezeLayer;
use crate::circuit_functions::layers::unsqueeze::UnsqueezeLayer;

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
            pub const SUPPORTED_OP_NAMES: &[&str] = &[ $( $name, )* ];

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
    Add       => { name: "Add", builder: BinaryArithLayer::build },
    Clip      => { name: "Clip", builder: ClipLayer::build },
    Batchnorm => { name: "BatchNormalization", builder: BatchnormLayer::build },
    Div       => { name: "Div", builder: DivLayer::build },
    Sub       => { name: "Sub", builder: BinaryArithLayer::build },
    Mul       => { name: "Mul", builder: MulLayer::build },
    Constant  => { name: "Constant", builder: ConstantLayer::build },
    Conv      => { name: "Conv", builder: ConvLayer::build },
    Flatten   => { name: "Flatten", builder: FlattenLayer::build },
    Gemm      => { name: "Gemm", builder: GemmLayer::build },
    MaxPool   => { name: "MaxPool", builder: MaxPoolLayer::build },
    Max       => { name: "Max", builder: BinaryCompareLayer::build },
    Min       => { name: "Min", builder: BinaryCompareLayer::build },
    ReLU      => { name: "ReLU", builder: ReluLayer::build, aliases: ["Relu"] },
    Reshape   => { name: "Reshape", builder: ReshapeLayer::build },
    Squeeze   => { name: "Squeeze", builder: SqueezeLayer::build },
    Unsqueeze => { name: "Unsqueeze", builder: UnsqueezeLayer::build },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_canonical_op_names() {
        let ops = [
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Conv",
            "Gemm",
            "Flatten",
            "Reshape",
            "MaxPool",
            "Clip",
            "Squeeze",
            "Unsqueeze",
            "BatchNormalization",
            "Constant",
            "Max",
            "Min",
            "ReLU",
        ];
        for name in ops {
            assert!(
                LayerKind::try_from(name).is_ok(),
                "expected Ok for op {name}"
            );
        }
    }

    #[test]
    fn parse_relu_alias() {
        let kind = LayerKind::try_from("Relu").unwrap();
        assert!(matches!(kind, LayerKind::ReLU));
    }

    #[test]
    fn parse_unknown_op_returns_error() {
        let err = LayerKind::try_from("Unknown");
        assert!(err.is_err());
    }

    #[test]
    fn display_matches_canonical_name() {
        assert_eq!(LayerKind::Add.to_string(), "Add");
        assert_eq!(LayerKind::Batchnorm.to_string(), "BatchNormalization");
        assert_eq!(LayerKind::ReLU.to_string(), "ReLU");
        assert_eq!(LayerKind::Gemm.to_string(), "Gemm");
    }

    #[test]
    fn parse_from_string_owned() {
        let kind = LayerKind::try_from("Gemm".to_string()).unwrap();
        assert!(matches!(kind, LayerKind::Gemm));
    }

    #[test]
    fn supported_op_names_contains_all_canonical() {
        let expected = [
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Conv",
            "Gemm",
            "Flatten",
            "Reshape",
            "MaxPool",
            "Clip",
            "Squeeze",
            "Unsqueeze",
            "BatchNormalization",
            "Constant",
            "Max",
            "Min",
            "ReLU",
        ];
        for name in expected {
            assert!(
                LayerKind::SUPPORTED_OP_NAMES.contains(&name),
                "expected {name} in SUPPORTED_OP_NAMES"
            );
        }
    }
}
