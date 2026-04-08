use crate::circuit_functions::CircuitError;
use crate::circuit_functions::layers::LayerError;
use crate::circuit_functions::layers::batchnorm::BatchnormLayer;
use crate::circuit_functions::layers::binary_arith::BinaryArithLayer;
use crate::circuit_functions::layers::binary_compare::BinaryCompareLayer;
use crate::circuit_functions::layers::cast::CastLayer;
use crate::circuit_functions::layers::concat::ConcatLayer;
use crate::circuit_functions::layers::div::DivLayer;
use crate::circuit_functions::layers::layer_ops::LayerOp;
use crate::circuit_functions::layers::mul::MulLayer;
use crate::circuit_functions::utils::build_layers::BuildLayerContext;
use crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry;
use crate::circuit_functions::utils::onnx_model::CircuitParams;
use crate::circuit_functions::utils::onnx_types::ONNXLayer;

use crate::circuit_functions::layers::averagepool::AveragePoolLayer;
use crate::circuit_functions::layers::clip::ClipLayer;
use crate::circuit_functions::layers::constant::ConstantLayer;
use crate::circuit_functions::layers::conv::ConvLayer;
use crate::circuit_functions::layers::conv_transpose::ConvTransposeLayer;
use crate::circuit_functions::layers::erf::ErfLayer;
use crate::circuit_functions::layers::exp::ExpLayer;
use crate::circuit_functions::layers::expand::ExpandLayer;
use crate::circuit_functions::layers::flatten::FlattenLayer;
use crate::circuit_functions::layers::gather::GatherLayer;
use crate::circuit_functions::layers::gelu::GeluLayer;
use crate::circuit_functions::layers::gemm::GemmLayer;
use crate::circuit_functions::layers::global_averagepool::GlobalAveragePoolLayer;
use crate::circuit_functions::layers::gridsample::GridSampleLayer;
use crate::circuit_functions::layers::group_norm::GroupNormLayer;
use crate::circuit_functions::layers::hardswish::HardSwishLayer;
use crate::circuit_functions::layers::identity::IdentityLayer;
use crate::circuit_functions::layers::instance_norm::InstanceNormLayer;
use crate::circuit_functions::layers::layer_norm::LayerNormLayer;
use crate::circuit_functions::layers::leaky_relu::LeakyReluLayer;
use crate::circuit_functions::layers::log::LogLayer;
use crate::circuit_functions::layers::matmul::MatMulLayer;
use crate::circuit_functions::layers::maxpool::MaxPoolLayer;
use crate::circuit_functions::layers::neg::NegLayer;
use crate::circuit_functions::layers::pad::PadLayer;
use crate::circuit_functions::layers::pow::PowLayer;
use crate::circuit_functions::layers::reduce_mean::ReduceMeanLayer;
use crate::circuit_functions::layers::reduce_sum::ReduceSumLayer;
use crate::circuit_functions::layers::relu::ReluLayer;
use crate::circuit_functions::layers::reshape::ReshapeLayer;
use crate::circuit_functions::layers::resize::ResizeLayer;
use crate::circuit_functions::layers::shape::ShapeLayer;
use crate::circuit_functions::layers::sigmoid::SigmoidLayer;
use crate::circuit_functions::layers::slice::SliceLayer;
use crate::circuit_functions::layers::softmax::SoftmaxLayer;
use crate::circuit_functions::layers::split::SplitLayer;
use crate::circuit_functions::layers::sqrt::SqrtLayer;
use crate::circuit_functions::layers::squeeze::SqueezeLayer;
use crate::circuit_functions::layers::tanh::TanhLayer;
use crate::circuit_functions::layers::tile::TileLayer;
use crate::circuit_functions::layers::topk::TopKLayer;
use crate::circuit_functions::layers::transpose::TransposeLayer;
use crate::circuit_functions::layers::unsqueeze::UnsqueezeLayer;
use crate::circuit_functions::layers::where_op::WhereLayer;

use crate::circuit_functions::layers::and_op::AndLayer;
use crate::circuit_functions::layers::constant_of_shape::ConstantOfShapeLayer;
use crate::circuit_functions::layers::equal::EqualLayer;
use crate::circuit_functions::layers::greater::GreaterLayer;
use crate::circuit_functions::layers::less::LessLayer;
use crate::circuit_functions::layers::not_op::NotLayer;

use crate::circuit_functions::layers::cos::CosLayer;
use crate::circuit_functions::layers::gather_elements::GatherElementsLayer;
use crate::circuit_functions::layers::range::RangeLayer;
use crate::circuit_functions::layers::reduce_max::ReduceMaxLayer;
use crate::circuit_functions::layers::scatter_nd::ScatterNDLayer;
use crate::circuit_functions::layers::sin::SinLayer;

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

            #[must_use]
            pub const fn name(&self) -> &'static str {
                match self {
                    $( LayerKind::$variant => $name, )*
                }
            }

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
    Add                => { name: "Add", builder: BinaryArithLayer::build },
    Cast               => { name: "Cast", builder: CastLayer::build },
    Concat             => { name: "Concat", builder: ConcatLayer::build },
    Clip               => { name: "Clip", builder: ClipLayer::build },
    Batchnorm          => { name: "BatchNormalization", builder: BatchnormLayer::build },
    Div                => { name: "Div", builder: DivLayer::build },
    Exp                => { name: "Exp", builder: ExpLayer::build },
    Sub                => { name: "Sub", builder: BinaryArithLayer::build },
    Mul                => { name: "Mul", builder: MulLayer::build },
    Constant           => { name: "Constant", builder: ConstantLayer::build },
    Conv               => { name: "Conv", builder: ConvLayer::build },
    Flatten            => { name: "Flatten", builder: FlattenLayer::build },
    Gather             => { name: "Gather", builder: GatherLayer::build },
    Gelu               => { name: "Gelu", builder: GeluLayer::build },
    GridSample         => { name: "GridSample", builder: GridSampleLayer::build },
    Gemm               => { name: "Gemm", builder: GemmLayer::build },
    LayerNormalization => { name: "LayerNormalization", builder: LayerNormLayer::build },
    MaxPool            => { name: "MaxPool", builder: MaxPoolLayer::build },
    Max                => { name: "Max", builder: BinaryCompareLayer::build },
    Min                => { name: "Min", builder: BinaryCompareLayer::build },
    ReLU               => { name: "ReLU", builder: ReluLayer::build, aliases: ["Relu"] },
    Reshape            => { name: "Reshape", builder: ReshapeLayer::build },
    Resize             => { name: "Resize", builder: ResizeLayer::build },
    Sigmoid            => { name: "Sigmoid", builder: SigmoidLayer::build },
    Softmax            => { name: "Softmax", builder: SoftmaxLayer::build },
    Squeeze            => { name: "Squeeze", builder: SqueezeLayer::build },
    Slice              => { name: "Slice", builder: SliceLayer::build },
    Tile               => { name: "Tile", builder: TileLayer::build },
    TopK               => { name: "TopK", builder: TopKLayer::build },
    Transpose          => { name: "Transpose", builder: TransposeLayer::build },
    Unsqueeze          => { name: "Unsqueeze", builder: UnsqueezeLayer::build },
    Expand             => { name: "Expand", builder: ExpandLayer::build },
    Log                => { name: "Log", builder: LogLayer::build },
    ReduceMean         => { name: "ReduceMean", builder: ReduceMeanLayer::build },
    Shape              => { name: "Shape", builder: ShapeLayer::build },
    MatMul             => { name: "MatMul", builder: MatMulLayer::build },
    AveragePool        => { name: "AveragePool", builder: AveragePoolLayer::build },
    Pad                => { name: "Pad", builder: PadLayer::build },
    Split              => { name: "Split", builder: SplitLayer::build },
    Where              => { name: "Where", builder: WhereLayer::build },
    Pow                => { name: "Pow", builder: PowLayer::build },
    Sqrt               => { name: "Sqrt", builder: SqrtLayer::build },
    Tanh               => { name: "Tanh", builder: TanhLayer::build },
    ReduceSum          => { name: "ReduceSum", builder: ReduceSumLayer::build },
    Erf                => { name: "Erf", builder: ErfLayer::build },
    ConvTranspose      => { name: "ConvTranspose", builder: ConvTransposeLayer::build },
    LeakyRelu          => { name: "LeakyRelu", builder: LeakyReluLayer::build, aliases: ["LeakyReLU"] },
    Identity           => { name: "Identity", builder: IdentityLayer::build },
    Neg                => { name: "Neg", builder: NegLayer::build },
    HardSwish          => { name: "HardSwish", builder: HardSwishLayer::build, aliases: ["Hardswish"] },
    GlobalAveragePool  => { name: "GlobalAveragePool", builder: GlobalAveragePoolLayer::build },
    InstanceNormalization => { name: "InstanceNormalization", builder: InstanceNormLayer::build },
    GroupNormalization => { name: "GroupNormalization", builder: GroupNormLayer::build },
    Not                => { name: "Not", builder: NotLayer::build },
    And                => { name: "And", builder: AndLayer::build },
    Equal              => { name: "Equal", builder: EqualLayer::build },
    Greater            => { name: "Greater", builder: GreaterLayer::build },
    Less               => { name: "Less", builder: LessLayer::build },
    ConstantOfShape    => { name: "ConstantOfShape", builder: ConstantOfShapeLayer::build },
    Sin                => { name: "Sin", builder: SinLayer::build },
    Cos                => { name: "Cos", builder: CosLayer::build },
    Range              => { name: "Range", builder: RangeLayer::build },
    ReduceMax          => { name: "ReduceMax", builder: ReduceMaxLayer::build },
    ScatterND          => { name: "ScatterND", builder: ScatterNDLayer::build },
    GatherElements     => { name: "GatherElements", builder: GatherElementsLayer::build },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_canonical_op_names() {
        let ops = [
            "Add",
            "Cast",
            "Concat",
            "Sub",
            "Mul",
            "Div",
            "Exp",
            "Conv",
            "Gather",
            "Gelu",
            "Gemm",
            "Flatten",
            "LayerNormalization",
            "Reshape",
            "MaxPool",
            "Clip",
            "Sigmoid",
            "Softmax",
            "Squeeze",
            "Tile",
            "TopK",
            "Unsqueeze",
            "BatchNormalization",
            "Constant",
            "Max",
            "Min",
            "ReLU",
            "Resize",
            "GridSample",
            "Slice",
            "Transpose",
            "Expand",
            "Log",
            "ReduceMean",
            "Shape",
            "MatMul",
            "AveragePool",
            "Pad",
            "Split",
            "Where",
            "Pow",
            "Sqrt",
            "Tanh",
            "ReduceSum",
            "Erf",
            "ConvTranspose",
            "LeakyRelu",
            "Identity",
            "Neg",
            "HardSwish",
            "GlobalAveragePool",
            "InstanceNormalization",
            "GroupNormalization",
            "Not",
            "And",
            "Equal",
            "Greater",
            "Less",
            "ConstantOfShape",
            "Sin",
            "Cos",
            "Range",
            "ReduceMax",
            "ScatterND",
            "GatherElements",
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
        assert_eq!(LayerKind::Exp.to_string(), "Exp");
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
            "Cast",
            "Concat",
            "Sub",
            "Mul",
            "Div",
            "Exp",
            "Conv",
            "Gather",
            "Gelu",
            "Gemm",
            "Flatten",
            "LayerNormalization",
            "Reshape",
            "MaxPool",
            "Clip",
            "Sigmoid",
            "Softmax",
            "Squeeze",
            "Tile",
            "TopK",
            "Unsqueeze",
            "BatchNormalization",
            "Constant",
            "Max",
            "Min",
            "ReLU",
            "Resize",
            "GridSample",
            "Slice",
            "Transpose",
            "Expand",
            "Log",
            "ReduceMean",
            "Shape",
            "MatMul",
            "AveragePool",
            "Pad",
            "Split",
            "Where",
            "Pow",
            "Sqrt",
            "Tanh",
            "ReduceSum",
            "Erf",
            "ConvTranspose",
            "LeakyRelu",
            "Identity",
            "Neg",
            "HardSwish",
            "GlobalAveragePool",
            "InstanceNormalization",
            "GroupNormalization",
            "Not",
            "And",
            "Equal",
            "Greater",
            "Less",
            "ConstantOfShape",
            "Sin",
            "Cos",
            "Range",
            "ReduceMax",
            "ScatterND",
            "GatherElements",
        ];
        for name in expected {
            assert!(
                LayerKind::SUPPORTED_OP_NAMES.contains(&name),
                "expected {name} in SUPPORTED_OP_NAMES"
            );
        }
    }
}
