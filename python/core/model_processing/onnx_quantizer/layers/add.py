from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import onnx

from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    QuantizerBase,
    ScaleConfig,
)


class QuantizeAdd(QuantizerBase):
    OP_TYPE = "Add"
    DOMAIN = ""
    USE_WB = True
    USE_SCALING = False
    SCALE_PLAN: ClassVar = {1: 1}


class AddQuantizer(BaseOpQuantizer, QuantizeAdd):
    """
    Quantizer for ONNX Gemm layers.

    - Replaces standard Gemm with Int64Gemm from the `ai.onnx.contrib`
        domain and makes relevant additional changes to the graph.
    - Validates that all required Gemm parameters are present.
    """

    def __init__(
        self: AddQuantizer,
        new_initializers: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        self.new_initializers = new_initializers

    def quantize(
        self: AddQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        return QuantizeAdd.quantize(self, node, graph, scale_config, initializer_map)

    def check_supported(
        self: AddQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        pass
