from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import onnx

from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    QuantizerBase,
    ScaleConfig,
)


class QuantizeMul(QuantizerBase):
    OP_TYPE = "Int64Mul"
    USE_WB = True
    USE_SCALING = True
    SCALE_PLAN: ClassVar = {0: 1, 1: 1}


class MulQuantizer(BaseOpQuantizer, QuantizeMul):
    """
    Quantizer for ONNX Mul layers.

    - Uses custom Mul layer to incorporate rescaling, and
      makes relevant additional changes to the graph.
    """

    def __init__(
        self: MulQuantizer,
        new_initializers: list[onnx.TensorProto] | None = None,
    ) -> None:
        super().__init__()
        # Only replace if caller provided something
        if new_initializers is not None:
            self.new_initializers = new_initializers

    def quantize(
        self: MulQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        return QuantizeMul.quantize(self, node, graph, scale_config, initializer_map)

    def check_supported(
        self: MulQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        pass
