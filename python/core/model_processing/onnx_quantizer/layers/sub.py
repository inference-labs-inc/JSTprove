from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import onnx

from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    QuantizerBase,
    ScaleConfig,
)


class QuantizeSub(QuantizerBase):
    OP_TYPE = "Sub"
    DOMAIN = ""
    USE_WB = True
    USE_SCALING = False
    SCALE_PLAN: ClassVar = {0: 1, 1: 1}


class SubQuantizer(BaseOpQuantizer, QuantizeSub):
    """
    Quantizer for ONNX Sub layers.

    - Uses standard ONNX Sub layer in standard domain, and
      makes relevant additional changes to the graph.
    """

    def __init__(
        self: SubQuantizer,
        new_initializers: list[onnx.TensorProto] | None = None,
    ) -> None:
        super().__init__()
        # Only replace if caller provided something
        if new_initializers is not None:
            self.new_initializers = new_initializers

    def quantize(
        self: SubQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        return QuantizeSub.quantize(self, node, graph, scale_config, initializer_map)

    def check_supported(
        self: SubQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        pass
