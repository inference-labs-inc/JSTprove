from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import onnx

from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    QuantizerBase,
    ScaleConfig,
)


class QuantizeMax(QuantizerBase):
    OP_TYPE = "Max"
    DOMAIN = ""
    USE_WB = True
    USE_SCALING = False
    SCALE_PLAN: ClassVar = {1: 1}


class MaxQuantizer(BaseOpQuantizer, QuantizeMax):
    def __init__(
        self,
        new_initializers: list[onnx.TensorProto] | None = None,
    ) -> None:
        super().__init__()
        if new_initializers is not None:
            # Share the caller-provided buffer instead of the default list.
            self.new_initializers = new_initializers

    def quantize(
        self,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        # Delegate to the shared QuantizerBase logic
        return QuantizeMax.quantize(self, node, graph, scale_config, initializer_map)

    def check_supported(
        self,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        # If later we want to enforce/relax broadcasting, add it here.
        pass
