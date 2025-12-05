from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import onnx

from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    QuantizerBase,
    ScaleConfig,
)


class QuantizeMin(QuantizerBase):
    OP_TYPE = "Min"
    DOMAIN = ""  # standard ONNX domain
    USE_WB = True  # let framework wire inputs/outputs normally
    USE_SCALING = False  # passthrough: no internal scaling
    SCALE_PLAN: ClassVar = {1: 1}  # elementwise arity plan


class MinQuantizer(BaseOpQuantizer, QuantizeMin):
    """
    Passthrough quantizer for elementwise Min.
    We rely on the converter to quantize graph inputs; no extra scaling here.
    """

    def __init__(
        self: MinQuantizer,
        new_initializers: list[onnx.TensorProto] | None = None,
    ) -> None:
        super().__init__()
        if new_initializers is not None:
            self.new_initializers = new_initializers

    def quantize(
        self: MinQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        # Delegate to QuantizerBase's generic passthrough implementation.
        return QuantizeMin.quantize(self, node, graph, scale_config, initializer_map)

    def check_supported(
        self: MinQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        # Min has no attributes; elementwise, variadic ≥ 1 input per ONNX spec.
        # We mirror Add/Max broadcasting behavior; no extra checks here.
        _ = node, initializer_map
