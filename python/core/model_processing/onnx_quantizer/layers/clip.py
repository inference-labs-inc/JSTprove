# python/core/model_processing/onnx_quantizer/layers/clip.py
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import onnx

from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    QuantizerBase,
    ScaleConfig,
)


class QuantizeClip(QuantizerBase):
    OP_TYPE = "Clip"
    DOMAIN = ""  # standard ONNX domain
    USE_WB = False  # no W/B slots here
    USE_SCALING = False  # no internal scaling plan


class ClipQuantizer(BaseOpQuantizer, QuantizeClip):
    """Passthrough quantizer for ONNX Clip."""

    def __init__(
        self,
        new_initializers: dict[str, "onnx.TensorProto"] | None = None,
    ) -> None:
        # Same pattern as Max/Min/Add: just store the shared list/dict ref
        self.new_initializers = new_initializers

    def quantize(
        self,
        node: "onnx.NodeProto",
        graph: "onnx.GraphProto",
        scale_config: ScaleConfig,
        initializer_map: dict[str, "onnx.TensorProto"],
    ) -> list["onnx.NodeProto"]:
        # Delegate to the shared QuantizerBase logic (passthrough)
        return QuantizeClip.quantize(self, node, graph, scale_config, initializer_map)

    def check_supported(
        self,
        node: "onnx.NodeProto",
        initializer_map: dict[str, "onnx.TensorProto"] | None = None,
    ) -> None:
        # For now, accept any Clip node and defer detailed semantics to ORT
        _ = node, initializer_map
        return
