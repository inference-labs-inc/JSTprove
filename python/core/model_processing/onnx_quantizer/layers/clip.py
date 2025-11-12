# python/core/model_processing/onnx_quantizer/layers/clip.py
from __future__ import annotations

import onnx

from python.core.model_processing.onnx_quantizer.base import (
    BaseOpQuantizer,
    QuantizerBase,
    ScaleConfig,
)


class QuantizeClip(QuantizerBase):
    """
    Passthrough quantization traits for ONNX Clip.

    Rationale:
    - Inputs are already scaled & cast to INT64 by the converter at the graph boundary.
    - Clip is elementwise + broadcasting; no internal rescaling needed.
    - We allow weight/bias-style initializer scaling on slot 1/2 only if they appear
      as small scalar tensors (common ONNX forms), but we *don't* inject any new scaling
      here (USE_SCALING=False). In practice, Clip's extra inputs are bounds (min, max)
      that are constants and already integers (or will be cast by converter if they are inputs).
    """

    OP_TYPE = "Clip"
    DOMAIN = ""  # standard ONNX domain
    USE_WB = False  # no W/B logic for elementwise bounds
    USE_SCALING = False  # do not add an internal scale input
    # No SCALE_PLAN; we’re not touching bound inputs here.


class ClipQuantizer(BaseOpQuantizer, QuantizeClip):
    """
    Passthrough quantizer for ONNX Clip.
    """

    def __init__(self, new_initializers: list[onnx.TensorProto] | None = None) -> None:
        super().__init__()
        if new_initializers is not None:
            # Share the list with the converter so any constants we add are collected.
            self.new_initializers = new_initializers

    def quantize(
        self,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        # Passthrough: keep node as-is; converter already coerces dtypes globally.
        _ = graph, scale_config, initializer_map
        return QuantizeClip.quantize(self, node, graph, scale_config, initializer_map)

    def check_supported(
        self,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        """
        Be friendly but minimal:
        - Clip is variadic elementwise with optional min/max as inputs or attrs.
        - We accept both forms; if attrs are present, let runtime enforce semantics.
        - Broadcasting is fine (ONNX-standard).
        """
        _ = node, initializer_map
        return
