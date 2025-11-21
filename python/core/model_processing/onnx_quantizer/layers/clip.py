from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import onnx

from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    QuantizerBase,
    ScaleConfig,
)


class QuantizeClip(QuantizerBase):
    """
    Quantization traits for ONNX Clip.

    Semantics:
    - X is already scaled/cast to INT64 at the graph boundary by the converter.
    - Clip is elementwise + broadcasting.
    - The bound inputs (min, max) should live in the *same* fixed-point scale
      as X so that Clip(alpha*x; alpha*a, alpha*b) matches the original Clip(x; a, b).

    Implementation:
    - Treat inputs 1 and 2 (min, max) like "WB-style" slots: we let the
      QuantizerBase machinery rescale / cast those inputs using the same
      global scale factor.
    - No extra internal scaling input is added (USE_SCALING = False).
    """

    OP_TYPE = "Clip"
    DOMAIN = ""  # standard ONNX domain

    # We DO want WB-style handling so that min/max initializers get quantized:
    USE_WB = True

    # Clip does not introduce its own scale input; it just runs in the
    # existing fixed-point scale.
    USE_SCALING = False

    # Scale-plan for WB-style slots:
    #   - Input index 1: min
    #   - Input index 2: max
    # Each should be scaled once by the global alpha (same as activations).
    SCALE_PLAN: ClassVar = {1: 1, 2: 1}


class ClipQuantizer(BaseOpQuantizer, QuantizeClip):
    """
    Quantizer for ONNX Clip.

    - Keeps the node op_type as "Clip".
    - Ensures that any bound inputs (min, max), whether they are dynamic
      inputs or initializers, are converted to the same INT64 fixed-point
      representation as A.
    """

    def __init__(
        self,
        new_initializers: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        # Match Max/Min/Add: we simply share the new_initializers dict
        # with the converter so any constants we add are collected.
        self.new_initializers = new_initializers

    def quantize(
        self,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        # Delegate to the shared QuantizerBase logic, which will:
        # - keep X as-is (already scaled/cast by the converter),
        # - rescale / cast min/max according to SCALE_PLAN,
        # - update initializers as needed.
        return QuantizeClip.quantize(self, node, graph, scale_config, initializer_map)

    def check_supported(
        self,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        """
        Minimal support check for Clip:

        - Clip is variadic elementwise with optional min/max as inputs or attrs.
        - We accept both forms; if attrs are present, ORT enforces semantics.
        - Broadcasting is ONNX-standard; we don't restrict further here.
        """
        _ = node, initializer_map
