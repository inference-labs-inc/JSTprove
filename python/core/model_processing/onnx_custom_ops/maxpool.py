from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as f
from onnxruntime_extensions import PyCustomOpDef, onnx_op

from .custom_helpers import parse_attr


@onnx_op(
    op_type="Int64MaxPool",
    domain="ai.onnx.contrib",
    inputs=[PyCustomOpDef.dt_int64],  # input tensor
    outputs=[PyCustomOpDef.dt_int64],
    attrs={
        "strides": PyCustomOpDef.dt_string,
        "pads": PyCustomOpDef.dt_string,
        "kernel_shape": PyCustomOpDef.dt_string,
        "dilations": PyCustomOpDef.dt_string,
    },
)
def int64_maxpool(
    x: np.ndarray,
    strides: str | None = None,
    pads: str | None = None,
    kernel_shape: str | None = None,
    dilations: str | None = None,
) -> np.ndarray:
    """
    Performs a MaxPool operation on int64 input tensors.

    This function is registered as a custom ONNX operator via onnxruntime_extensions
    and is used in the JSTprove quantized inference pipeline. It parses ONNX-style
    maxpool attributes and applies maxpool.

    Parameters
    ----------
    X : Input tensor with dtype int64.
    kernel_shape : Kernel shape (default: `[2, 2]`).
    pads : Padding values (default: `[0, 0, 0, 0]`).
    strides : Stride values (default: `[1, 1]`).
    dilations : dilation values (default: `[1, 1]`).


    Returns
    -------
    numpy.ndarray
        Maxpool tensor with dtype int64.

    Notes
    -----
    - This op is part of the `ai.onnx.contrib` custom domain.
    - ONNX Runtime Extensions is required to register this op.

    References
    ----------
    For more information on the maxpool operation, please refer to the
    ONNX standard MaxPool operator documentation:
    https://onnx.ai/onnx/operators/onnx__MaxPool.html
    """
    try:
        strides = parse_attr(strides, [1, 1])
        pads = parse_attr(pads, [0, 0])
        kernel_size = parse_attr(kernel_shape, [2, 2])
        dilations = parse_attr(dilations, [1, 1])

        x = torch.from_numpy(x)
        result = f.max_pool2d(
            x,
            kernel_size=kernel_size,
            stride=strides,
            padding=pads[:2],
            dilation=dilations,
        )
        return result.numpy().astype(np.int64)
    except Exception as e:
        msg = f"Int64MaxPool failed: {e}"
        raise RuntimeError(msg) from e
