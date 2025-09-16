from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as f
from onnxruntime_extensions import PyCustomOpDef, onnx_op

from .custom_helpers import parse_attr, rescaling


@onnx_op(
    op_type="Int64Conv",
    domain="ai.onnx.contrib",
    inputs=[
        PyCustomOpDef.dt_int64,  # X
        PyCustomOpDef.dt_int64,  # W
        PyCustomOpDef.dt_int64,  # B
        PyCustomOpDef.dt_int64,  # scaling factor
    ],
    outputs=[PyCustomOpDef.dt_int64],
    attrs={
        "auto_pad": PyCustomOpDef.dt_string,
        "strides": PyCustomOpDef.dt_string,
        "pads": PyCustomOpDef.dt_string,
        "dilations": PyCustomOpDef.dt_string,
        "group": PyCustomOpDef.dt_int64,
        "kernel_shape": PyCustomOpDef.dt_string,
        "rescale": PyCustomOpDef.dt_int64,
    },
)
def int64_conv(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None = None,
    scaling_factor: np.ndarray | None = None,
    auto_pad: str | None = None,
    dilations: str | None = None,
    group: int | None = None,
    kernel_shape: str | None = None,
    pads: str | None = None,
    strides: str | None = None,
    rescale: int | None = None,
) -> np.ndarray:
    """
    Performs a convolution on int64 input tensors.

    This function is registered as a custom ONNX operator via onnxruntime_extensions
    and is used in the JSTprove quantized inference pipeline. It parses ONNX-style
    convolution attributes, applies convolution
    and optionally rescales the result.

    Parameters
    ----------
    X : Input tensor with dtype int64.
    W : Convolution weight tensor with dtype int64.
    B : Optional bias tensor with dtype int64.
    scaling_factor : Scaling factor for rescaling the output.
    auto_pad : Optional ONNX auto padding type (`SAME_UPPER`, `SAME_LOWER`, `VALID`).
    dilations : Dilation values for the convolution (default: `[1, 1]`).
    group : Group value for the convolution (default: 1).
    kernel_shape : Kernel shape (default: `[3, 3]`).
    pads : Padding values (default: `[0, 0, 0, 0]`).
    strides : Stride values (default: `[1, 1]`).
    rescale : Optional flag to apply output rescaling or not.

    Returns
    -------
    numpy.ndarray
        Convolved tensor with dtype int64.

    Notes
    -----
    - This op is part of the `ai.onnx.contrib` custom domain.
    - ONNX Runtime Extensions is required to register this op.

    References
    ----------
    For more information on the convolution operation, please refer to the
    ONNX standard Conv operator documentation:
    https://onnx.ai/onnx/operators/onnx__Conv.html
    """
    _ = auto_pad
    try:
        strides = parse_attr(strides, [1, 1])
        dilations = parse_attr(dilations, [1, 1])
        pads = parse_attr(pads, [0, 0, 0, 0])
        kernel_shape = parse_attr(kernel_shape, [3, 3])

        x = torch.from_numpy(x)
        w = torch.from_numpy(w)
        b = torch.from_numpy(b)

        result = (
            f.conv2d(
                x,
                w,
                bias=b,
                stride=strides,
                padding=pads[:2],
                dilation=dilations,
                groups=group,
            )
            .numpy()
            .astype(np.int64)
        )
        result = rescaling(scaling_factor, rescale, result)
        return result.astype(np.int64)

    except Exception as e:
        msg = f"Int64Conv failed: {e}"
        raise RuntimeError(msg) from e
