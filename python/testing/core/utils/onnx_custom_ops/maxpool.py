from typing import Any
import numpy as np
from onnxruntime_extensions import onnx_op, PyCustomOpDef
import torch
import torch.nn.functional as F


@onnx_op(
    op_type="Int64MaxPool",
    domain="ai.onnx.contrib",
    inputs=[
        PyCustomOpDef.dt_int64  # input tensor
    ],
    outputs=[PyCustomOpDef.dt_int64],
    attrs={
        "strides": PyCustomOpDef.dt_string,
        "pads": PyCustomOpDef.dt_string,
        "kernel_shape": PyCustomOpDef.dt_string,
    }
)
def int64_maxpool(
    X: Any,
    strides: Any | None = None,
    pads: Any | None = None,
    kernel_shape: Any | None = None,
):
    def parse_attr(attr, default):
        if attr is None:
            return default
        return [int(x) for x in attr.split(",")]

    strides = parse_attr(strides, [1, 1])
    pads = parse_attr(pads, [0, 0])
    kernel_size = parse_attr(kernel_shape, [2, 2])

    X = torch.from_numpy(X)
    result = F.max_pool2d(X, kernel_size=kernel_size, stride=strides, padding=pads[:2])
    return result.numpy().astype(np.int64)