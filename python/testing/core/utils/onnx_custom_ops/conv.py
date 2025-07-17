from typing import Any
import numpy as np
from onnxruntime_extensions import onnx_op, PyCustomOpDef
# from onnx.reference.ops.op_conv import Conv
import torch
import torch.nn.functional as F
from python.testing.core.utils.onnx_custom_ops.custom_helpers import rescaling

from scipy.signal import correlate2d

x = PyCustomOpDef.dt_string

def parse_attr(attr, default):
        if attr is None:
            return default
        return [int(x) for x in attr.split(",")]

@onnx_op(
    op_type="Int64Conv",
    domain="ai.onnx.contrib",
    inputs=[
        PyCustomOpDef.dt_int64,  # X
        PyCustomOpDef.dt_int64,  # W
        PyCustomOpDef.dt_int64,   # B
        PyCustomOpDef.dt_int64, # scaling factor
    ],
    outputs=[PyCustomOpDef.dt_int64],
    attrs={
        "auto_pad": PyCustomOpDef.dt_string,
        "strides": PyCustomOpDef.dt_string,
        "pads": PyCustomOpDef.dt_string,
        "dilations": PyCustomOpDef.dt_string,
        "group": PyCustomOpDef.dt_int64,
        "kernel_shape": PyCustomOpDef.dt_string,
        "rescale": PyCustomOpDef.dt_int64
    }
)
def int64_conv(
    X: Any,
    W: Any,
    B: Any | None = None,
    scaling_factor: Any | None = None,
    auto_pad: Any | None = None,
    dilations: Any | None = None,
    group: Any | None = None,
    kernel_shape: Any | None = None,
    pads: Any | None = None,
    strides: Any | None = None,
    rescale: Any | None = None,
):

    
    strides = parse_attr(strides, [1, 1])
    dilations = parse_attr(dilations, [1, 1])
    pads = parse_attr(pads, [0, 0, 0, 0])
    kernel_shape = parse_attr(kernel_shape, [3, 3])

    
    # return Conv()._run(X, W,B, auto_pad, dilations, group, kernel_shape, pads, strides).asarray().astype(np.int64)
    X = torch.from_numpy(X)  # add batch dim if needed
    W = torch.from_numpy(W)
    B = torch.from_numpy(B)

    result = F.conv2d(X, W, bias=B, stride=strides, padding=pads[:2], dilation=dilations, groups=group).numpy().astype(np.int64)
    result = rescaling(scaling_factor, rescale, result)
    return result.astype(np.int64)




# @onnx_op(
#     op_type="Int64ConvFull",
#     domain="ai.onnx.contrib",
#     inputs=[
#         PyCustomOpDef.dt_float,  # X
#         PyCustomOpDef.dt_float,  # W
#         PyCustomOpDef.dt_float   # B
#     ],
#     outputs=[PyCustomOpDef.dt_float],
#     attrs={
#         "scale_base": PyCustomOpDef.dt_int64,
#         "scaling": PyCustomOpDef.dt_int64,
#         "auto_pad": PyCustomOpDef.dt_string,
#         "strides": PyCustomOpDef.dt_string,
#         "pads": PyCustomOpDef.dt_string,
#         "dilations": PyCustomOpDef.dt_string,
#         "group": PyCustomOpDef.dt_int64,
#         "kernel_shape": PyCustomOpDef.dt_string,
#     }
# )
# def int64_conv_full(
#     X: Any,
#     W: Any,
#     B: Any | None = None,
#     scale_base: Any | None = None,
#     scaling: Any | None = None,
#     auto_pad: Any | None = None,
#     dilations: Any | None = None,
#     group: Any | None = None,
#     kernel_shape: Any | None = None,
#     pads: Any | None = None,
#     strides: Any | None = None,
# ):
#     scale = scale_base**scaling

#     scale_squared = scale**2

#     strides = parse_attr(strides, [1, 1])
#     dilations = parse_attr(dilations, [1, 1])
#     pads = parse_attr(pads, [0, 0, 0, 0])
#     kernel_shape = parse_attr(kernel_shape, [3, 3])

    
#     # return Conv()._run(X, W,B, auto_pad, dilations, group, kernel_shape, pads, strides).asarray().astype(np.int64)
#     X = torch.mul(torch.from_numpy(X), scale)  # add batch dim if needed
#     W = torch.mul(torch.from_numpy(W), scale) 
#     B = torch.mul(torch.from_numpy(B), scale_squared) 

#     Y = F.conv2d(X, W, bias=B, stride=strides, padding=pads[:2], dilation=dilations, groups=group).long()

#     out = torch.div(Y, scale_squared)

#     return Y.numpy().astype(np.float32)