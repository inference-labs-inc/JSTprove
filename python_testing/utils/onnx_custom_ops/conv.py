from typing import Any
import numpy as np
from onnxruntime_extensions import onnx_op, PyCustomOpDef
# from onnx.reference.ops.op_conv import Conv
import torch
import torch.nn.functional as F

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
        PyCustomOpDef.dt_int64   # B
    ],
    outputs=[PyCustomOpDef.dt_int64],
    attrs={
        "auto_pad": PyCustomOpDef.dt_string,
        "strides": x,
        "pads": x,
        "dilations": x,
        "group": PyCustomOpDef.dt_int64,
        "kernel_shape": x
    }
)
def int64_conv(
    X: Any,
    W: Any,
    B: Any | None = None,
    auto_pad: Any | None = None,
    dilations: Any | None = None,
    group: Any | None = None,
    kernel_shape: Any | None = None,
    pads: Any | None = None,
    strides: Any | None = None
):
    # N, C, H, W_ = X.shape
    # OC, IC, KH, KW = W.shape
    # output = np.zeros((N, OC, H, W_), dtype=np.int64)

    # for n in range(N):
    #     for oc in range(OC):
    #         for ic in range(IC):
    #             output[n, oc] += correlate2d(X[n, ic], W[oc, ic], mode="same")

    # if B is not None:
    #     output += B.reshape(1, -1, 1, 1)

    # # N, C, H, W_ = X.shape
    # # OC, IC, KH, KW = W.shape
    # # output = np.zeros((N, OC, H, W_), dtype=np.int64)

    # return output.astype(np.int64)
    
    strides = parse_attr(strides, [1, 1])
    dilations = parse_attr(dilations, [1, 1])
    pads = parse_attr(pads, [0, 0, 0, 0])
    kernel_shape = parse_attr(kernel_shape, [3, 3])
    # out = F.conv2d(
    #         X,
    #         W,
    #         B,  # do bias separately to handle shift
    #         strides,
    #         padding = pads,
    #         dilation=dilations,
    #         groups = group
    #     )
    # return np.asarray(out).astype(dtype=np.int64)

    
    # return Conv()._run(X, W,B, auto_pad, dilations, group, kernel_shape, pads, strides).asarray().astype(np.int64)
    X = torch.from_numpy(X)  # add batch dim if needed
    W = torch.from_numpy(W)
    B = torch.from_numpy(B)

    Y = F.conv2d(X, W, bias=B, stride=strides, padding=pads[:2], dilation=dilations, groups=group)
    return Y.numpy().astype(np.int64)


    # strides = kwargs.get("strides", [1, 1])
    # pads = kwargs.get("pads", [0, 0, 0, 0])
    # dilations = kwargs.get("dilations", [1, 1])
    # group = kwargs.get("group", 1)

    # Input shape: (N, C_in, H_in, W_in)
    # Weights shape: (C_out, C_in/group, kH, kW)
    N, C_in, H_in, W_in = X.shape
    C_out, C_in_per_group, kH, kW = W.shape


    assert C_in == C_in_per_group * group, "Input channels must match weights channels * groups"

    # Pad input
    pad_top, pad_left, pad_bottom, pad_right = pads
    padded = np.pad(
        X,
        ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0,
    )

    # Calculate output dimensions
    H_out = (H_in + pad_top + pad_bottom - dilations[0] * (kH - 1) - 1) // strides[0] + 1
    W_out = (W_in + pad_left + pad_right - dilations[1] * (kW - 1) - 1) // strides[1] + 1

    # Output tensor
    output = np.zeros((N, C_out, H_out, W_out), dtype=np.int64)

    # Convolution operation
    for n in range(N):
        for g in range(group):
            for cout in range(C_out // group):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * strides[0]
                        w_start = w * strides[1]
                        sum_val = 0
                        for cin in range(C_in_per_group):
                            for kh in range(kH):
                                for kw in range(kW):
                                    h_in = h_start + kh * dilations[0]
                                    w_in = w_start + kw * dilations[1]
                                    in_val = padded[n, g * C_in_per_group + cin, h_in, w_in]
                                    w_val = W[g * (C_out // group) + cout, cin, kh, kw]
                                    sum_val += in_val * w_val
                        output[n, g * (C_out // group) + cout, h, w] = sum_val
    

    return output