from typing import Any
import numpy as np
from onnxruntime_extensions import onnx_op, PyCustomOpDef
from onnx.reference.ops.op_gemm import Gemm_7
import torch

@onnx_op(
    op_type="Int64Gemm",
    domain="ai.onnx.contrib",
    inputs=[
        PyCustomOpDef.dt_int64,  # X
        PyCustomOpDef.dt_int64,  # W
        PyCustomOpDef.dt_int64   # B
    ],
    outputs=[PyCustomOpDef.dt_int64],
    attrs={
        "alpha": PyCustomOpDef.dt_float,
        "beta": PyCustomOpDef.dt_float,
        "transA": PyCustomOpDef.dt_int64,
        "transB": PyCustomOpDef.dt_int64,
    }
)
def int64_gemm7(a: Any, b: Any, c: Any | None = None, alpha: Any | None = None, beta: Any | None = None, transA: Any | None = None, transB: Any | None = None):
    
    alpha = int(alpha)
    beta = int(beta)

    a = a.T if transA else a
    b = b.T if transB else b

    result = alpha * (a @ b)

    if c is not None:
        result += beta * c

    # result = np.zeros([a.shape[0],b.shape[1]])

    return result.astype(np.int64)

@onnx_op(
    op_type="Int64GemmFull",
    domain="ai.onnx.contrib",
    inputs=[
        PyCustomOpDef.dt_float,  # a
        PyCustomOpDef.dt_float,  # b
        PyCustomOpDef.dt_float   # c
    ],
    outputs=[PyCustomOpDef.dt_float],
    attrs={
        "scale_base": PyCustomOpDef.dt_int64,
        "scaling": PyCustomOpDef.dt_int64,
        "alpha": PyCustomOpDef.dt_float,
        "beta": PyCustomOpDef.dt_float,
        "transA": PyCustomOpDef.dt_int64,
        "transB": PyCustomOpDef.dt_int64,
    }
)
def int64_gemm7_full(a: Any, b: Any, c: Any | None = None, scale_base: Any | None = None, scaling: Any | None = None, alpha: Any | None = None, beta: Any | None = None, transA: Any | None = None, transB: Any | None = None):
    scale = scale_base**scaling

    scale_squared = scale**2


    alpha = int(alpha)
    beta = int(beta)

    a = a.T if transA else a
    b = b.T if transB else b

    result = alpha * (a @ b)

    if c is not None:
        result += beta * c

    # result = np.zeros([a.shape[0],b.shape[1]])

    return result.astype(np.int64)
    
    # return Conv()._run(X, W,B, auto_pad, dilations, group, kernel_shape, pads, strides).asarray().astype(np.int64)
    X = torch.mul(torch.from_numpy(X), scale)  # add batch dim if needed
    W = torch.mul(torch.from_numpy(W), scale) 
    B = torch.mul(torch.from_numpy(B), scale_squared) 

    Y = F.conv2d(X, W, bias=B, stride=strides, padding=pads[:2], dilation=dilations, groups=group).long()

    out = torch.div(Y, scale_squared)

    return Y.numpy().astype(np.float32)