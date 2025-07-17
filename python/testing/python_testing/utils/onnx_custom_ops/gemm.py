from typing import Any
import numpy as np
from onnxruntime_extensions import onnx_op, PyCustomOpDef
from onnx.reference.ops.op_gemm import Gemm_7

from python.testing.python_testing.utils.onnx_custom_ops.custom_helpers import rescaling

@onnx_op(
    op_type="Int64Gemm",
    domain="ai.onnx.contrib",
    inputs=[
        PyCustomOpDef.dt_int64,  # X
        PyCustomOpDef.dt_int64,  # W
        PyCustomOpDef.dt_int64,   # B
        PyCustomOpDef.dt_int64   # Scalar
    ],
    outputs=[PyCustomOpDef.dt_int64],
    attrs={
        "alpha": PyCustomOpDef.dt_float,
        "beta": PyCustomOpDef.dt_float,
        "transA": PyCustomOpDef.dt_int64,
        "transB": PyCustomOpDef.dt_int64,
        "rescale": PyCustomOpDef.dt_int64
    }
)
def int64_gemm7(a: Any, b: Any, c: Any | None = None, scaling_factor: Any | None = None, alpha: Any | None = None, beta: Any | None = None, transA: Any | None = None, transB: Any | None = None, rescale: Any | None = None):
    
    alpha = int(alpha)
    beta = int(beta)

    a = a.T if transA else a
    b = b.T if transB else b

    result = alpha * (a @ b)

    if c is not None:
        result += beta * c

    # result = np.zeros([a.shape[0],b.shape[1]])
    result = rescaling(scaling_factor, rescale, result)
    return result.astype(np.int64)