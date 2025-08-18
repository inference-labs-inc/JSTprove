from typing import Any
import numpy as np
from onnxruntime_extensions import onnx_op, PyCustomOpDef
from onnx.reference.ops.op_gemm import Gemm_7
import torch

@onnx_op(
    op_type="Int64Relu",
    domain="ai.onnx.contrib",
    inputs=[PyCustomOpDef.dt_int64],
    outputs=[PyCustomOpDef.dt_int64],
)
def int64_relu(X):
    return np.maximum(X, 0).astype(np.int64)
