from __future__ import annotations

import numpy as np
from onnxruntime_extensions import PyCustomOpDef, onnx_op

from .custom_helpers import rescaling


@onnx_op(
    op_type="Int64Gemm",
    domain="ai.onnx.contrib",
    inputs=[
        PyCustomOpDef.dt_int64,  # X
        PyCustomOpDef.dt_int64,  # W
        PyCustomOpDef.dt_int64,  # B
        PyCustomOpDef.dt_int64,  # Scalar
    ],
    outputs=[PyCustomOpDef.dt_int64],
    attrs={
        "alpha": PyCustomOpDef.dt_float,
        "beta": PyCustomOpDef.dt_float,
        "transA": PyCustomOpDef.dt_int64,
        "transB": PyCustomOpDef.dt_int64,
        "rescale": PyCustomOpDef.dt_int64,
    },
)
def int64_gemm7(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    scaling_factor: np.ndarray | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    transA: int | None = None,  # noqa: N803
    transB: int | None = None,  # noqa: N803
    rescale: int | None = None,
) -> np.ndarray:
    """
    Performs a Gemm (alternatively: Linear layer) on int64 input tensors.

    This function is registered as a custom ONNX operator via onnxruntime_extensions
    and is used in the JSTprove quantized inference pipeline. It parses ONNX-style
    gemm attributes, applies gemm
    and optionally rescales the result.

    Parameters
    ----------
    a : Input tensor with dtype int64.
    b : Gemm weight tensor with dtype int64.
    c : Optional bias tensor with dtype int64.
    scaling_factor : Scaling factor for rescaling the output.
    alpha : alpha value for Gemm operation.
    beta : beta value for Gemm operation.
    transA : Transpose the a matrix before the Gemm operation
    transB : Transpose the b matrix before the Gemm operation
    rescale : Optional flag to apply output rescaling or not.

    Returns
    -------
    numpy.ndarray
        Gemm tensor with dtype int64.

    Notes
    -----
    - This op is part of the `ai.onnx.contrib` custom domain.
    - ONNX Runtime Extensions is required to register this op.

    References
    ----------
    For more information on the gemm operation, please refer to the
    ONNX standard Gemm operator documentation:
    https://onnx.ai/onnx/operators/onnx__Gemm.html
    """
    try:
        alpha = int(alpha)
        beta = int(beta)

        a = a.T if transA else a
        b = b.T if transB else b

        result = alpha * (a @ b)

        if c is not None:
            result += beta * c

        result = rescaling(scaling_factor, rescale, result)
        return result.astype(np.int64)

    except Exception as e:
        msg = f"Int64Gemm failed: {e}"
        raise RuntimeError(msg) from e
