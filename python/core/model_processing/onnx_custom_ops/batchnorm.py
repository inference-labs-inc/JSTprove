from __future__ import annotations

import numpy as np
from onnxruntime_extensions import PyCustomOpDef, onnx_op

from .custom_helpers import rescaling


@onnx_op(
    op_type="Int64BatchNorm",
    domain="ai.onnx.contrib",
    inputs=[
        PyCustomOpDef.dt_int64,  # X (int64)
        PyCustomOpDef.dt_int64,  # mul (int64 scaled multiplier)
        PyCustomOpDef.dt_int64,  # add (int64 scaled adder)
        PyCustomOpDef.dt_int64,  # scaling_factor
    ],
    outputs=[PyCustomOpDef.dt_int64],
    attrs={"rescale": PyCustomOpDef.dt_int64},
)
def int64_batchnorm(
    x: np.ndarray,
    mul: np.ndarray,
    add: np.ndarray,
    scaling_factor: np.ndarray | None = None,
    rescale: int | None = None,
) -> np.ndarray:
    """
    Int64 BatchNorm (folded into affine transform).

    Computes:
        Y = X * mul + add
    where mul/add are already scaled to int64.

    Parameters
    ----------
    x : Input int64 tensor
    mul : Per-channel int64 scale multipliers
    add : Per-channel int64 bias terms
    scaling_factor: factor to rescale
    rescale : Optional flag to apply post-scaling

    Returns
    -------
    numpy.ndarray (int64)
    """
    try:
        # Broadcasting shapes must match batchnorm layout: NCHW
        # Typically mul/add have shape [C]
        dims_x = len(x.shape)
        dim_ones = (1,) * (dims_x - 2)
        mul = mul.reshape(-1, *dim_ones)
        add = add.reshape(-1, *dim_ones)

        y = x * mul + add

        if rescale is not None:
            y = rescaling(scaling_factor, rescale, y)

        return y.astype(np.int64)

    except Exception as e:
        msg = f"Int64BatchNorm failed: {e}"
        raise RuntimeError(msg) from e
