import numpy as np
from onnxruntime_extensions import PyCustomOpDef, onnx_op

from .custom_helpers import rescaling


@onnx_op(
    op_type="Int64Mul",
    domain="ai.onnx.contrib",
    inputs=[
        PyCustomOpDef.dt_int64,
        PyCustomOpDef.dt_int64,
        PyCustomOpDef.dt_int64,  # Scalar
    ],
    outputs=[PyCustomOpDef.dt_int64],
    attrs={
        "rescale": PyCustomOpDef.dt_int64,
    },
)
def int64_mul(
    a: np.ndarray,
    b: np.ndarray,
    scaling_factor: np.ndarray | None = None,
    rescale: int | None = None,
) -> np.ndarray:
    """
    Performs a Mul (hadamard product) operation on int64 input tensors.

    This function is registered as a custom ONNX operator via onnxruntime_extensions
    and is used in the JSTprove quantized inference pipeline.
    It applies Mul with the rescaling the outputs back to the original scale.

    Parameters
    ----------
    a : np.ndarray
        First input tensor with dtype int64.
    b : np.ndarray
        Second input tensor with dtype int64.
    scaling_factor : Scaling factor for rescaling the output.
        Optional scalar tensor for rescaling when rescale=1.
    rescale : int, optional
        Whether to apply rescaling (0=no, 1=yes).

    Returns
    -------
    numpy.ndarray
        Mul tensor with dtype int64.

    Notes
    -----
    - This op is part of the `ai.onnx.contrib` custom domain.
    - ONNX Runtime Extensions is required to register this op.

    References
    ----------
    For more information on the Mul operation, please refer to the
    ONNX standard Mul operator documentation:
    https://onnx.ai/onnx/operators/onnx__Mul.html
    """
    try:
        result = a * b
        result = rescaling(scaling_factor, rescale, result)
        return result.astype(np.int64)
    except Exception as e:
        msg = f"Int64Mul failed: {e}"
        raise RuntimeError(msg) from e
