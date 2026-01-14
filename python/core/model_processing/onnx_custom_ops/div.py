import numpy as np
from onnxruntime_extensions import PyCustomOpDef, onnx_op


@onnx_op(
    op_type="Int64Div",
    domain="ai.onnx.contrib",
    inputs=[
        PyCustomOpDef.dt_int64,
        PyCustomOpDef.dt_int64,
        PyCustomOpDef.dt_int64,  # Scalar
    ],
    outputs=[PyCustomOpDef.dt_int64],
    attrs={"mode": PyCustomOpDef.dt_string},
)
def int64_div(
    a: np.ndarray,
    b: np.ndarray,
    _scaling_factor: np.ndarray | None = None,
    mode: str | None = None,
) -> np.ndarray:
    """
    Performs a Div (elementwise) operation on int64 input tensors.

    This function is registered as a custom ONNX operator via onnxruntime_extensions
    and is used in the JSTprove quantized inference pipeline.
    It applies Div with the rescaling the outputs back to the original scale.

    Parameters
    ----------
    a : np.ndarray
        First input tensor with dtype int64.
    b : np.ndarray
        Second input tensor with dtype int64.
    scaling_factor : Scaling factor for rescaling the output.
        Optional scalar tensor for rescaling when rescale=1.
        Kept for future use, currently does not do anything.
    mode : str, optional
        Mode of operation. Currently only "constant_pos_int" is supported.

    Returns
    -------
    numpy.ndarray
        Div tensor with dtype int64.

    Notes
    -----
    - This op is part of the `ai.onnx.contrib` custom domain.
    - ONNX Runtime Extensions is required to register this op.

    References
    ----------
    For more information on the Div operation, please refer to the
    ONNX standard Div operator documentation:
    https://onnx.ai/onnx/operators/onnx__Div.html
    """
    try:
        if mode == "constant_pos_int":
            result = a // b
            return result.astype(np.int64)
        msg = f"Unsupported mode: {mode}"
        raise ValueError(msg)  # noqa: TRY301
    except Exception as e:
        msg = f"Int64Div failed: {e}"
        raise RuntimeError(msg) from e
