import numpy as np
from onnxruntime_extensions import PyCustomOpDef, onnx_op


@onnx_op(
    op_type="Int64Softmax",
    domain="ai.onnx.contrib",
    inputs=[PyCustomOpDef.dt_int64],
    outputs=[PyCustomOpDef.dt_int64],
    attrs={"scale": PyCustomOpDef.dt_int64, "axis": PyCustomOpDef.dt_int64},
)
def int64_softmax(x: np.ndarray, scale: int, axis: int) -> np.ndarray:
    """
    Performs a Softmax operation on int64 quantized input tensors.

    This function is registered as a custom ONNX operator via onnxruntime_extensions
    and is used in the JSTprove quantized inference pipeline.

    The operation:
    1. Converts scaled int64 to float: x_float = x / scale
    2. Clamps to [-8, 8) to prevent overflow (matches Rust circuit)
    3. Computes exp for each element
    4. Normalizes by sum along axis
    5. Rescales back to int64

    Parameters
    ----------
    x : np.ndarray
        Input tensor with dtype int64, representing scaled fixed-point values.
    scale : int
        The quantization scale factor (e.g., 2^18).
    axis : int
        The axis along which to compute softmax.

    Returns
    -------
    np.ndarray
        Softmax output tensor with dtype int64, scaled by the same factor.
    """
    try:
        scale_f = float(scale)
        x_float = x.astype(np.float64) / scale_f
        x_clamped = np.clip(x_float, -8.0, 8.0 - 1.0 / scale_f)
        exp_vals = np.exp(x_clamped)
        exp_clamped = np.clip(exp_vals, 0, 2981.0)
        exp_sum = np.sum(exp_clamped, axis=axis, keepdims=True)
        softmax_result = exp_clamped / exp_sum
        return np.floor(softmax_result * scale_f).astype(np.int64)
    except Exception as e:
        msg = f"Int64Softmax failed: {e}"
        raise RuntimeError(msg) from e
