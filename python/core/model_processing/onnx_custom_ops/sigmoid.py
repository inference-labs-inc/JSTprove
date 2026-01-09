import numpy as np
from onnxruntime_extensions import PyCustomOpDef, onnx_op


@onnx_op(
    op_type="Int64Sigmoid",
    domain="ai.onnx.contrib",
    inputs=[PyCustomOpDef.dt_int64],
    outputs=[PyCustomOpDef.dt_int64],
    attrs={"scale": PyCustomOpDef.dt_int64},
)
def int64_sigmoid(x: np.ndarray, scale: int) -> np.ndarray:
    """
    Performs a Sigmoid operation on int64 quantized input tensors.

    This function is registered as a custom ONNX operator via onnxruntime_extensions
    and is used in the JSTprove quantized inference pipeline.

    The operation:
    1. Converts scaled int64 to float: x_float = x / scale
    2. Clamps to [-8, 8) to match the Rust circuit implementation
    3. Applies sigmoid: 1 / (1 + exp(-x_float))
    4. Rescales back to int64: result * scale

    Parameters
    ----------
    x : np.ndarray
        Input tensor with dtype int64, representing scaled fixed-point values.
    scale : int
        The quantization scale factor (e.g., 2^18).

    Returns
    -------
    np.ndarray
        Sigmoid output tensor with dtype int64, scaled by the same factor.
    """
    try:
        scale_f = float(scale)
        x_float = x.astype(np.float64) / scale_f
        x_clamped = np.clip(x_float, -8.0, 8.0 - 1.0 / scale_f)
        sigmoid_result = 1.0 / (1.0 + np.exp(-x_clamped))
        return np.floor(sigmoid_result * scale_f).astype(np.int64)
    except Exception as e:
        msg = f"Int64Sigmoid failed: {e}"
        raise RuntimeError(msg) from e
