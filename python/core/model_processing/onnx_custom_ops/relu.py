import numpy as np
from onnxruntime_extensions import onnx_op, PyCustomOpDef

@onnx_op(
    op_type="Int64Relu",
    domain="ai.onnx.contrib",
    inputs=[PyCustomOpDef.dt_int64],
    outputs=[PyCustomOpDef.dt_int64],
)
def int64_relu(X):
    """
    Performs a ReLU operation on int64 input tensors.

    This function is registered as a custom ONNX operator via onnxruntime_extensions
    and is used in the JSTProve quantized inference pipeline. It applies ReLU as is (there are no attributes to ReLU).

    Parameters
    ----------
    X : Input tensor with dtype int64.

    Returns
    -------
    numpy.ndarray
        ReLU tensor with dtype int64.

    Notes
    -----
    - This op is part of the `ai.onnx.contrib` custom domain.
    - ONNX Runtime Extensions is required to register this op.

    References
    ----------
    For more information on the ReLU operation, please refer to the
    ONNX standard ReLU operator documentation:
    https://onnx.ai/onnx/operators/onnx__Relu.html
    """
    return np.maximum(X, 0).astype(np.int64)