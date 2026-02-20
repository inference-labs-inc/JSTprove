from __future__ import annotations

import numpy as np
from onnxruntime_extensions import PyCustomOpDef, onnx_op

from .custom_helpers import parse_attr, rescaling


def _conv2d_bigint(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
) -> np.ndarray:
    x = np.asarray(x, dtype=object)
    w = np.asarray(w, dtype=object)

    n_batch, c_in, h_in, w_in = x.shape
    c_out, c_in_g, kh, kw = w.shape
    c_out_g = c_out // groups

    ph, pw = padding
    sh, sw = stride
    dh, dw = dilation

    if ph > 0 or pw > 0:
        x_pad = np.zeros((n_batch, c_in, h_in + 2 * ph, w_in + 2 * pw), dtype=object)
        x_pad[:, :, ph : ph + h_in, pw : pw + w_in] = x
        x = x_pad
        h_in += 2 * ph
        w_in += 2 * pw

    h_out = (h_in - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in - dw * (kw - 1) - 1) // sw + 1

    output = np.zeros((n_batch, c_out, h_out, w_out), dtype=object)

    for g in range(groups):
        o_s = g * c_out_g
        i_s = g * c_in_g
        w_g = w[o_s : o_s + c_out_g]

        for kkh in range(kh):
            for kkw in range(kw):
                x_slice = x[
                    :,
                    i_s : i_s + c_in_g,
                    kkh * dh : kkh * dh + h_out * sh : sh,
                    kkw * dw : kkw * dw + w_out * sw : sw,
                ].copy()

                w_kk = w_g[:, :, kkh, kkw]

                for n in range(n_batch):
                    x_mat = x_slice[n].reshape(c_in_g, -1)
                    contrib = np.dot(w_kk, x_mat)
                    output[n, o_s : o_s + c_out_g] += contrib.reshape(
                        c_out_g,
                        h_out,
                        w_out,
                    )

    if b is not None:
        b_obj = np.asarray(b, dtype=object)
        output += b_obj[np.newaxis, :, np.newaxis, np.newaxis]

    return output


@onnx_op(
    op_type="Int64Conv",
    domain="ai.onnx.contrib",
    inputs=[
        PyCustomOpDef.dt_int64,  # X
        PyCustomOpDef.dt_int64,  # W
        PyCustomOpDef.dt_int64,  # B
        PyCustomOpDef.dt_int64,  # scaling factor
    ],
    outputs=[PyCustomOpDef.dt_int64],
    attrs={
        "auto_pad": PyCustomOpDef.dt_string,
        "strides": PyCustomOpDef.dt_string,
        "pads": PyCustomOpDef.dt_string,
        "dilations": PyCustomOpDef.dt_string,
        "group": PyCustomOpDef.dt_int64,
        "kernel_shape": PyCustomOpDef.dt_string,
        "rescale": PyCustomOpDef.dt_int64,
    },
)
def int64_conv(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None = None,
    scaling_factor: np.ndarray | None = None,
    auto_pad: str | None = None,
    dilations: str | None = None,
    group: int | None = None,
    kernel_shape: str | None = None,
    pads: str | None = None,
    strides: str | None = None,
    rescale: int | None = None,
) -> np.ndarray:
    """
    Performs a convolution on int64 input tensors using arbitrary-precision
    Python integers to avoid int64 overflow in intermediate accumulations.

    Parameters
    ----------
    x : Input tensor with dtype int64.
    w : Convolution weight tensor with dtype int64.
    b : Optional bias tensor with dtype int64.
    scaling_factor : Scaling factor for rescaling the output.
    auto_pad : Optional ONNX auto padding type.
    dilations : Dilation values for the convolution (default: `[1, 1]`).
    group : Group value for the convolution (default: 1).
    kernel_shape : Kernel shape (default: `[3, 3]`).
    pads : Padding values (default: `[0, 0, 0, 0]`).
    strides : Stride values (default: `[1, 1]`).
    rescale : Optional flag to apply output rescaling or not.

    Returns
    -------
    numpy.ndarray
        Convolved tensor with dtype int64.
    """
    _ = auto_pad
    try:
        strides = parse_attr(strides, [1, 1])
        dilations = parse_attr(dilations, [1, 1])
        pads = parse_attr(pads, [0, 0, 0, 0])
        kernel_shape = parse_attr(kernel_shape, [3, 3])

        result = _conv2d_bigint(
            x,
            w,
            b,
            stride=strides,
            padding=pads[:2],
            dilation=dilations,
            groups=group,
        )

        if scaling_factor is not None:
            scaling_factor = int(scaling_factor.flat[0])
        result = rescaling(scaling_factor, rescale, result)
        return result.astype(np.int64)

    except Exception as e:
        msg = f"Int64Conv failed: {e}"
        raise RuntimeError(msg) from e
