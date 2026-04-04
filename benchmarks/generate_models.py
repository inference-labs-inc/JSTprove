#!/usr/bin/env python3
import os
import struct
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import msgpack

OUT = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUT, exist_ok=True)

OPSET = 17
DIM = 32


def save(name, graph, inputs_shape, extra_opsets=None, skip_check=False):
    opsets = [helper.make_opsetid("", OPSET)]
    if extra_opsets:
        opsets.extend(extra_opsets)
    model = helper.make_model(graph, opset_imports=opsets)
    model.ir_version = 8
    if not skip_check:
        onnx.checker.check_model(model)
    path = os.path.join(OUT, f"{name}.onnx")
    onnx.save(model, path)

    total = 1
    for d in inputs_shape:
        total *= d
    rng = np.random.default_rng(42)
    data = rng.standard_normal(total).tolist()
    input_path = os.path.join(OUT, f"{name}_input.msgpack")
    with open(input_path, "wb") as f:
        f.write(msgpack.packb({"input": data}))

    print(f"  {name}: {path} ({os.path.getsize(path)} bytes), input: {input_path}")


def make_exp():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, DIM])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, DIM])
    node = helper.make_node("Exp", ["X"], ["Y"])
    graph = helper.make_graph([node], "exp_graph", [X], [Y])
    save("exp", graph, [1, DIM])


def make_sigmoid():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, DIM])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, DIM])
    node = helper.make_node("Sigmoid", ["X"], ["Y"])
    graph = helper.make_graph([node], "sigmoid_graph", [X], [Y])
    save("sigmoid", graph, [1, DIM])


def make_gelu():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, DIM])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, DIM])
    node = helper.make_node("Gelu", ["X"], ["Y"], approximate="tanh")
    graph = helper.make_graph([node], "gelu_graph", [X], [Y])
    save("gelu", graph, [1, DIM], skip_check=True)


def make_softmax():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, DIM])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, DIM])
    node = helper.make_node("Softmax", ["X"], ["Y"], axis=-1)
    graph = helper.make_graph([node], "softmax_graph", [X], [Y])
    save("softmax", graph, [1, DIM])


def make_layer_norm():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, DIM])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, DIM])
    gamma = numpy_helper.from_array(
        np.ones(DIM, dtype=np.float32), name="gamma"
    )
    beta = numpy_helper.from_array(
        np.zeros(DIM, dtype=np.float32), name="beta"
    )
    node = helper.make_node(
        "LayerNormalization", ["X", "gamma", "beta"], ["Y"], axis=-1, epsilon=1e-5
    )
    graph = helper.make_graph([node], "layernorm_graph", [X], [Y], initializer=[gamma, beta])
    save("layer_norm", graph, [1, DIM])


def make_resize():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 16, 16])
    roi = numpy_helper.from_array(np.array([], dtype=np.float32), name="roi")
    scales = numpy_helper.from_array(np.array([], dtype=np.float32), name="scales")
    sizes = numpy_helper.from_array(np.array([1, 1, 16, 16], dtype=np.int64), name="sizes")
    node = helper.make_node(
        "Resize", ["X", "roi", "scales", "sizes"], ["Y"],
        mode="linear",
        coordinate_transformation_mode="half_pixel",
    )
    graph = helper.make_graph(
        [node], "resize_graph", [X], [Y], initializer=[roi, scales, sizes]
    )
    save("resize", graph, [1, 1, 8, 8])


def make_gridsample():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 8, 8])
    rng = np.random.default_rng(99)
    grid_data = (rng.random((1, 8, 8, 2)).astype(np.float32) * 2 - 1)
    grid_init = numpy_helper.from_array(grid_data, name="grid")
    node = helper.make_node(
        "GridSample", ["X", "grid"], ["Y"],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=0,
    )
    graph = helper.make_graph(
        [node], "gridsample_graph", [X], [Y], initializer=[grid_init]
    )
    save("gridsample", graph, [1, 1, 8, 8])


def make_sqrt():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, DIM])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, DIM])
    node = helper.make_node("Sqrt", ["X"], ["Y"])
    graph = helper.make_graph([node], "sqrt_graph", [X], [Y])
    save("sqrt", graph, [1, DIM])


def make_tanh():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, DIM])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, DIM])
    node = helper.make_node("Tanh", ["X"], ["Y"])
    graph = helper.make_graph([node], "tanh_graph", [X], [Y])
    save("tanh", graph, [1, DIM])


def make_erf():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, DIM])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, DIM])
    node = helper.make_node("Erf", ["X"], ["Y"])
    graph = helper.make_graph([node], "erf_graph", [X], [Y])
    save("erf", graph, [1, DIM])


def make_pow():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, DIM])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, DIM])
    exp = numpy_helper.from_array(
        np.array([2.0], dtype=np.float32), name="exp"
    )
    node = helper.make_node("Pow", ["X", "exp"], ["Y"])
    graph = helper.make_graph([node], "pow_graph", [X], [Y], initializer=[exp])
    save("pow", graph, [1, DIM])


def make_matmul():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, DIM])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, DIM])
    rng = np.random.default_rng(42)
    W = numpy_helper.from_array(
        rng.standard_normal((DIM, DIM)).astype(np.float32), name="W"
    )
    node = helper.make_node("MatMul", ["X", "W"], ["Y"])
    graph = helper.make_graph([node], "matmul_graph", [X], [Y], initializer=[W])
    save("matmul", graph, [1, DIM])


def make_averagepool():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 4, 4])
    node = helper.make_node(
        "AveragePool", ["X"], ["Y"],
        kernel_shape=[2, 2], strides=[2, 2],
    )
    graph = helper.make_graph([node], "avgpool_graph", [X], [Y])
    save("averagepool", graph, [1, 1, 8, 8])


def make_pad():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 10, 10])
    pads = numpy_helper.from_array(
        np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64), name="pads"
    )
    const_val = numpy_helper.from_array(
        np.array([0.0], dtype=np.float32), name="constant_value"
    )
    node = helper.make_node("Pad", ["X", "pads", "constant_value"], ["Y"], mode="constant")
    graph = helper.make_graph(
        [node], "pad_graph", [X], [Y], initializer=[pads, const_val]
    )
    save("pad", graph, [1, 1, 8, 8])


def make_reducesum():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, DIM])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    node = helper.make_node("ReduceSum", ["X", "axes"], ["Y"], keepdims=1)
    graph = helper.make_graph([node], "reducesum_graph", [X], [Y], initializer=[axes])
    save("reducesum", graph, [1, DIM])


if __name__ == "__main__":
    print("Generating benchmark ONNX models...")
    make_exp()
    make_sigmoid()
    make_gelu()
    make_softmax()
    make_layer_norm()
    make_resize()
    make_gridsample()
    make_sqrt()
    make_tanh()
    make_erf()
    make_pow()
    make_matmul()
    make_averagepool()
    make_pad()
    make_reducesum()
    print("Done.")
