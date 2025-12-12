from __future__ import annotations

import onnx
from onnx import TensorProto, helper, load, shape_inference
from onnx.utils import extract_model


def prune_model(
    model_path: str,
    output_names: list[str],
    save_path: str,
) -> None:
    """Extract a sub-model with the same inputs and new outputs."""
    model = load(model_path)

    # Provide model input names and the new desired output names.
    input_names = [i.name for i in model.graph.input]

    extract_model(
        input_path=model_path,
        output_path=save_path,
        input_names=input_names,
        output_names=output_names,
    )

    print(f"Pruned model saved to {save_path}")  # noqa: T201


def cut_model(
    model_path: str,
    output_names: list[str],
    save_path: str,
) -> None:
    """Replace the graph outputs with the tensors named in `output_names`."""
    model = onnx.load(model_path)
    model = shape_inference.infer_shapes(model)

    graph = model.graph

    # Remove all current outputs one by one (cannot use .clear() or assignment).
    while graph.output:
        graph.output.pop()

    # Add new outputs.
    for name in output_names:
        # Look in value_info, input, or output.
        candidates = list(graph.value_info) + list(graph.input) + list(graph.output)
        value_info = next((vi for vi in candidates if vi.name == name), None)
        if value_info is None:
            msg = f"Tensor {name} not found in model graph."
            raise ValueError(msg)

        elem_type = value_info.type.tensor_type.elem_type
        shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
        new_output = helper.make_tensor_value_info(name, elem_type, shape)
        graph.output.append(new_output)

    for output in graph.output:
        print(output)  # noqa: T201
        if output.name == "/conv1/Conv_output_0":
            output.type.tensor_type.elem_type = TensorProto.INT64

    onnx.save(model, save_path)
    print(f"Saved cut model with outputs {output_names} to {save_path}")  # noqa: T201


if __name__ == "__main__":
    prune_model(
        model_path="models_onnx/doom.onnx",
        output_names=["/Relu_3_output_0"],  # replace with your intermediate tensor
        save_path="models_onnx/test_doom_cut.onnx",
    )
