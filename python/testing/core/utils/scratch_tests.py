import onnx
from onnx import TensorProto, helper, shape_inference
from onnx import numpy_helper
from onnx import load, save
from onnx.utils import extract_model

def prune_model(model_path, output_names, save_path):
    model = load(model_path)

    # Provide model input names and the new desired output names
    input_names = [i.name for i in model.graph.input]

    extract_model(
        input_path=model_path,
        output_path=save_path,
        input_names=input_names,
        output_names=output_names
    )

    print(f"Pruned model saved to {save_path}")


def cut_model(model_path, output_names, save_path):
    model = onnx.load(model_path)
    model = shape_inference.infer_shapes(model)

    graph = model.graph

    # Remove all current outputs one by one (cannot use .clear() or assignment)
    while len(graph.output) > 0:
        graph.output.pop()

    # Add new outputs
    for name in output_names:
        # Look in value_info, input, or output
        candidates = list(graph.value_info) + list(graph.input) + list(graph.output)
        value_info = next((vi for vi in candidates if vi.name == name), None)
        if value_info is None:
            raise ValueError(f"Tensor {name} not found in model graph.")

        elem_type = value_info.type.tensor_type.elem_type
        shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
        new_output = helper.make_tensor_value_info(name, elem_type, shape)
        graph.output.append(new_output)
    for output in graph.output:
        print(output)
        if output.name == "/conv1/Conv_output_0":
            output.type.tensor_type.elem_type = TensorProto.INT64

    onnx.save(model, save_path)
    print(f"Saved cut model with outputs {output_names} to {save_path}")


if __name__ == "__main__":
    # /conv1/Conv_output_0
    # prune_model(
    #     model_path="models_onnx/doom.onnx",
    #     output_names=["/Relu_2_output_0"],  # replace with your intermediate tensor
    #     save_path= "models_onnx/test_doom_cut.onnx"
    # )
    # cut_model("models_onnx/doom.onnx",["/Relu_2_output_0"], "test_doom_after_conv.onnx")
    prune_model(
        model_path="models_onnx/doom.onnx",
        output_names=["/Relu_3_output_0"],  # replace with your intermediate tensor
        save_path= "models_onnx/test_doom_cut.onnx"
    )
