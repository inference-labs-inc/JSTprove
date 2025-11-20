from onnx import ModelProto


# Helper to extract input shapes
def get_input_shapes(model: ModelProto) -> dict:
    input_shapes = {}
    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            elif dim.dim_param:
                shape.append(1)  # Default for dynamic dims
            else:
                shape.append(1)
        input_shapes[inp.name] = shape
    return input_shapes
