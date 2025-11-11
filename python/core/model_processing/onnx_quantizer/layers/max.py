# python/core/model_processing/onnx_quantizer/layers/max.py


def quantize(
    self,
    node: onnx.NodeProto,
    graph: onnx.GraphProto,
    scale_config: ScaleConfig,
    initializer_map: dict[str, onnx.TensorProto],
) -> list[onnx.NodeProto]:
    """
    Passthrough INT64 version:
      - Insert Cast(INT64) for each input (graph input or initializer-fed)
      - Keep standard ONNX Max
      - No scaling, no new initializers, no custom ops
    """
    _ = graph, scale_config, initializer_map  # unused in passthrough
    new_nodes: list[onnx.NodeProto] = []
    casted_inputs: list[str] = []

    for idx, inp_name in enumerate(node.input):
        cast_out = f"{node.name}_in{idx}_i64"
        new_nodes.append(
            helper.make_node(
                "Cast",
                inputs=[inp_name],
                outputs=[cast_out],
                to=onnx.TensorProto.INT64,
                name=f"{node.name}_cast_in{idx}",
            )
        )
        casted_inputs.append(cast_out)

    # Standard ONNX Max with uniform INT64 inputs
    new_nodes.append(
        helper.make_node(
            "Max",
            inputs=casted_inputs,
            outputs=list(node.output),
            name=node.name,
        )
    )
    return new_nodes
