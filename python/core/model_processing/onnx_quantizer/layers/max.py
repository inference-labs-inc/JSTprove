def quantize(
    self,
    node: onnx.NodeProto,
    graph: onnx.GraphProto,
    scale_config: ScaleConfig,
    initializer_map: dict[str, onnx.TensorProto],
) -> List[onnx.NodeProto]:
    _ = graph
    if not isinstance(node, onnx.NodeProto):
        raise HandlerImplementationError(
            op_type="Max", message="quantize() expected an ONNX NodeProto"
        )

    # --- scales as Constant node outputs (robust for ORT) ---
    S = self.get_scaling(scale_config.base, scale_config.exponent)

    scale_f_name = f"{node.name}_scale_f64"
    scale_i_name = f"{node.name}_scale_i64"

    # scalar (0-D) tensors
    scale_f_tensor = numpy_helper.from_array(np.array(S, dtype=np.float64))
    scale_i_tensor = numpy_helper.from_array(np.array(S, dtype=np.int64))

    const_f = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[scale_f_name],
        name=f"{node.name}_const_scale_f64",
        value=scale_f_tensor,
    )
    const_i = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[scale_i_name],
        name=f"{node.name}_const_scale_i64",
        value=scale_i_tensor,
    )

    new_nodes: List[onnx.NodeProto] = [const_f, const_i]
    casted_inputs: List[str] = []

    # --- per input: pick scale dtype, Mul -> Cast(INT64) ---
    for idx, inp in enumerate(node.input):
        use_int_scale = False
        init_t = initializer_map.get(inp)
        if init_t is not None:
            use_int_scale = init_t.data_type == onnx.TensorProto.INT64

        chosen_scale = scale_i_name if use_int_scale else scale_f_name

        mul_out = f"{node.name}_in{idx}_scaled"
        cast_out = f"{mul_out}_i64"

        mul_node = helper.make_node(
            "Mul",
            inputs=[inp, chosen_scale],
            outputs=[mul_out],
            name=f"{node.name}_scale_in{idx}",
        )
        cast_node = helper.make_node(
            "Cast",
            inputs=[mul_out],
            outputs=[cast_out],
            to=onnx.TensorProto.INT64,
            name=f"{node.name}_cast_in{idx}",
        )

        new_nodes.extend([mul_node, cast_node])
        casted_inputs.append(cast_out)

    # --- Max over casted INT64 inputs ---
    max_node = helper.make_node(
        "Max",
        inputs=casted_inputs,
        outputs=list(node.output),
        name=node.name,
    )
    new_nodes.append(max_node)

    return new_nodes
