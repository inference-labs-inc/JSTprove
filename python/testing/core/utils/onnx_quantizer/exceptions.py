REPORTING_URL = "..."

class QuantizationError(Exception):
    # TODO fix message
    GENERIC_MESSAGE = (
        "\nThis model is not currently supported by JSTProve. JSTProve is still building up its supported onnx layers."
        f"\nWe can't promise that we can add your layer soon, but to request support for your model,\nPlease message the JSTProve authors at {REPORTING_URL}"
    )

    def __init__(self, message: str):
        full_msg = f"{self.GENERIC_MESSAGE}\n\n{message}"
        super().__init__(full_msg)


class InvalidParamError(QuantizationError):
    def __init__(
        self,
        node_name: str,
        op_type: str,
        message: str,
        attr_key: str = None,
        expected: str = None,
    ):
        self.node_name = node_name
        self.op_type = op_type
        self.message = message
        self.attr_key = attr_key
        self.expected = expected

        error_msg = (
            f"Invalid parameters in node '{node_name}' "
            f"(op_type='{op_type}'): {message}"
        )
        if attr_key:
            error_msg += f" [Attribute: {attr_key}]"
        if expected:
            error_msg += f" [Expected: {expected}]"
        super().__init__(error_msg)

class UnsupportedOpError(QuantizationError):
    def __init__(self, op_type: str, node_name: str = None):
        error_msg = f"Unsupported op type: '{op_type}'"
        if node_name:
            error_msg += f" in node '{node_name}'"
        error_msg += ". Please check out the documentation for supported layers."
        super().__init__(error_msg)

