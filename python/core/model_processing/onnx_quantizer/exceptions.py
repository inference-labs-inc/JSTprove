REPORTING_URL = " https://discord.com/invite/inferencelabs"

class QuantizationError(Exception):
    """
    Base exception class for errors raised during model quantization.
    Can be extended for specific quantization-related errors.
    """
    GENERIC_MESSAGE = (
        "\nThe model submitted is not currently supported by JSTProve."
        f"\nTo enter your model in our support integration queue, please request support for your model in the JSTProve channel of the Inference Labs Discord:{REPORTING_URL}"
    )

    def __init__(self, message: str):
        """Initialize QuantizationError with a detailed message.

        Args:
            message (str): Specific error message describing the quantization issue.
        """        
        full_msg = f"{self.GENERIC_MESSAGE}\n\n{message}"
        super().__init__(full_msg)


class InvalidParamError(QuantizationError):
    """
    Exception raised when invalid parameters or unsupported parameters are encountered in a node that is reached during quantization the quantization process.
    """
    def __init__(
        self,
        node_name: str,
        op_type: str,
        message: str,
        attr_key: str = None,
        expected: str = None,
    ):
        """Initialize InvalidParamError with context about the invalid parameter.

        Args:
            node_name (str): The name of the node where the error occurred.
            op_type (str): The type of operation of the node.
            message (str): Description of the invalid parameter error.
            attr_key (str, optional): The attribute key that caused the error. Defaults to None.
            expected (str, optional): The expected value or format for the attribute. Defaults to None.
        """        
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
    """
    Exception to be raised when an unsupported operation type is reached during quantization.
    """
    def __init__(self, op_type: str, node_name: str = None):
        """Initialize UnsupportedOpError with details about the unsupported operation.

        Args:
            op_type (str): The type of the unsupported operation.
            node_name (str, optional): The name of the node where the unsupported operation was found to help with debugging. Defaults to None.
        """        
        error_msg = f"Unsupported op type: '{op_type}'"
        if node_name:
            error_msg += f" in node '{node_name}'"
        error_msg += ". Please check out the documentation for supported layers."
        super().__init__(error_msg)
