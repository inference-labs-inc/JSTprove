from __future__ import annotations

REPORTING_URL = " https://discord.com/invite/inferencelabs"


class QuantizationError(Exception):
    """
    Base exception class for errors raised during model quantization.
    Can be extended for specific quantization-related errors.
    """

    GENERIC_MESSAGE = (
        "\nThis model is not supported by JSTprove."
        f"\nSubmit model support requests via the JSTprove channel: {REPORTING_URL},"
    )

    def __init__(self: Exception, message: str) -> None:
        """Initialize QuantizationError with a detailed message.

        Args:
            message (str): Specific error message describing the quantization issue.
        """
        full_msg = f"{self.GENERIC_MESSAGE}\n\n{message}"
        super().__init__(full_msg)


class InvalidParamError(QuantizationError):
    """
    Exception raised when invalid parameters or unsupported
    parameters are encountered in a node that is reached during
    quantization the quantization process.
    """

    def __init__(
        self: QuantizationError,
        node_name: str,
        op_type: str,
        message: str,
        attr_key: str | None = None,
        expected: str | None = None,
    ) -> None:
        """Initialize InvalidParamError with context about the invalid parameter.

        Args:
            node_name (str): The name of the node where the error occurred.
            op_type (str): The type of operation of the node.
            message (str): Description of the invalid parameter error.
            attr_key (str, optional): The attribute key that caused the error.
                Defaults to None.
            expected (str, optional): The expected value or format for the attribute.
                Defaults to None.
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
    Exception to be raised when an unsupported operation type is
    reached during quantization.
    """

    def __init__(
        self: QuantizationError,
        op_type: str,
        node_name: str | None = None,
    ) -> None:
        """Initialize UnsupportedOpError with details about the unsupported operation.

        Args:
            op_type (str): The type of the unsupported operation.
            node_name (str, optional): The name of the node where the
                unsupported operation was found to help with debugging.
                    Defaults to None.
        """
        error_msg = f"Unsupported op type: '{op_type}'"
        if node_name:
            error_msg += f" in node '{node_name}'"
        error_msg += ". Please check out the documentation for supported layers."
        self.unsupported_ops = op_type
        super().__init__(error_msg)


class MissingHandlerError(QuantizationError):
    """
    Raised when no handler is registered for an operator.
    """

    def __init__(self: QuantizationError, op_type: str) -> None:
        error_msg = f"No quantization handler registered for operator type '{op_type}'."
        super().__init__(error_msg)


class InitializerNotFoundError(QuantizationError):
    """
    Raised when an initializer required by a node is missing from the initializer map.
    """

    def __init__(
        self: QuantizationError,
        node_name: str,
        initializer_name: str,
    ) -> None:
        error_msg = (
            f"Initializer '{initializer_name}' required for node '{node_name}' "
            "was not found in the initializer map."
        )
        super().__init__(error_msg)


class HandlerImplementationError(QuantizationError):
    """
    Raised when a handler does not conform to the expected quantizer interface.
    For example, missing 'quantize' method, wrong return type, etc.
    """

    def __init__(self: QuantizationError, op_type: str, message: str) -> None:
        error_msg = f"Handler implementation error for operator '{op_type}': {message}"
        super().__init__(error_msg)


class InvalidGraphError(QuantizationError):
    """
    Raised when the ONNX graph is malformed or missing critical information.
    """

    def __init__(self: QuantizationError, message: str) -> None:
        error_msg = f"Invalid ONNX graph structure: {message}"
        super().__init__(error_msg)


class InvalidConfigError(QuantizationError):
    """
    Exception raised when the overall quantization configuration is invalid
    or unsupported (e.g., bad scale_base, scale_exponent, or global settings).
    """

    def __init__(
        self: QuantizationError,
        key: str,
        value: str | float | bool | None,  # noqa: FBT001
        expected: str | None = None,
    ) -> None:
        """Initialize InvalidConfigError with context about the bad config.

        Args:
            key (str): The name of the configuration parameter.
            value (Union[str, int, float, bool, None]):
                The invalid value that was provided.
            expected (str, optional): Description of the expected valid range or type.
        """
        error_msg = f"Invalid configuration for '{key}': got {value}"
        if expected:
            error_msg += f" (expected {expected})"
        super().__init__(error_msg)
