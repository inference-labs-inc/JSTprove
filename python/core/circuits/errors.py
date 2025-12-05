from __future__ import annotations

from python.core.utils.helper_functions import RunType


class CircuitError(Exception):
    """
    Base class for all circuit-related errors.

    Attributes:
        message (str): Human-readable description of the error.
        details (dict): Optional structured details for debugging or logging.
    """

    def __init__(
        self: CircuitError,
        message: str,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self: CircuitError) -> str:
        parts = [self.message]
        if self.details:
            parts.append(f"Details: {self.details}")

        # Show the chained exception cause if present
        if self.__cause__ is not None:
            parts.append(f"Caused by: {self.__cause__}")
        return " | ".join(parts)


class CircuitConfigurationError(CircuitError):
    """
    Raised when circuit is not properly configured (missing or invalid attributes).

    Attributes:
        missing_attributes (list): List of missing attributes (if known).
    """

    def __init__(
        self: CircuitConfigurationError,
        message: str | None = None,
        missing_attributes: list | None = None,
        details: dict | None = None,
    ) -> None:
        if missing_attributes and not message:
            message = (
                "Circuit class (python) is misconfigured."
                f" Missing required attributes: {', '.join(missing_attributes)}"
            )
        elif not message:
            message = "Circuit is misconfigured."
        super().__init__(message, details)
        self.missing_attributes = missing_attributes or []


class CircuitInputError(CircuitError):
    """
    Raised when input validation fails (missing or invalid values).

    Attributes:
        parameter (str): Name of the problematic parameter (if known).
        expected (str): Expected type or value description (optional).
        actual (any): Actual value encountered (optional).
    """

    def __init__(
        self: CircuitInputError,
        message: str | None = None,
        parameter: str | None = None,
        expected: str | None = None,
        actual: any | None = None,
        details: dict | None = None,
    ) -> None:
        if parameter and not message:
            msg_parts = [f"Issue with parameter '{parameter}'."]
            if expected:
                msg_parts.append(f"Expected: {expected}.")
            if actual is not None:
                msg_parts.append(f"Got: {actual!r}.")
            message = " ".join(msg_parts)
        elif not message:
            message = "Invalid circuit class (python) input."
        super().__init__(message, details)
        self.parameter = parameter
        self.expected = expected
        self.actual = actual


class CircuitRunError(CircuitError):
    """
    Raised when an operation (compile, prove, verify, etc.) fails.

    Attributes:
        operation (str): Name of the operation that failed (if known).
    """

    def __init__(
        self: CircuitRunError,
        message: str | None = None,
        operation: RunType | None = None,
        details: dict | None = None,
    ) -> None:
        operations_to_name = {
            RunType.COMPILE_CIRCUIT: "Compile",
            RunType.GEN_VERIFY: "Verify",
            RunType.PROVE_WITNESS: "Prove",
            RunType.GEN_WITNESS: "Witness",
        }
        if operation and not message:
            message = f"Circuit operation '{operations_to_name.get(operation)}' failed."
        elif not message:
            message = "Circuit run failed."
        super().__init__(message, details)
        self.operation = operation


class CircuitFileError(CircuitError):
    """
    Raised when file-related operations fail
    (e.g., reading, writing, or accessing files).

    Attributes:
        file_path (str): Path to the problematic file (if known).
    """

    def __init__(
        self: CircuitFileError,
        message: str | None = None,
        file_path: str | None = None,
        details: dict | None = None,
    ) -> None:
        if file_path and not message:
            message = f"File operation failed for path: {file_path}"
        elif not message:
            message = "Circuit file operation failed."
        super().__init__(message, details)
        self.file_path = file_path


class CircuitProcessingError(CircuitError):
    """
    Raised when data processing operations fail
    (e.g., tensor operations, scaling, reshaping).

    Attributes:
        operation (str): Name of the operation that failed (if known).
        data_type (str): Type of data being processed (if known).
    """

    def __init__(
        self: CircuitProcessingError,
        message: str | None = None,
        operation: str | None = None,
        data_type: str | None = None,
        details: dict | None = None,
    ) -> None:
        if operation and not message:
            message = f"Data processing failed during {operation}."
        elif not message:
            message = "Circuit data processing failed."
        super().__init__(message, details)
        self.operation = operation
        self.data_type = data_type


class WitnessMatchError(CircuitError):
    """
    Raised when input validation fails (missing or invalid values).

    Attributes:
        parameter (str): Name of the problematic parameter (if known).
        expected (str): Expected type or value description (optional).
        actual (any): Actual value encountered (optional).
    """

    def __init__(
        self: CircuitInputError,
        message: str | None = None,
    ) -> None:
        common_message = "Witness does not match provided inputs and outputs!"
        if message:
            common_message += f" {message}"
        super().__init__(common_message)
