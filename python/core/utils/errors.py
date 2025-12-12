from __future__ import annotations


class CircuitExecutionError(Exception):
    """Base exception for all circuit execution-related errors."""

    def __init__(self: CircuitExecutionError, message: str) -> None:
        super().__init__(message)
        self.message = message


class MissingFileError(CircuitExecutionError):
    """Raised when cant find file"""

    def __init__(self: MissingFileError, message: str, path: str | None = None) -> None:
        full_message = message if path is None else f"{message} [Path: {path}]"
        super().__init__(full_message)
        self.path = path


class FileCacheError(CircuitExecutionError):
    """Raised when reading or writing cached output fails."""

    def __init__(self: FileCacheError, message: str, path: str | None = None) -> None:
        full_message = message if path is None else f"{message} [Path: {path}]"
        super().__init__(full_message)
        self.path = path


class ProofBackendError(CircuitExecutionError):
    """Raised when a Cargo command fails."""

    def __init__(
        self: ProofBackendError,
        message: str,
        command: list[str] | None = None,
        returncode: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> None:
        parts = [message]
        if command is not None:
            command2 = [str(c) for c in command]
            parts.append(f"Command: {' '.join(command2)}")
            command = command2
        if returncode is not None:
            parts.append(f"Exit code: {returncode}")
        if stdout:
            parts.append(f"STDOUT:\n{stdout}")
        if stderr:
            parts.append(f"STDERR:\n{stderr}")
        full_message = "\n".join(parts)
        super().__init__(full_message)
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class ProofSystemNotImplementedError(CircuitExecutionError):
    """Raised when a proof system is not implemented."""

    def __init__(self: ProofSystemNotImplementedError, proof_system: object) -> None:
        message = f"Proof system '{proof_system}' is not implemented."
        super().__init__(message)
        self.proof_system = proof_system


class CircuitUtilsError(Exception):
    """Base exception for layer utility errors."""


class InputFileError(CircuitUtilsError):
    """Raised when reading an input file fails."""

    def __init__(
        self: InputFileError,
        file_path: str,
        message: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        full_msg = f"Failed to read input file '{file_path}': {message}"
        super().__init__(full_msg)
        self.file_path = file_path
        self.__cause__ = cause


class MissingCircuitAttributeError(CircuitUtilsError):
    """Raised when a required attribute is missing or not set."""

    def __init__(
        self: MissingCircuitAttributeError,
        attribute_name: str,
        context: str | None = None,
    ) -> None:
        msg = f"Required attribute '{attribute_name}' is missing"
        if context:
            msg += f" ({context})"
        super().__init__(msg)
        self.attribute_name = attribute_name


class ShapeMismatchError(CircuitUtilsError):
    """Raised when reshaping tensors fails due to incompatible shapes."""

    def __init__(
        self: ShapeMismatchError,
        expected_shape: list[int],
        actual_shape: list[int],
    ) -> None:
        super().__init__(
            f"Cannot reshape tensor of shape {actual_shape}"
            f" to expected shape {expected_shape}",
        )
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
