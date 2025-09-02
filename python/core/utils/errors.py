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

    def __init__(  # noqa: PLR0913
        self: ProofBackendError,
        message: str,
        command: list[str] | None = None,
        returncode: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> None:
        parts = [message]
        if command is not None:
            parts.append(f"Command: {' '.join(command)}")
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
