"""High-level msgpack operations for zero-file-IO witness/prove/verify."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from python.core.msgpack_schema import (
    CompiledCircuit,
    ProofBundle,
    ProveRequest,
    VerifyRequest,
    VerifyResponse,
    WitnessBundle,
    WitnessRequest,
)
from python.core.utils.helper_functions import run_msgpack_command

logger = logging.getLogger(__name__)


def load_circuit_bundle(circuit_path: str | Path) -> tuple[bytes, bytes]:
    """Load circuit and witness_solver bytes from compiled circuit files.

    Supports both formats:
    - Msgpack bundle: single .msgpack file with circuit+witness_solver
    - Legacy format: separate .txt and _witness_solver.txt files

    Args:
        circuit_path: Path to circuit file (.msgpack or .txt).

    Returns:
        Tuple of (circuit_bytes, witness_solver_bytes).
    """
    circuit_path = Path(circuit_path)

    if circuit_path.suffix == ".msgpack" or (
        circuit_path.with_suffix(".msgpack").exists()
    ):
        msgpack_path = (
            circuit_path
            if circuit_path.suffix == ".msgpack"
            else circuit_path.with_suffix(".msgpack")
        )
        return load_circuit_msgpack(msgpack_path)

    stem = circuit_path.stem
    parent = circuit_path.parent

    circuit_file = parent / f"{stem}.txt"
    ws_file = parent / f"{stem}_witness_solver.txt"

    with circuit_file.open("rb") as f:
        circuit_bytes = f.read()
    with ws_file.open("rb") as f:
        ws_bytes = f.read()

    return circuit_bytes, ws_bytes


def load_circuit_msgpack(path: str | Path) -> tuple[bytes, bytes]:
    """Load circuit bundle from msgpack file.

    Args:
        path: Path to .msgpack circuit file.

    Returns:
        Tuple of (circuit_bytes, witness_solver_bytes).
    """
    with Path(path).open("rb") as f:
        data = f.read()
    bundle = CompiledCircuit.unpack(data)
    return bundle.circuit, bundle.witness_solver


def save_circuit_msgpack(
    path: str | Path,
    circuit_bytes: bytes,
    witness_solver_bytes: bytes,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save circuit bundle to msgpack file.

    Args:
        path: Output path for .msgpack file.
        circuit_bytes: Raw circuit bytes.
        witness_solver_bytes: Raw witness solver bytes.
        metadata: Optional metadata dict.
    """
    bundle = CompiledCircuit(
        circuit=circuit_bytes,
        witness_solver=witness_solver_bytes,
        metadata=metadata,
    )
    with Path(path).open("wb") as f:
        f.write(bundle.pack())


def msgpack_witness(
    binary_name: str,
    circuit_bytes: bytes,
    witness_solver_bytes: bytes,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    metadata_path: str | Path | None = None,
) -> WitnessBundle:
    """Generate witness via msgpack stdin/stdout (zero file I/O).

    Args:
        binary_name: Name of the circuit binary.
        circuit_bytes: Raw circuit bytes.
        witness_solver_bytes: Raw witness solver bytes.
        inputs: Circuit inputs dict.
        outputs: Circuit outputs dict.
        metadata_path: Optional path to circuit metadata JSON.

    Returns:
        WitnessBundle with witness bytes.
    """
    inputs_json = json.dumps(inputs).encode("utf-8")
    outputs_json = json.dumps(outputs).encode("utf-8")

    req = WitnessRequest(
        circuit=circuit_bytes,
        witness_solver=witness_solver_bytes,
        inputs=inputs_json,
        outputs=outputs_json,
    )

    args = {"m": str(metadata_path)} if metadata_path else None

    result = run_msgpack_command(
        binary_name=binary_name,
        command_type="msgpack_witness_stdin",
        payload=req.pack(),
        args=args,
    )

    return WitnessBundle.unpack(result)


def msgpack_prove(
    binary_name: str,
    circuit_bytes: bytes,
    witness_bytes: bytes,
) -> ProofBundle:
    """Generate proof via msgpack stdin/stdout (zero file I/O).

    Args:
        binary_name: Name of the circuit binary.
        circuit_bytes: Raw circuit bytes.
        witness_bytes: Raw witness bytes from msgpack_witness.

    Returns:
        ProofBundle with proof bytes.
    """
    req = ProveRequest(
        circuit=circuit_bytes,
        witness=witness_bytes,
    )

    result = run_msgpack_command(
        binary_name=binary_name,
        command_type="msgpack_prove_stdin",
        payload=req.pack(),
    )

    return ProofBundle.unpack(result)


def msgpack_verify(
    binary_name: str,
    circuit_bytes: bytes,
    witness_bytes: bytes,
    proof_bytes: bytes,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
) -> VerifyResponse:
    """Verify proof via msgpack stdin/stdout (zero file I/O).

    Args:
        binary_name: Name of the circuit binary.
        circuit_bytes: Raw circuit bytes.
        witness_bytes: Raw witness bytes.
        proof_bytes: Raw proof bytes.
        inputs: Circuit inputs dict.
        outputs: Circuit outputs dict.

    Returns:
        VerifyResponse with valid bool and optional error.
    """
    inputs_json = json.dumps(inputs).encode("utf-8")
    outputs_json = json.dumps(outputs).encode("utf-8")

    req = VerifyRequest(
        circuit=circuit_bytes,
        witness=witness_bytes,
        proof=proof_bytes,
        inputs=inputs_json,
        outputs=outputs_json,
    )

    result = run_msgpack_command(
        binary_name=binary_name,
        command_type="msgpack_verify_stdin",
        payload=req.pack(),
    )

    return VerifyResponse.unpack(result)


def msgpack_witness_prove(
    binary_name: str,
    circuit_bytes: bytes,
    witness_solver_bytes: bytes,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    metadata_path: str | Path | None = None,
) -> tuple[WitnessBundle, ProofBundle]:
    """Generate witness and proof in sequence (zero file I/O).

    Args:
        binary_name: Name of the circuit binary.
        circuit_bytes: Raw circuit bytes.
        witness_solver_bytes: Raw witness solver bytes.
        inputs: Circuit inputs dict.
        outputs: Circuit outputs dict.
        metadata_path: Optional path to circuit metadata JSON.

    Returns:
        Tuple of (WitnessBundle, ProofBundle).
    """
    witness_bundle = msgpack_witness(
        binary_name=binary_name,
        circuit_bytes=circuit_bytes,
        witness_solver_bytes=witness_solver_bytes,
        inputs=inputs,
        outputs=outputs,
        metadata_path=metadata_path,
    )

    proof_bundle = msgpack_prove(
        binary_name=binary_name,
        circuit_bytes=circuit_bytes,
        witness_bytes=witness_bundle.witness,
    )

    return witness_bundle, proof_bundle


class MsgpackCircuitRunner:
    """Stateful runner for msgpack-based circuit operations.

    Caches circuit bytes to avoid repeated file reads.
    """

    def __init__(
        self,
        binary_name: str,
        circuit_path: str | Path,
        metadata_path: str | Path | None = None,
    ) -> None:
        self.binary_name = binary_name
        self.circuit_path = Path(circuit_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self._circuit_bytes: bytes | None = None
        self._ws_bytes: bytes | None = None

        if self.metadata_path is None:
            stem = self.circuit_path.stem
            parent = self.circuit_path.parent
            default_meta = parent / f"{stem}_metadata.json"
            if default_meta.exists():
                self.metadata_path = default_meta

    def _load_circuit(self) -> tuple[bytes, bytes]:
        if self._circuit_bytes is None or self._ws_bytes is None:
            self._circuit_bytes, self._ws_bytes = load_circuit_bundle(self.circuit_path)
        return self._circuit_bytes, self._ws_bytes

    def witness(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> WitnessBundle:
        circuit, ws = self._load_circuit()
        return msgpack_witness(
            binary_name=self.binary_name,
            circuit_bytes=circuit,
            witness_solver_bytes=ws,
            inputs=inputs,
            outputs=outputs,
            metadata_path=self.metadata_path,
        )

    def prove(self, witness_bytes: bytes) -> ProofBundle:
        circuit, _ = self._load_circuit()
        return msgpack_prove(
            binary_name=self.binary_name,
            circuit_bytes=circuit,
            witness_bytes=witness_bytes,
        )

    def verify(
        self,
        witness_bytes: bytes,
        proof_bytes: bytes,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> VerifyResponse:
        circuit, _ = self._load_circuit()
        return msgpack_verify(
            binary_name=self.binary_name,
            circuit_bytes=circuit,
            witness_bytes=witness_bytes,
            proof_bytes=proof_bytes,
            inputs=inputs,
            outputs=outputs,
        )

    def witness_and_prove(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> tuple[WitnessBundle, ProofBundle]:
        circuit, ws = self._load_circuit()
        return msgpack_witness_prove(
            binary_name=self.binary_name,
            circuit_bytes=circuit,
            witness_solver_bytes=ws,
            inputs=inputs,
            outputs=outputs,
            metadata_path=self.metadata_path,
        )
