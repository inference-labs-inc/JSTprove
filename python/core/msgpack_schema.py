"""MessagePack schema definitions matching Rust jstprove_circuits::runner::schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import msgpack


@dataclass
class CompiledCircuit:
    circuit: bytes
    witness_solver: bytes
    metadata: dict[str, Any] | None = None

    def pack(self) -> bytes:
        return msgpack.packb(
            {
                "circuit": self.circuit,
                "witness_solver": self.witness_solver,
                "metadata": self.metadata,
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, data: bytes) -> CompiledCircuit:
        d = msgpack.unpackb(data, raw=False)
        return cls(
            circuit=d["circuit"],
            witness_solver=d["witness_solver"],
            metadata=d.get("metadata"),
        )


@dataclass
class WitnessRequest:
    circuit: bytes
    witness_solver: bytes
    inputs: bytes
    outputs: bytes

    def pack(self) -> bytes:
        return msgpack.packb(
            {
                "circuit": self.circuit,
                "witness_solver": self.witness_solver,
                "inputs": self.inputs,
                "outputs": self.outputs,
            },
            use_bin_type=True,
        )


@dataclass
class WitnessBundle:
    witness: bytes
    output_data: list[int] | None = None

    @classmethod
    def unpack(cls, data: bytes) -> WitnessBundle:
        d = msgpack.unpackb(data, raw=False)
        return cls(
            witness=d["witness"],
            output_data=d.get("output_data"),
        )


@dataclass
class ProveRequest:
    circuit: bytes
    witness: bytes

    def pack(self) -> bytes:
        return msgpack.packb(
            {
                "circuit": self.circuit,
                "witness": self.witness,
            },
            use_bin_type=True,
        )


@dataclass
class ProofBundle:
    proof: bytes

    @classmethod
    def unpack(cls, data: bytes) -> ProofBundle:
        d = msgpack.unpackb(data, raw=False)
        return cls(proof=d["proof"])


@dataclass
class VerifyRequest:
    circuit: bytes
    witness: bytes
    proof: bytes
    inputs: bytes
    outputs: bytes

    def pack(self) -> bytes:
        return msgpack.packb(
            {
                "circuit": self.circuit,
                "witness": self.witness,
                "proof": self.proof,
                "inputs": self.inputs,
                "outputs": self.outputs,
            },
            use_bin_type=True,
        )


@dataclass
class VerifyResponse:
    valid: bool
    error: str | None = None

    @classmethod
    def unpack(cls, data: bytes) -> VerifyResponse:
        d = msgpack.unpackb(data, raw=False)
        return cls(
            valid=d["valid"],
            error=d.get("error"),
        )
