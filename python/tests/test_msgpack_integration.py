"""Integration tests for msgpack pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgpack

if TYPE_CHECKING:
    from pathlib import Path

from python.core.msgpack_ops import (
    load_circuit_bundle,
    load_circuit_msgpack,
)
from python.core.msgpack_schema import (
    CompiledCircuit,
    ProofBundle,
    VerifyResponse,
    WitnessBundle,
)


class TestMsgpackSchema:
    """Test msgpack schema serialization/deserialization."""

    def test_compiled_circuit_roundtrip(self) -> None:
        original = CompiledCircuit(
            circuit=b"circuit_data_here",
            witness_solver=b"witness_solver_data",
            metadata={"scale": 1.0},
        )
        packed = original.pack()
        unpacked = CompiledCircuit.unpack(packed)

        assert unpacked.circuit == original.circuit
        assert unpacked.witness_solver == original.witness_solver
        assert unpacked.metadata == original.metadata

    def test_witness_bundle_roundtrip(self) -> None:
        original = WitnessBundle(
            witness=b"witness_bytes",
            output_data=[1, 2, 3],
        )
        packed = msgpack.packb(
            {"witness": original.witness, "output_data": original.output_data},
            use_bin_type=True,
        )
        unpacked = WitnessBundle.unpack(packed)

        assert unpacked.witness == original.witness
        assert unpacked.output_data == original.output_data

    def test_proof_bundle_roundtrip(self) -> None:
        packed = msgpack.packb({"proof": b"proof_data"}, use_bin_type=True)
        unpacked = ProofBundle.unpack(packed)
        assert unpacked.proof == b"proof_data"

    def test_verify_response_roundtrip(self) -> None:
        packed = msgpack.packb({"valid": True, "error": None}, use_bin_type=True)
        unpacked = VerifyResponse.unpack(packed)
        assert unpacked.valid is True
        assert unpacked.error is None

        packed_err = msgpack.packb(
            {"valid": False, "error": "bad proof"},
            use_bin_type=True,
        )
        unpacked_err = VerifyResponse.unpack(packed_err)
        assert unpacked_err.valid is False
        assert unpacked_err.error == "bad proof"


class TestLoadCircuitBundle:
    """Test circuit loading functions."""

    def test_load_msgpack_format(self, tmp_path: Path) -> None:
        circuit_bytes = b"test_circuit"
        ws_bytes = b"test_witness_solver"

        bundle = CompiledCircuit(
            circuit=circuit_bytes,
            witness_solver=ws_bytes,
            metadata=None,
        )

        msgpack_file = tmp_path / "test.msgpack"
        with msgpack_file.open("wb") as f:
            f.write(bundle.pack())

        loaded_circuit, loaded_ws = load_circuit_msgpack(msgpack_file)
        assert loaded_circuit == circuit_bytes
        assert loaded_ws == ws_bytes

    def test_load_bundle_prefers_msgpack(self, tmp_path: Path) -> None:
        circuit_bytes = b"msgpack_circuit"
        ws_bytes = b"msgpack_ws"

        bundle = CompiledCircuit(circuit=circuit_bytes, witness_solver=ws_bytes)
        msgpack_file = tmp_path / "test.msgpack"
        with msgpack_file.open("wb") as f:
            f.write(bundle.pack())

        with (tmp_path / "test.txt").open("wb") as f:
            f.write(b"legacy_circuit")
        with (tmp_path / "test_witness_solver.txt").open("wb") as f:
            f.write(b"legacy_ws")

        loaded_circuit, loaded_ws = load_circuit_bundle(tmp_path / "test.txt")
        assert loaded_circuit == circuit_bytes
        assert loaded_ws == ws_bytes

    def test_load_legacy_format(self, tmp_path: Path) -> None:
        with (tmp_path / "circuit.txt").open("wb") as f:
            f.write(b"legacy_circuit")
        with (tmp_path / "circuit_witness_solver.txt").open("wb") as f:
            f.write(b"legacy_ws")

        loaded_circuit, loaded_ws = load_circuit_bundle(tmp_path / "circuit.txt")
        assert loaded_circuit == b"legacy_circuit"
        assert loaded_ws == b"legacy_ws"
