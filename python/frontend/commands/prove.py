from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, ClassVar

from python.core.circuits.errors import CircuitRunError
from python.core.utils.helper_functions import CircuitExecutionConfig, RunType
from python.frontend.commands.base import BaseCommand

DEFAULT_CIRCUIT_MODULE = "python.core.circuit_models.generic_onnx"
DEFAULT_CIRCUIT_CLASS = "GenericModelONNX"


class ProveCommand(BaseCommand):
    """Generate proof from witness."""

    name: ClassVar[str] = "prove"
    aliases: ClassVar[list[str]] = ["prov"]
    help: ClassVar[str] = "Generate a proof from a circuit and witness."

    @classmethod
    def configure_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "pos_circuit_path",
            nargs="?",
            metavar="circuit_path",
            help="Path to the compiled circuit.",
        )
        parser.add_argument(
            "pos_witness_path",
            nargs="?",
            metavar="witness_path",
            help="Path to an existing witness.",
        )
        parser.add_argument(
            "pos_proof_path",
            nargs="?",
            metavar="proof_path",
            help="Path to write proof.",
        )
        parser.add_argument(
            "-c",
            "--circuit-path",
            help="Path to the compiled circuit.",
        )
        parser.add_argument(
            "-w",
            "--witness-path",
            help="Path to an existing witness.",
        )
        parser.add_argument(
            "-p",
            "--proof-path",
            help="Path to write proof.",
        )

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        args.circuit_path = args.circuit_path or args.pos_circuit_path
        args.witness_path = args.witness_path or args.pos_witness_path
        args.proof_path = args.proof_path or args.pos_proof_path

        if not all([args.circuit_path, args.witness_path, args.proof_path]):
            raise ValueError("prove requires circuit_path, witness_path, and proof_path")

        cls._ensure_file_exists(args.circuit_path)
        cls._ensure_file_exists(args.witness_path)
        cls._ensure_parent_dir(args.proof_path)

        circuit = cls._build_circuit("cli")

        try:
            circuit.base_testing(
                CircuitExecutionConfig(
                    run_type=RunType.PROVE_WITNESS,
                    circuit_path=args.circuit_path,
                    witness_file=args.witness_path,
                    proof_file=args.proof_path,
                    ecc=False,
                ),
            )
        except CircuitRunError as e:
            raise RuntimeError(e) from e

        print(f"[prove] wrote proof → {args.proof_path}")

    @staticmethod
    def _ensure_file_exists(path: str) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
        if not p.exists() or not p.stat().st_mode & 0o444:
            raise PermissionError(f"Cannot read file: {path}")

    @staticmethod
    def _ensure_parent_dir(path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_circuit(model_name_hint: str | None = None) -> Any:
        mod = importlib.import_module(DEFAULT_CIRCUIT_MODULE)
        try:
            cls = getattr(mod, DEFAULT_CIRCUIT_CLASS)
        except AttributeError as e:
            msg = f"Default circuit class '{DEFAULT_CIRCUIT_CLASS}' not found in '{DEFAULT_CIRCUIT_MODULE}'"
            raise RuntimeError(msg) from e

        name = model_name_hint or "cli"

        for attempt in (
            lambda: cls(model_name=name),
            lambda: cls(name=name),
            lambda: cls(name),
            lambda: cls(),
        ):
            try:
                return attempt()
            except TypeError:
                continue

        raise RuntimeError(f"Could not construct {cls.__name__} with/without name '{name}'")
