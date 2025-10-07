from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import argparse

from python.core.circuits.errors import CircuitRunError
from python.core.utils.helper_functions import CircuitExecutionConfig, RunType
from python.frontend.commands.base import BaseCommand


class ProveCommand(BaseCommand):
    """Generate proof from witness."""

    name: ClassVar[str] = "prove"
    aliases: ClassVar[list[str]] = ["prov"]
    help: ClassVar[str] = "Generate a proof from a circuit and witness."

    @classmethod
    def configure_parser(
        cls: type[ProveCommand],
        parser: argparse.ArgumentParser,
    ) -> None:
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
    def run(cls: type[ProveCommand], args: argparse.Namespace) -> None:
        args.circuit_path = args.circuit_path or args.pos_circuit_path
        args.witness_path = args.witness_path or args.pos_witness_path
        args.proof_path = args.proof_path or args.pos_proof_path

        if not all([args.circuit_path, args.witness_path, args.proof_path]):
            msg = "prove requires circuit_path, witness_path, and proof_path"
            raise ValueError(msg)

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

        print(f"[prove] wrote proof → {args.proof_path}")  # noqa: T201
