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
    @BaseCommand.validate_required("circuit_path", "witness_path", "proof_path")
    @BaseCommand.validate_paths("circuit_path", "witness_path")
    @BaseCommand.validate_parent_paths("proof_path")
    def run(cls: type[ProveCommand], args: argparse.Namespace) -> None:
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
