from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import argparse

from python.core.circuits.errors import CircuitRunError
from python.core.utils.helper_functions import CircuitExecutionConfig, RunType
from python.frontend.commands.base import BaseCommand


class VerifyCommand(BaseCommand):
    """Verify a proof."""

    name: ClassVar[str] = "verify"
    aliases: ClassVar[list[str]] = ["ver"]
    help: ClassVar[str] = "Verify a proof."

    @classmethod
    def configure_parser(
        cls: type[VerifyCommand],
        parser: argparse.ArgumentParser,
    ) -> None:
        parser.add_argument(
            "pos_circuit_path",
            nargs="?",
            metavar="circuit_path",
            help="Path to the compiled circuit.",
        )
        parser.add_argument(
            "pos_input_path",
            nargs="?",
            metavar="input_path",
            help="Path to input JSON.",
        )
        parser.add_argument(
            "pos_output_path",
            nargs="?",
            metavar="output_path",
            help="Path to expected outputs JSON.",
        )
        parser.add_argument(
            "pos_witness_path",
            nargs="?",
            metavar="witness_path",
            help="Path to witness.",
        )
        parser.add_argument(
            "pos_proof_path",
            nargs="?",
            metavar="proof_path",
            help="Path to proof.",
        )
        parser.add_argument(
            "-c",
            "--circuit-path",
            help="Path to the compiled circuit.",
        )
        parser.add_argument(
            "-i",
            "--input-path",
            help="Path to input JSON.",
        )
        parser.add_argument(
            "-o",
            "--output-path",
            help="Path to expected outputs JSON.",
        )
        parser.add_argument(
            "-w",
            "--witness-path",
            help="Path to witness.",
        )
        parser.add_argument("-p", "--proof-path", help="Path to proof.")

    @classmethod
    @BaseCommand.validate_required(
        "circuit_path",
        "input_path",
        "output_path",
        "witness_path",
        "proof_path",
    )
    @BaseCommand.validate_paths(
        "circuit_path",
        "input_path",
        "output_path",
        "witness_path",
        "proof_path",
    )
    def run(cls: type[VerifyCommand], args: argparse.Namespace) -> None:
        circuit = cls._build_circuit("cli")

        try:
            circuit.base_testing(
                CircuitExecutionConfig(
                    run_type=RunType.GEN_VERIFY,
                    circuit_path=args.circuit_path,
                    input_file=args.input_path,
                    output_file=args.output_path,
                    witness_file=args.witness_path,
                    proof_file=args.proof_path,
                    ecc=False,
                ),
            )
        except CircuitRunError as e:
            raise RuntimeError(e) from e

        print(  # noqa: T201
            f"[verify] verification complete for proof → {args.proof_path}",
        )
