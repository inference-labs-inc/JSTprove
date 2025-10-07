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
    def run(cls: type[VerifyCommand], args: argparse.Namespace) -> None:
        args.circuit_path = args.circuit_path or args.pos_circuit_path
        args.input_path = args.input_path or args.pos_input_path
        args.output_path = args.output_path or args.pos_output_path
        args.witness_path = args.witness_path or args.pos_witness_path
        args.proof_path = args.proof_path or args.pos_proof_path

        if not all(
            [
                args.circuit_path,
                args.input_path,
                args.output_path,
                args.witness_path,
                args.proof_path,
            ],
        ):
            msg = (
                "verify requires circuit_path, input_path, output_path, "
                "witness_path, and proof_path"
            )
            raise ValueError(msg)

        cls._ensure_file_exists(args.circuit_path)
        cls._ensure_file_exists(args.input_path)
        cls._ensure_file_exists(args.output_path)
        cls._ensure_file_exists(args.witness_path)
        cls._ensure_file_exists(args.proof_path)

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
