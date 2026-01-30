from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import argparse

from python.core.circuits.errors import CircuitRunError
from python.core.utils.helper_functions import CircuitExecutionConfig, RunType
from python.frontend.commands.args import (
    CIRCUIT_PATH,
    INPUT_PATH,
    OUTPUT_PATH,
    PROOF_PATH,
    VKEY_PATH,
    WITNESS_PATH,
)
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
        CIRCUIT_PATH.add_to_parser(parser)
        VKEY_PATH.add_to_parser(parser)
        INPUT_PATH.add_to_parser(parser)
        OUTPUT_PATH.add_to_parser(parser, "Path to expected outputs JSON.")
        WITNESS_PATH.add_to_parser(parser)
        PROOF_PATH.add_to_parser(parser)

    @classmethod
    @BaseCommand.validate_required(
        INPUT_PATH,
        OUTPUT_PATH,
        PROOF_PATH,
    )
    @BaseCommand.validate_paths(
        INPUT_PATH,
        OUTPUT_PATH,
        PROOF_PATH,
    )
    @BaseCommand.validate_optional_paths(
        CIRCUIT_PATH,
        VKEY_PATH,
        WITNESS_PATH,
    )
    def run(cls: type[VerifyCommand], args: argparse.Namespace) -> None:
        circuit_path = getattr(args, "circuit_path", None)
        vkey_path = getattr(args, "vkey_path", None)
        if not circuit_path and not vkey_path:
            msg = "Either --circuit-path or --vkey-path must be provided"
            raise ValueError(msg)

        circuit = cls._build_circuit("cli")

        try:
            circuit.base_testing(
                CircuitExecutionConfig(
                    run_type=RunType.GEN_VERIFY,
                    circuit_path=circuit_path,
                    verification_key=vkey_path,
                    input_file=args.input_path,
                    output_file=args.output_path,
                    witness_file=getattr(args, "witness_path", None),
                    proof_file=args.proof_path,
                    ecc=False,
                ),
            )
        except CircuitRunError as e:
            raise RuntimeError(e) from e

        print(  # noqa: T201
            f"[verify] verification complete for proof → {args.proof_path}",
        )
