from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import argparse

from python.core.circuits.errors import CircuitRunError
from python.core.utils.helper_functions import CircuitExecutionConfig, RunType
from python.frontend.commands.args import CIRCUIT_PATH, PROOF_PATH, WITNESS_PATH
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
        CIRCUIT_PATH.add_to_parser(parser)
        WITNESS_PATH.add_to_parser(parser, "Path to an existing witness.")
        PROOF_PATH.add_to_parser(parser)

    @classmethod
    @BaseCommand.validate_required(CIRCUIT_PATH, WITNESS_PATH, PROOF_PATH)
    @BaseCommand.validate_paths(CIRCUIT_PATH, WITNESS_PATH)
    @BaseCommand.validate_parent_paths(PROOF_PATH)
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
