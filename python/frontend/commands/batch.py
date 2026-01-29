from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import argparse

from python.core.utils.helper_functions import run_cargo_command
from python.frontend.commands.args import CIRCUIT_PATH
from python.frontend.commands.base import BaseCommand


class BatchCommand(BaseCommand):
    """Run batch operations on multiple inputs."""

    name: ClassVar[str] = "batch"
    aliases: ClassVar[list[str]] = []
    help: ClassVar[str] = "Run batch witness/prove/verify operations."

    @classmethod
    def configure_parser(
        cls: type[BatchCommand],
        parser: argparse.ArgumentParser,
    ) -> None:
        subparsers = parser.add_subparsers(dest="batch_mode", required=True)

        witness_parser = subparsers.add_parser(
            "witness",
            help="Generate witnesses for multiple inputs.",
        )
        CIRCUIT_PATH.add_to_parser(witness_parser)
        witness_parser.add_argument(
            "-f",
            "--manifest",
            required=True,
            help="Path to batch manifest JSON file.",
        )
        witness_parser.add_argument(
            "-j",
            "--parallel",
            type=int,
            default=1,
            help="Number of parallel threads (default: 1).",
        )

        prove_parser = subparsers.add_parser(
            "prove",
            help="Generate proofs for multiple witnesses.",
        )
        CIRCUIT_PATH.add_to_parser(prove_parser)
        prove_parser.add_argument(
            "-f",
            "--manifest",
            required=True,
            help="Path to batch manifest JSON file.",
        )
        prove_parser.add_argument(
            "-j",
            "--parallel",
            type=int,
            default=1,
            help="Number of parallel threads (default: 1).",
        )

        verify_parser = subparsers.add_parser(
            "verify",
            help="Verify multiple proofs.",
        )
        CIRCUIT_PATH.add_to_parser(verify_parser)
        verify_parser.add_argument(
            "-f",
            "--manifest",
            required=True,
            help="Path to batch manifest JSON file.",
        )
        verify_parser.add_argument(
            "-j",
            "--parallel",
            type=int,
            default=1,
            help="Number of parallel threads (default: 1).",
        )

    @classmethod
    def run(cls: type[BatchCommand], args: argparse.Namespace) -> None:
        circuit_path = getattr(args, "circuit_path", None) or getattr(
            args,
            "pos_circuit_path",
            None,
        )
        if not circuit_path:
            msg = "Missing required argument: circuit_path"
            raise ValueError(msg)

        circuit_file = Path(circuit_path)
        if not circuit_file.is_file():
            msg = f"Circuit file not found: {circuit_path}"
            raise FileNotFoundError(msg)

        manifest_path = args.manifest
        if not Path(manifest_path).is_file():
            msg = f"Manifest file not found: {manifest_path}"
            raise FileNotFoundError(msg)

        parallel = args.parallel
        batch_mode = args.batch_mode

        circuit = cls._build_circuit("cli")

        run_type_map = {
            "witness": "run_batch_witness",
            "prove": "run_batch_prove",
            "verify": "run_batch_verify",
        }

        run_type_str = run_type_map.get(batch_mode)
        if not run_type_str:
            msg = f"Unknown batch mode: {batch_mode}"
            raise ValueError(msg)

        circuit_dir = circuit_file.parent
        name = circuit_file.stem
        metadata_path = str(circuit_dir / f"{name}_metadata.json")

        try:
            run_cargo_command(
                binary_name=circuit.name,
                command_type=run_type_str,
                args={
                    "c": circuit_path,
                    "f": manifest_path,
                    "j": str(parallel),
                    "m": metadata_path,
                },
                dev_mode=False,
            )
        except Exception as e:
            raise RuntimeError(e) from e

        print(f"[batch {batch_mode}] complete")  # noqa: T201
