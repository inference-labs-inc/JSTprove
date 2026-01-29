from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from python.core.utils.helper_functions import (
    read_from_json,
    run_cargo_command,
    to_json,
)
from python.frontend.commands.args import CIRCUIT_PATH
from python.frontend.commands.base import BaseCommand

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

    from python.core.circuits.base import Circuit


def _preprocess_manifest(
    circuit: Circuit,
    manifest_path: str,
    circuit_path: str,
    transform_job: Callable[[Circuit, dict[str, Any]], None],
) -> str:
    circuit_file = Path(circuit_path)
    quantized_path = str(
        circuit_file.parent / f"{circuit_file.stem}_quantized_model.onnx",
    )
    circuit.load_quantized_model(quantized_path)

    manifest: dict[str, Any] = read_from_json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("jobs"),
        list,
    ):
        msg = f"Invalid manifest: expected {{'jobs': [...]}} in {manifest_path}"
        raise TypeError(msg)
    for job in manifest["jobs"]:
        transform_job(circuit, job)

    manifest_file = Path(manifest_path)
    processed_path = str(
        manifest_file.with_name(
            manifest_file.stem + "_processed" + manifest_file.suffix,
        ),
    )
    to_json(manifest, processed_path)
    return processed_path


def _transform_witness_job(circuit: Circuit, job: dict[str, Any]) -> None:
    inputs = read_from_json(job["input"])
    scaled = circuit.scale_inputs_only(inputs)

    inference_inputs = circuit.reshape_inputs_for_inference(scaled)
    circuit_inputs = circuit.reshape_inputs_for_circuit(scaled)

    path = Path(job["input"])
    adjusted_path = str(path.with_name(path.stem + "_adjusted" + path.suffix))
    to_json(circuit_inputs, adjusted_path)

    outputs = circuit.get_outputs(inference_inputs)
    formatted = circuit.format_outputs(outputs)
    to_json(formatted, job["output"])

    job["input"] = adjusted_path


def _transform_verify_job(circuit: Circuit, job: dict[str, Any]) -> None:
    inputs = read_from_json(job["input"])
    circuit_inputs = circuit.reshape_inputs_for_circuit(inputs)

    path = Path(job["input"])
    processed_path = str(path.with_name(path.stem + "_veri" + path.suffix))
    to_json(circuit_inputs, processed_path)

    job["input"] = processed_path


class BatchCommand(BaseCommand):
    """Run batch operations on multiple inputs."""

    name: ClassVar[str] = "batch"
    aliases: ClassVar[list[str]] = []
    help: ClassVar[str] = "Run batch witness/prove/verify operations."

    @classmethod
    def _add_batch_args(cls, subparser: argparse.ArgumentParser) -> None:
        CIRCUIT_PATH.add_to_parser(subparser)
        subparser.add_argument(
            "-f",
            "--manifest",
            required=True,
            help="Path to batch manifest JSON file.",
        )

    @classmethod
    def configure_parser(
        cls: type[BatchCommand],
        parser: argparse.ArgumentParser,
    ) -> None:
        subparsers = parser.add_subparsers(dest="batch_mode", required=True)

        for name, help_text in [
            ("witness", "Generate witnesses for multiple inputs."),
            ("prove", "Generate proofs for multiple witnesses."),
            ("verify", "Verify multiple proofs."),
        ]:
            sub = subparsers.add_parser(name, help=help_text)
            cls._add_batch_args(sub)

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

        preprocess_map = {
            "witness": _transform_witness_job,
            "verify": _transform_verify_job,
        }
        if batch_mode in preprocess_map:
            manifest_path = _preprocess_manifest(
                circuit,
                manifest_path,
                circuit_path,
                preprocess_map[batch_mode],
            )

        try:
            run_cargo_command(
                binary_name=circuit.name,
                command_type=run_type_str,
                args={
                    "c": circuit_path,
                    "f": manifest_path,
                    "m": metadata_path,
                },
                dev_mode=False,
            )
        except Exception as e:
            raise RuntimeError(e) from e

        print(f"[batch {batch_mode}] complete")  # noqa: T201
