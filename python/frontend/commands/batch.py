from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from python.core.utils.helper_functions import (
    read_from_json,
    run_cargo_command,
    run_cargo_command_piped,
    to_json,
)
from python.frontend.commands.args import CIRCUIT_PATH
from python.frontend.commands.base import BaseCommand

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

    from python.core.circuits.base import Circuit


def _parse_piped_result(stdout: bytes) -> dict[str, Any]:
    lines = stdout.strip().split(b"\n")
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith(b"{"):
            return json.loads(stripped)
    msg = f"No JSON object found in piped stdout: {stdout[:200]}"
    raise ValueError(msg)


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


def _validate_job_keys(job: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in job]
    if missing:
        msg = f"Job missing required keys {missing}: {job}"
        raise ValueError(msg)


def _transform_witness_job(circuit: Circuit, job: dict[str, Any]) -> None:
    _validate_job_keys(job, "input", "output")
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
    _validate_job_keys(job, "input")
    inputs = read_from_json(job["input"])
    circuit_inputs = circuit.reshape_inputs_for_circuit(inputs)

    path = Path(job["input"])
    processed_path = str(path.with_name(path.stem + "_veri" + path.suffix))
    to_json(circuit_inputs, processed_path)

    job["input"] = processed_path


def _run_witness_chunk_piped(
    binary_name: str,
    circuit_path: str,
    metadata_path: str | None,
    chunk_jobs: list[dict[str, Any]],
    wandb_path: str | None = None,
) -> dict[str, Any]:
    payload_obj = {
        "jobs": [
            {
                "input": job["_circuit_inputs"],
                "output": job["_circuit_outputs"],
                "witness": job["witness"],
            }
            for job in chunk_jobs
        ],
    }
    payload = json.dumps(payload_obj).encode()

    args: dict[str, str] = {"c": circuit_path}
    if metadata_path:
        args["m"] = metadata_path
    if wandb_path:
        args["b"] = wandb_path

    result = run_cargo_command_piped(
        binary_name=binary_name,
        command_type="run_pipe_witness",
        payload=payload,
        args=args,
    )

    return _parse_piped_result(result.stdout)


def _resolve_metadata_path(circuit_path: str) -> str | None:
    circuit_file = Path(circuit_path)
    candidate = circuit_file.parent / f"{circuit_file.stem}_metadata.json"
    if candidate.exists():
        return str(candidate)
    return None


def batch_witness_from_tensors(
    circuit: Circuit,
    jobs: list[dict[str, Any]],
    circuit_path: str,
    chunk_size: int = 0,
    wandb_path: str | None = None,
) -> dict[str, Any]:
    circuit_file = Path(circuit_path)
    file_stem = circuit_file.stem
    binary_name = circuit.name
    metadata_path = _resolve_metadata_path(circuit_path)
    quantized_path = str(circuit_file.parent / f"{file_stem}_quantized_model.onnx")
    circuit.load_quantized_model(quantized_path)

    piped_jobs: list[dict[str, Any]] = []
    for job in jobs:
        raw = job["input"]
        inputs = read_from_json(raw) if isinstance(raw, str) else raw
        scaled = circuit.scale_inputs_only(inputs)
        inference_inputs = circuit.reshape_inputs_for_inference(scaled)
        circuit_inputs = circuit.reshape_inputs_for_circuit(scaled)
        outputs = circuit.get_outputs(inference_inputs)
        formatted = circuit.format_outputs(outputs)
        piped_jobs.append(
            {
                "_circuit_inputs": circuit_inputs,
                "_circuit_outputs": formatted,
                "witness": job["witness"],
            },
        )

    if chunk_size <= 0:
        chunks = [piped_jobs]
    else:
        chunks = [
            piped_jobs[i : i + chunk_size]
            for i in range(0, len(piped_jobs), chunk_size)
        ]

    total_succeeded = 0
    total_failed = 0
    all_errors: list[Any] = []

    for chunk in chunks:
        result = _run_witness_chunk_piped(
            binary_name=binary_name,
            circuit_path=circuit_path,
            metadata_path=metadata_path,
            chunk_jobs=chunk,
            wandb_path=wandb_path,
        )
        total_succeeded += result.get("succeeded", 0)
        total_failed += result.get("failed", 0)
        all_errors.extend(result.get("errors", []))

    return {
        "succeeded": total_succeeded,
        "failed": total_failed,
        "errors": all_errors,
    }


def _run_prove_chunk_piped(
    binary_name: str,
    circuit_path: str,
    metadata_path: str | None,
    chunk_jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    payload_obj = {
        "jobs": [
            {"witness": job["witness"], "proof": job["proof"]} for job in chunk_jobs
        ],
    }
    payload = json.dumps(payload_obj).encode()

    args: dict[str, str] = {"c": circuit_path}
    if metadata_path:
        args["m"] = metadata_path

    result = run_cargo_command_piped(
        binary_name=binary_name,
        command_type="run_pipe_prove",
        payload=payload,
        args=args,
    )

    return _parse_piped_result(result.stdout)


def batch_prove_piped(
    binary_name: str,
    jobs: list[dict[str, Any]],
    circuit_path: str,
    chunk_size: int = 0,
) -> dict[str, Any]:
    metadata_path = _resolve_metadata_path(circuit_path)

    if chunk_size <= 0:
        chunks = [jobs]
    else:
        chunks = [jobs[i : i + chunk_size] for i in range(0, len(jobs), chunk_size)]

    total_succeeded = 0
    total_failed = 0
    all_errors: list[Any] = []

    for chunk in chunks:
        result = _run_prove_chunk_piped(
            binary_name=binary_name,
            circuit_path=circuit_path,
            metadata_path=metadata_path,
            chunk_jobs=chunk,
        )
        total_succeeded += result.get("succeeded", 0)
        total_failed += result.get("failed", 0)
        all_errors.extend(result.get("errors", []))

    return {
        "succeeded": total_succeeded,
        "failed": total_failed,
        "errors": all_errors,
    }


def _run_verify_chunk_piped(
    binary_name: str,
    circuit_path: str,
    metadata_path: str | None,
    chunk_jobs: list[dict[str, Any]],
    wandb_path: str | None = None,
) -> dict[str, Any]:
    payload_obj = {
        "jobs": [
            {
                "input": job["_circuit_inputs"],
                "output": job["_circuit_outputs"],
                "witness": job["witness"],
                "proof": job["proof"],
            }
            for job in chunk_jobs
        ],
    }
    payload = json.dumps(payload_obj).encode()

    args: dict[str, str] = {"c": circuit_path}
    if metadata_path:
        args["m"] = metadata_path
    if wandb_path:
        args["b"] = wandb_path

    result = run_cargo_command_piped(
        binary_name=binary_name,
        command_type="run_pipe_verify",
        payload=payload,
        args=args,
    )

    return _parse_piped_result(result.stdout)


def batch_verify_from_tensors(
    circuit: Circuit,
    jobs: list[dict[str, Any]],
    circuit_path: str,
    chunk_size: int = 0,
    wandb_path: str | None = None,
) -> dict[str, Any]:
    circuit_file = Path(circuit_path)
    file_stem = circuit_file.stem
    binary_name = circuit.name
    metadata_path = _resolve_metadata_path(circuit_path)
    quantized_path = str(circuit_file.parent / f"{file_stem}_quantized_model.onnx")
    circuit.load_quantized_model(quantized_path)

    piped_jobs: list[dict[str, Any]] = []
    for job in jobs:
        raw_in = job["input"]
        inputs = read_from_json(raw_in) if isinstance(raw_in, str) else raw_in
        circuit_inputs = circuit.reshape_inputs_for_circuit(inputs)
        raw_out = job["output"]
        outputs = read_from_json(raw_out) if isinstance(raw_out, str) else raw_out
        piped_jobs.append(
            {
                "_circuit_inputs": circuit_inputs,
                "_circuit_outputs": outputs,
                "witness": job["witness"],
                "proof": job["proof"],
            },
        )

    if chunk_size <= 0:
        chunks = [piped_jobs]
    else:
        chunks = [
            piped_jobs[i : i + chunk_size]
            for i in range(0, len(piped_jobs), chunk_size)
        ]

    total_succeeded = 0
    total_failed = 0
    all_errors: list[Any] = []

    for chunk in chunks:
        result = _run_verify_chunk_piped(
            binary_name=binary_name,
            circuit_path=circuit_path,
            metadata_path=metadata_path,
            chunk_jobs=chunk,
            wandb_path=wandb_path,
        )
        total_succeeded += result.get("succeeded", 0)
        total_failed += result.get("failed", 0)
        all_errors.extend(result.get("errors", []))

    return {
        "succeeded": total_succeeded,
        "failed": total_failed,
        "errors": all_errors,
    }


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

        metadata_path = _resolve_metadata_path(circuit_path)

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
            cargo_args: dict[str, str] = {
                "c": circuit_path,
                "f": manifest_path,
            }
            if metadata_path:
                cargo_args["m"] = metadata_path
            run_cargo_command(
                binary_name=circuit.name,
                command_type=run_type_str,
                args=cargo_args,
                dev_mode=False,
            )
        except Exception as e:
            raise RuntimeError(e) from e

        print(f"[batch {batch_mode}] complete")  # noqa: T201
