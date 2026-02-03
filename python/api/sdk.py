from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from google.protobuf.message import DecodeError
from onnx import numpy_helper

from python.core import RUST_BINARY_NAME
from python.core.circuit_models.generic_onnx import GenericModelONNX
from python.core.utils.helper_functions import (
    compile_circuit,
    generate_proof,
    generate_verification,
    read_from_json,
    to_json,
)
from python.frontend.commands.batch import (
    _run_prove_chunk_piped,
    _run_verify_chunk_piped,
    _run_witness_chunk_piped,
)

logger = logging.getLogger(__name__)

_CONV_BIAS_INPUT_COUNT = 2

SUPPORTED_OPS = {
    "Add",
    "Clip",
    "BatchNormalization",
    "Div",
    "Sub",
    "Mul",
    "Constant",
    "Flatten",
    "Gemm",
    "MaxPool",
    "Max",
    "Min",
    "Relu",
    "Reshape",
    "Conv",
}


@dataclass
class WitnessResult:
    witness_path: str
    output: dict[str, Any]
    rescaled_output: list[float] | None


@dataclass
class BatchResult:
    succeeded: int
    failed: int
    errors: list[Any] = field(default_factory=list)


class Circuit:
    def __init__(self, circuit_path: str | Path) -> None:
        self._circuit_path = Path(circuit_path)
        self._model: GenericModelONNX | None = None
        self._paths = self._resolve_paths()

    def _resolve_paths(self) -> dict[str, str]:
        stem = self._circuit_path.stem
        parent = self._circuit_path.parent
        return {
            "metadata": str(parent / f"{stem}_metadata.json"),
            "architecture": str(parent / f"{stem}_architecture.json"),
            "wandb": str(parent / f"{stem}_wandb.json"),
            "quantized_model": str(parent / f"{stem}_quantized_model.onnx"),
        }

    def _get_model(self) -> GenericModelONNX:
        if self._model is not None:
            return self._model
        circuit = GenericModelONNX("_")
        quantized_path = self._paths["quantized_model"]
        if not Path(quantized_path).exists():
            msg = f"Quantized model not found: {quantized_path}"
            raise FileNotFoundError(msg)
        circuit.load_quantized_model(quantized_path)
        self._model = circuit
        return circuit

    def _normalize_outputs(self, output_data: dict[str, Any]) -> dict[str, Any]:
        raw_output = output_data.get("raw_output")
        if raw_output is not None:
            return {"output": raw_output}

        output_values = output_data.get("output")
        if output_values is None:
            return output_data

        flat = torch.tensor(output_values).flatten()
        if flat.is_floating_point():
            with Path(self._paths["metadata"]).open() as f:
                metadata = json.load(f)
            scale = metadata["scale_base"] ** metadata["scale_exponent"]
            flat = torch.round(flat * scale).long()
        return {"output": flat.long().tolist()}

    @classmethod
    def compile(
        cls,
        model_path: str | Path,
        output_dir: str | Path,
    ) -> Circuit:
        model_path = Path(model_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        model = onnx.load(str(model_path))
        model = cls.add_zero_bias_to_conv(model)

        fd, preprocessed_path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        preprocessed = Path(preprocessed_path)
        try:
            onnx.save(model, preprocessed_path)

            circuit_obj = GenericModelONNX(preprocessed_path)
            stem = model_path.stem
            circuit_path = str(output_dir / f"{stem}_circuit.txt")
            metadata_path = str(
                output_dir / f"{stem}_circuit_metadata.json",
            )
            architecture_path = str(
                output_dir / f"{stem}_circuit_architecture.json",
            )
            wandb_path = str(output_dir / f"{stem}_circuit_wandb.json")
            quantized_path = str(
                output_dir / f"{stem}_circuit_quantized_model.onnx",
            )

            circuit_obj._compile_preprocessing(  # noqa: SLF001
                metadata_path,
                architecture_path,
                wandb_path,
                quantized_path,
            )

            compile_circuit(
                RUST_BINARY_NAME,
                circuit_path,
                metadata_path,
                architecture_path,
                wandb_path,
                dev_mode=False,
            )

            if not Path(circuit_path).exists():
                msg = (
                    "Compilation succeeded but circuit file "
                    f"not found at {circuit_path}"
                )
                raise RuntimeError(msg)

            return cls(circuit_path)
        finally:
            if preprocessed.exists():
                preprocessed.unlink()

    def _process_inputs(self, inputs: str | Path | dict) -> tuple[dict, dict]:
        if isinstance(inputs, (str, Path)):
            inputs = read_from_json(str(inputs))

        circuit = self._get_model()
        scaled = circuit.scale_inputs_only(inputs)
        inference_inputs = circuit.reshape_inputs_for_inference(scaled)
        circuit_inputs = circuit.reshape_inputs_for_circuit(scaled)
        outputs = circuit.get_outputs(inference_inputs)
        formatted = circuit.format_outputs(outputs)
        return circuit_inputs, formatted

    def generate_witness(
        self,
        inputs: str | Path | dict,
        witness_path: str | Path,
        output_path: str | Path | None = None,
    ) -> WitnessResult:
        witness_path = Path(witness_path)
        witness_path.parent.mkdir(parents=True, exist_ok=True)

        circuit_inputs, formatted = self._process_inputs(inputs)

        result = _run_witness_chunk_piped(
            binary_name=RUST_BINARY_NAME,
            circuit_path=str(self._circuit_path),
            metadata_path=self._paths["metadata"],
            chunk_jobs=[
                {
                    "_circuit_inputs": circuit_inputs,
                    "_circuit_outputs": formatted,
                    "witness": str(witness_path),
                },
            ],
        )

        failed = result.get("failed", 0)
        if failed > 0:
            errors = result.get("errors", [])
            msg = f"Witness generation failed: {errors}"
            raise RuntimeError(msg)

        if output_path is not None:
            to_json(formatted, str(output_path))

        return WitnessResult(
            witness_path=str(witness_path),
            output=formatted,
            rescaled_output=formatted.get("rescaled_output"),
        )

    def generate_witness_batch(
        self,
        jobs: list[dict[str, Any]],
        chunk_size: int = 0,
    ) -> BatchResult:
        piped_jobs: list[dict[str, Any]] = []
        per_job_formatted: list[dict] = []

        for job in jobs:
            raw = job["input"]
            witness = job["witness"]
            Path(witness).parent.mkdir(parents=True, exist_ok=True)

            circuit_inputs, formatted = self._process_inputs(raw)
            piped_jobs.append(
                {
                    "_circuit_inputs": circuit_inputs,
                    "_circuit_outputs": formatted,
                    "witness": witness,
                },
            )
            per_job_formatted.append(formatted)

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
                binary_name=RUST_BINARY_NAME,
                circuit_path=str(self._circuit_path),
                metadata_path=self._paths["metadata"],
                chunk_jobs=chunk,
            )
            total_succeeded += result.get("succeeded", 0)
            total_failed += result.get("failed", 0)
            all_errors.extend(result.get("errors", []))

        for job, formatted in zip(jobs, per_job_formatted, strict=False):
            output = job.get("output")
            if output is not None and Path(job["witness"]).exists():
                to_json(formatted, output)

        return BatchResult(
            succeeded=total_succeeded,
            failed=total_failed,
            errors=all_errors,
        )

    def prove(
        self,
        witness_path: str | Path,
        proof_path: str | Path,
    ) -> str:
        witness_path = Path(witness_path)
        proof_path = Path(proof_path)
        proof_path.parent.mkdir(parents=True, exist_ok=True)

        if not witness_path.exists():
            msg = f"Witness file not found: {witness_path}"
            raise FileNotFoundError(msg)

        generate_proof(
            RUST_BINARY_NAME,
            str(self._circuit_path),
            str(witness_path),
            str(proof_path),
            self._paths["metadata"],
            dev_mode=False,
            ecc=True,
        )

        return str(proof_path)

    def prove_batch(
        self,
        jobs: list[dict[str, Any]],
        chunk_size: int = 0,
    ) -> BatchResult:
        for job in jobs:
            witness = Path(job["witness"])
            if not witness.exists():
                msg = f"Witness file not found: {witness}"
                raise FileNotFoundError(msg)
            Path(job["proof"]).parent.mkdir(parents=True, exist_ok=True)

        if chunk_size <= 0:
            chunks = [jobs]
        else:
            chunks = [jobs[i : i + chunk_size] for i in range(0, len(jobs), chunk_size)]

        total_succeeded = 0
        total_failed = 0
        all_errors: list[Any] = []

        for chunk in chunks:
            result = _run_prove_chunk_piped(
                binary_name=RUST_BINARY_NAME,
                circuit_path=str(self._circuit_path),
                metadata_path=self._paths["metadata"],
                chunk_jobs=chunk,
            )
            total_succeeded += result.get("succeeded", 0)
            total_failed += result.get("failed", 0)
            all_errors.extend(result.get("errors", []))

        return BatchResult(
            succeeded=total_succeeded,
            failed=total_failed,
            errors=all_errors,
        )

    def verify(
        self,
        input_path: str | Path,
        output_path: str | Path,
        witness_path: str | Path,
        proof_path: str | Path,
    ) -> bool:
        input_path = Path(input_path)
        output_path = Path(output_path)
        witness_path = Path(witness_path)
        proof_path = Path(proof_path)

        for p in [input_path, output_path, witness_path, proof_path]:
            if not p.exists():
                msg = f"Required file not found: {p}"
                raise FileNotFoundError(msg)

        circuit = self._get_model()

        with output_path.open() as f:
            output_data = json.load(f)

        scaled_output = self._normalize_outputs(output_data)

        with input_path.open() as f:
            input_data = json.load(f)
        circuit_inputs = circuit.reshape_inputs_for_circuit(input_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            veri_input = str(Path(tmpdir) / "input_veri.json")
            veri_output = str(Path(tmpdir) / "output_veri.json")
            to_json(circuit_inputs, veri_input)
            to_json(scaled_output, veri_output)

            generate_verification(
                RUST_BINARY_NAME,
                str(self._circuit_path),
                veri_input,
                veri_output,
                str(witness_path),
                str(proof_path),
                self._paths["metadata"],
                dev_mode=False,
                ecc=True,
            )

        return True

    def verify_batch(
        self,
        jobs: list[dict[str, Any]],
        chunk_size: int = 0,
    ) -> BatchResult:
        circuit = self._get_model()

        piped_jobs: list[dict[str, Any]] = []
        for job in jobs:
            raw_in = job["input"]
            inputs = read_from_json(raw_in) if isinstance(raw_in, str) else raw_in
            circuit_inputs = circuit.reshape_inputs_for_circuit(inputs)

            raw_out = job["output"]
            outputs = read_from_json(raw_out) if isinstance(raw_out, str) else raw_out
            normalized = self._normalize_outputs(outputs)

            piped_jobs.append(
                {
                    "_circuit_inputs": circuit_inputs,
                    "_circuit_outputs": normalized,
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
                binary_name=RUST_BINARY_NAME,
                circuit_path=str(self._circuit_path),
                metadata_path=self._paths["metadata"],
                chunk_jobs=chunk,
            )
            total_succeeded += result.get("succeeded", 0)
            total_failed += result.get("failed", 0)
            all_errors.extend(result.get("errors", []))

        return BatchResult(
            succeeded=total_succeeded,
            failed=total_failed,
            errors=all_errors,
        )

    @staticmethod
    def add_zero_bias_to_conv(
        model: onnx.ModelProto,
    ) -> onnx.ModelProto:
        for node in model.graph.node:
            if node.op_type == "Conv" and len(node.input) == _CONV_BIAS_INPUT_COUNT:
                weight_name = node.input[1]
                weight_init = next(
                    (i for i in model.graph.initializer if i.name == weight_name),
                    None,
                )
                if weight_init is None:
                    continue

                weight_arr = numpy_helper.to_array(weight_init)
                out_channels = weight_arr.shape[0]

                node_id = node.name or node.output[0] if node.output else weight_name
                bias_name = f"{node_id}_zero_bias"
                zero_bias = np.zeros(out_channels, dtype=weight_arr.dtype)
                bias_init = numpy_helper.from_array(
                    zero_bias,
                    name=bias_name,
                )
                model.graph.initializer.append(bias_init)
                node.input.append(bias_name)

        return model

    @staticmethod
    def is_compatible(model_path: str | Path) -> tuple[bool, set]:
        model_path = Path(model_path)
        if not model_path.exists():
            return False, {"FILE_NOT_FOUND"}

        try:
            model = onnx.load(str(model_path))
            ops = {node.op_type for node in model.graph.node}
            unsupported = ops - SUPPORTED_OPS
        except (OSError, ValueError, KeyError, DecodeError):
            logger.warning("Failed to check compatibility", exc_info=True)
            return False, {"LOAD_ERROR"}
        else:
            if unsupported:
                return False, unsupported
            return True, set()
