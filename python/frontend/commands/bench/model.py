from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import argparse

from python.core.utils.constants import MODEL_SOURCE_CLASS, MODEL_SOURCE_ONNX
from python.core.utils.helper_functions import (
    ensure_parent_dir,
    run_subprocess,
    to_json,
)
from python.frontend.commands.args import ArgSpec
from python.frontend.commands.base import BaseCommand

SOURCE_CHOICES: tuple[str, ...] = (MODEL_SOURCE_CLASS, MODEL_SOURCE_ONNX)

BENCH_MODEL_PATH = ArgSpec(
    name="model_path",
    flag="--model-path",
    short="-m",
    help_text="Direct path to ONNX model file to benchmark.",
)

MODEL = ArgSpec(
    name="model",
    flag="--model",
    help_text=(
        "Model name(s) from registry to benchmark. "
        "Use multiple times to test more than one."
    ),
    extra_kwargs={"action": "append", "default": None},
)

SOURCE = ArgSpec(
    name="source",
    flag="--source",
    help_text="Restrict registry models to a specific source: class or onnx.",
    extra_kwargs={"choices": list(SOURCE_CHOICES), "default": None},
)

ITERATIONS = ArgSpec(
    name="iterations",
    flag="--iterations",
    help_text="E2E loops per model (default 5)",
    arg_type=int,
)

RESULTS = ArgSpec(
    name="results",
    flag="--results",
    help_text="Path to JSONL results (e.g., benchmarking/model_name.jsonl)",
)


class ModelCommand(BaseCommand):
    """Benchmark specific models from registry or file path."""

    name: ClassVar[str] = "model"
    aliases: ClassVar[list[str]] = []
    help: ClassVar[str] = "Benchmark specific models from registry or file path."

    DEFAULT_ITERATIONS: ClassVar[int] = 5
    SCRIPT_BENCHMARK_RUNNER: ClassVar[str] = "python.scripts.benchmark_runner"

    @classmethod
    def configure_parser(
        cls: type[ModelCommand],
        parser: argparse.ArgumentParser,
    ) -> None:
        BENCH_MODEL_PATH.add_to_parser(parser)
        MODEL.add_to_parser(parser)
        SOURCE.add_to_parser(parser)
        ITERATIONS.add_to_parser(parser)
        RESULTS.add_to_parser(parser)

    @classmethod
    @BaseCommand.validate_optional_paths(BENCH_MODEL_PATH)
    def run(cls: type[ModelCommand], args: argparse.Namespace) -> None:
        if args.model_path:
            name = Path(args.model_path).stem
            cls._run_bench_single_model(args, args.model_path, name)
            return

        if args.model or args.source:
            cls._run_bench_on_models(args)
            return

        msg = "Specify --model-path, --model, or --source"
        raise ValueError(msg)

    @classmethod
    def _run_bench_on_models(cls: type[ModelCommand], args: argparse.Namespace) -> None:
        from python.core.utils.model_registry import get_models_to_test  # noqa: PLC0415

        models = get_models_to_test(args.model, args.source or SOURCE_CHOICES[1])
        if not models:
            msg = "No models selected for benchmarking."
            raise ValueError(msg)

        for model_entry in models:
            instance = model_entry.loader()
            cls._run_bench_single_model(
                args,
                instance.model_file_name,
                model_entry.name,
            )

    @classmethod
    def _generate_model_input(
        cls: type[ModelCommand],
        model_path: str,
        input_file: Path,
        name: str,
    ) -> None:
        instance = BaseCommand._build_circuit()  # noqa: SLF001
        instance.model_file_name = model_path

        try:
            instance.load_model(model_path)
        except Exception as e:
            msg = f"Failed to load model {model_path}: {e}"
            raise RuntimeError(msg) from e

        try:
            inputs = instance.get_inputs()
            formatted_inputs = instance.format_inputs(inputs)
            to_json(formatted_inputs, str(input_file))
        except Exception as e:
            msg = f"Failed to generate input for {name}: {e}"
            raise RuntimeError(msg) from e

    @classmethod
    def _run_bench_single_model(
        cls: type[ModelCommand],
        args: argparse.Namespace,
        model_path: str,
        name: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_file = tmp_path / "input.json"

            cls._generate_model_input(model_path, input_file, name)

            iterations = str(args.iterations or cls.DEFAULT_ITERATIONS)
            results = args.results or f"benchmarking/{name}.jsonl"
            ensure_parent_dir(results)

            cmd = [
                sys.executable,
                "-m",
                cls.SCRIPT_BENCHMARK_RUNNER,
                "--model",
                model_path,
                "--input",
                str(input_file),
                "--iterations",
                iterations,
                "--output",
                results,
                "--summarize",
            ]
            if os.environ.get("JSTPROVE_DEBUG") == "1":
                print(f"[debug] bench {name} cmd:", " ".join(cmd))  # noqa: T201

            run_subprocess(cmd)
