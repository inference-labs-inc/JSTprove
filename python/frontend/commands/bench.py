from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import argparse

from python.core.utils.helper_functions import to_json
from python.frontend.commands.base import BaseCommand


class BenchCommand(BaseCommand):
    """Generate ONNX models and benchmark JSTprove
    (depth/breadth sweeps or specific models)."""

    name: ClassVar[str] = "bench"
    aliases: ClassVar[list[str]] = []
    help: ClassVar[str] = (
        "Generate ONNX models and benchmark JSTprove "
        "(depth/breadth sweeps or specific models)."
    )

    @classmethod
    def configure_parser(
        cls: type[BenchCommand],
        parser: argparse.ArgumentParser,
    ) -> None:
        parser.add_argument(
            "mode",
            choices=["depth", "breadth", "lenet"],
            nargs="?",
            help="Shorthand for --sweep when you just want defaults "
            "(e.g., 'jst bench depth').",
        )

        parser.add_argument(
            "--sweep",
            choices=["depth", "breadth", "lenet"],
            default=None,  # allow omission when using positional mode
            help="depth: vary conv depth; breadth: vary input H=W. "
            "If omitted, the positional MODE is used.",
        )

        # depth sweep options
        parser.add_argument("--depth-min", type=int, help="(depth) minimum conv depth")
        parser.add_argument("--depth-max", type=int, help="(depth) maximum conv depth")
        parser.add_argument("--input-hw", type=int, help="(depth) input H=W (e.g., 56)")

        # breadth sweep options
        parser.add_argument(
            "--arch-depth",
            type=int,
            help="(breadth) conv blocks at fixed topology",
        )
        parser.add_argument(
            "--input-hw-list",
            type=str,
            help="(breadth) sizes like '28,56,84,112' or '32:160:32'",
        )

        # common knobs
        parser.add_argument(
            "--iterations",
            type=int,
            help="E2E loops per model (default 3 in examples)",
        )
        parser.add_argument(
            "--results",
            help="Path to JSONL results (e.g., benchmarking/depth_sweep.jsonl)",
        )
        parser.add_argument(
            "--onnx-dir",
            help="Override ONNX output dir (else chosen by sweep)",
        )
        parser.add_argument(
            "--inputs-dir",
            help="Override inputs output dir (else chosen by sweep)",
        )
        parser.add_argument(
            "--pool-cap",
            type=int,
            help="Max pool blocks at start (Lenet-like: 2)",
        )
        parser.add_argument(
            "--stop-at-hw",
            type=int,
            help="Allow pooling while H >= this",
        )
        parser.add_argument("--conv-out-ch", type=int, help="Conv output channels")
        parser.add_argument("--fc-hidden", type=int, help="Fully-connected hidden size")
        parser.add_argument(
            "--n-actions",
            type=int,
            help="Classifier outputs (classes)",
        )
        parser.add_argument("--tag", help="Optional tag suffix for filenames")

        # Model selection options (similar to pytest)
        parser.add_argument(
            "--model",
            action="append",
            default=None,
            help="Model name(s) from registry to benchmark. "
            "Use multiple times to test more than one.",
        )
        parser.add_argument(
            "--model-path",
            help="Direct path to ONNX model file to benchmark (alt. to --model).",
        )
        parser.add_argument(
            "--source",
            choices=["class", "onnx"],
            default=None,
            help="Restrict registry models to a specific source: class or onnx.",
        )
        parser.add_argument(
            "--list-models",
            action="store_true",
            default=False,
            help="List all available circuit models.",
        )

    @classmethod
    def run(cls: type[BenchCommand], args: argparse.Namespace) -> None:
        cls._run_bench(args)

    @staticmethod
    def _run_bench(args: argparse.Namespace) -> None:
        """
        Run benchmarks:
          - depth/breadth: call python.scripts.gen_and_bench (existing sweeps)
          - lenet: run benchmark_runner on the repo's fixed LeNet model/input
          - specific models: run benchmark_runner on selected models from registry
        """
        from python.core.utils.model_registry import (  # noqa: PLC0415
            list_available_models,
        )

        # Handle --list-models
        if args.list_models:
            available_models = list_available_models()
            print("\nAvailable Circuit Models:")  # noqa: T201
            for model in available_models:
                print(f"- {model}")  # noqa: T201
            return

        # Handle direct model path
        if args.model_path:
            BaseCommand._ensure_file_exists(args.model_path)  # noqa: SLF001
            name = Path(args.model_path).stem
            BenchCommand._run_bench_single_model(args, args.model_path, name)
            return

        # Check for model selection
        if args.model or args.source:
            BenchCommand._run_bench_on_models(args)
            return

        sweep = args.sweep or args.mode
        if not sweep:
            msg = "Please specify --sweep {depth|breadth|lenet}, a positional mode, "
            "or --model/--source for specific models."
            raise ValueError(msg)

        # --- Depth/breadth ---
        provided_knobs = [
            args.depth_min,
            args.depth_max,
            args.input_hw,
            args.arch_depth,
            args.input_hw_list,
            args.iterations,
            args.results,
            args.pool_cap,
            args.stop_at_hw,
            args.conv_out_ch,
            args.fc_hidden,
            args.n_actions,
        ]
        simple = (args.mode is not None) and all(v is None for v in provided_knobs)

        cmd = [sys.executable, "-m", "python.scripts.gen_and_bench", "--sweep", sweep]
        if simple:
            if sweep == "depth":
                cmd += [
                    "--depth-min",
                    "1",
                    "--depth-max",
                    "16",
                    "--iterations",
                    "3",
                    "--results",
                    "benchmarking/depth_sweep.jsonl",
                ]
            else:  # breadth
                cmd += [
                    "--arch-depth",
                    "5",
                    "--input-hw-list",
                    "28,56,84,112",
                    "--iterations",
                    "3",
                    "--results",
                    "benchmarking/breadth_sweep.jsonl",
                    "--pool-cap",
                    "2",
                    "--conv-out-ch",
                    "16",
                    "--fc-hidden",
                    "256",
                ]
            BenchCommand._append_arg(cmd, "--onnx-dir", args.onnx_dir)
            BenchCommand._append_arg(cmd, "--inputs-dir", args.inputs_dir)
            BenchCommand._append_arg(cmd, "--tag", args.tag)
        else:
            BenchCommand._append_arg(cmd, "--depth-min", args.depth_min)
            BenchCommand._append_arg(cmd, "--depth-max", args.depth_max)
            BenchCommand._append_arg(cmd, "--input-hw", args.input_hw)
            BenchCommand._append_arg(cmd, "--arch-depth", args.arch_depth)
            BenchCommand._append_arg(cmd, "--input-hw-list", args.input_hw_list)
            BenchCommand._append_arg(cmd, "--iterations", args.iterations)
            BenchCommand._append_arg(cmd, "--results", args.results)
            BenchCommand._append_arg(cmd, "--onnx-dir", args.onnx_dir)
            BenchCommand._append_arg(cmd, "--inputs-dir", args.inputs_dir)
            BenchCommand._append_arg(cmd, "--pool-cap", args.pool_cap)
            BenchCommand._append_arg(cmd, "--stop-at-hw", args.stop_at_hw)
            BenchCommand._append_arg(cmd, "--conv-out-ch", args.conv_out_ch)
            BenchCommand._append_arg(cmd, "--fc-hidden", args.fc_hidden)
            BenchCommand._append_arg(cmd, "--n-actions", args.n_actions)
            BenchCommand._append_arg(cmd, "--tag", args.tag)

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.run(cmd, text=True, env=env, check=False)  # noqa: S603
        if proc.returncode != 0:
            msg = f"Benchmark command failed with exit code {proc.returncode}"
            raise RuntimeError(msg)

    @staticmethod
    def _run_bench_on_models(args: argparse.Namespace) -> None:
        """Run benchmarks on selected models from the registry."""

        from python.core.utils.model_registry import get_models_to_test  # noqa: PLC0415

        selected_models = args.model
        source_filter = args.source
        models = get_models_to_test(selected_models, source_filter)
        # Filter to ONNX models only, as bench uses benchmark_runner
        models = [m for m in models if m.source == "onnx"]
        if not models:
            msg = "No ONNX models selected for benchmarking."
            raise ValueError(msg)

        # Run benchmarks for each selected model
        for model_entry in models:
            # Instantiate to get the path
            instance = model_entry.loader()
            model_path = instance.model_file_name
            name = model_entry.name
            BenchCommand._run_bench_single_model(args, model_path, name)

    @staticmethod
    def _run_bench_single_model(
        args: argparse.Namespace,
        model_path: str,
        name: str,
    ) -> None:
        """Run benchmark on a single model path."""

        # Generate input on the fly, similar to tests
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_file = tmp_path / "input.json"

            # Create instance for the model
            instance = BaseCommand._build_circuit()  # noqa: SLF001
            instance.model_file_name = model_path

            # Load the model to set input_shape
            try:
                instance.load_model(model_path)
            except Exception as e:
                msg = f"Failed to load model {model_path}: {e}"
                raise RuntimeError(msg) from e

            # Generate random inputs and format them
            try:
                inputs = instance.get_inputs()  # generates random inputs
                formatted_inputs = instance.format_inputs(inputs)
                to_json(formatted_inputs, str(input_file))
            except Exception as e:
                msg = f"Failed to generate input for {name}: {e}"
                raise RuntimeError(msg) from e

            # Now run benchmark with the generated input
            iterations = str(args.iterations if args.iterations is not None else 5)
            results = (
                args.results
                if (args.results and str(args.results).strip())
                else f"benchmarking/{name}.jsonl"
            )
            Path(results).parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-m",
                "python.scripts.benchmark_runner",
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
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            rc = subprocess.run(  # noqa: S603
                cmd,
                text=True,
                env=env,
                check=False,
            ).returncode
            if rc != 0:
                msg = f"Benchmark for {name} failed with exit code {rc}"
                raise RuntimeError(msg)

    @staticmethod
    def _append_arg(cmd: list[str], flag: str, val: object | None) -> None:
        """
        Append '--flag val' only if val is not None/empty (for simple scalars/strings).
        """
        if val is None:
            return
        if isinstance(val, str) and not val.strip():
            return
        cmd += [flag, str(val)]
