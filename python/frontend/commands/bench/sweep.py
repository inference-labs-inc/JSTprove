from __future__ import annotations

import sys
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import argparse

from python.core.utils.helper_functions import run_subprocess
from python.frontend.commands.args import ArgSpec
from python.frontend.commands.base import BaseCommand

SWEEP_CHOICES: tuple[str, ...] = ("depth", "breadth", "lenet")

DEPTH_MIN = ArgSpec(
    name="depth_min",
    flag="--depth-min",
    help_text="(depth) minimum conv depth",
    arg_type=int,
)

DEPTH_MAX = ArgSpec(
    name="depth_max",
    flag="--depth-max",
    help_text="(depth) maximum conv depth",
    arg_type=int,
)

INPUT_HW = ArgSpec(
    name="input_hw",
    flag="--input-hw",
    help_text="(depth) input H=W (e.g., 56)",
    arg_type=int,
)

ARCH_DEPTH = ArgSpec(
    name="arch_depth",
    flag="--arch-depth",
    help_text="(breadth) conv blocks at fixed topology",
    arg_type=int,
)

INPUT_HW_LIST = ArgSpec(
    name="input_hw_list",
    flag="--input-hw-list",
    help_text="(breadth) sizes like '28,56,84,112' or '32:160:32'",
)

ITERATIONS = ArgSpec(
    name="iterations",
    flag="--iterations",
    help_text="E2E loops per model (default 3 in sweeps)",
    arg_type=int,
)

RESULTS = ArgSpec(
    name="results",
    flag="--results",
    help_text="Path to JSONL results (e.g., benchmarking/depth_sweep.jsonl)",
)

ONNX_DIR = ArgSpec(
    name="onnx_dir",
    flag="--onnx-dir",
    help_text="Override ONNX output dir (else chosen by sweep)",
)

INPUTS_DIR = ArgSpec(
    name="inputs_dir",
    flag="--inputs-dir",
    help_text="Override inputs output dir (else chosen by sweep)",
)

POOL_CAP = ArgSpec(
    name="pool_cap",
    flag="--pool-cap",
    help_text="Max pool blocks at start (Lenet-like: 2)",
    arg_type=int,
)

STOP_AT_HW = ArgSpec(
    name="stop_at_hw",
    flag="--stop-at-hw",
    help_text="Allow pooling while H >= this",
    arg_type=int,
)

CONV_OUT_CH = ArgSpec(
    name="conv_out_ch",
    flag="--conv-out-ch",
    help_text="Conv output channels",
    arg_type=int,
)

FC_HIDDEN = ArgSpec(
    name="fc_hidden",
    flag="--fc-hidden",
    help_text="Fully-connected hidden size",
    arg_type=int,
)

N_ACTIONS = ArgSpec(
    name="n_actions",
    flag="--n-actions",
    help_text="Classifier outputs (classes)",
    arg_type=int,
)

TAG = ArgSpec(
    name="tag",
    flag="--tag",
    help_text="Optional tag suffix for filenames",
)

MODE = ArgSpec(
    name="mode",
    flag="mode",
    help_text="Sweep type: depth, breadth, or lenet.",
    extra_kwargs={"choices": list(SWEEP_CHOICES)},
)


class SweepCommand(BaseCommand):
    """Generate ONNX models and benchmark parameter sweeps (depth/breadth)."""

    name: ClassVar[str] = "sweep"
    aliases: ClassVar[list[str]] = []
    help: ClassVar[str] = (
        "Generate ONNX models and benchmark parameter sweeps (depth/breadth)."
    )

    SCRIPT_GEN_AND_BENCH: ClassVar[str] = "python.scripts.gen_and_bench"

    DEPTH_DEFAULTS: ClassVar[tuple[tuple[ArgSpec, str], ...]] = (
        (DEPTH_MIN, "1"),
        (DEPTH_MAX, "16"),
        (ITERATIONS, "3"),
        (RESULTS, "benchmarking/depth_sweep.jsonl"),
    )

    BREADTH_DEFAULTS: ClassVar[tuple[tuple[ArgSpec, str], ...]] = (
        (ARCH_DEPTH, "5"),
        (INPUT_HW_LIST, "28,56,84,112"),
        (ITERATIONS, "3"),
        (RESULTS, "benchmarking/breadth_sweep.jsonl"),
        (POOL_CAP, "2"),
        (CONV_OUT_CH, "16"),
        (FC_HIDDEN, "256"),
    )

    SWEEP_CONFIGS: ClassVar[dict[str, tuple[tuple[ArgSpec, str], ...]]] = {
        SWEEP_CHOICES[0]: DEPTH_DEFAULTS,
        SWEEP_CHOICES[1]: BREADTH_DEFAULTS,
    }

    SWEEP_ARGS: ClassVar[tuple[ArgSpec, ...]] = (
        DEPTH_MIN,
        DEPTH_MAX,
        INPUT_HW,
        ARCH_DEPTH,
        INPUT_HW_LIST,
        ITERATIONS,
        RESULTS,
        ONNX_DIR,
        INPUTS_DIR,
        POOL_CAP,
        STOP_AT_HW,
        CONV_OUT_CH,
        FC_HIDDEN,
        N_ACTIONS,
        TAG,
    )

    OPTIONAL_SWEEP_ARGS: ClassVar[tuple[ArgSpec, ...]] = (ONNX_DIR, INPUTS_DIR, TAG)

    @classmethod
    def configure_parser(
        cls: type[SweepCommand],
        parser: argparse.ArgumentParser,
    ) -> None:
        MODE.add_to_parser(parser)

        for arg_spec in cls.SWEEP_ARGS:
            arg_spec.add_to_parser(parser)

    @classmethod
    def run(cls: type[SweepCommand], args: argparse.Namespace) -> None:
        if not args.mode:
            msg = "Specify sweep type: depth, breadth, or lenet"
            raise ValueError(msg)

        cls._run_sweep_benchmark(args, args.mode)

    @classmethod
    def _run_sweep_benchmark(
        cls: type[SweepCommand],
        args: argparse.Namespace,
        sweep: str,
    ) -> None:
        provided_knobs = [getattr(args, spec.name, None) for spec in cls.SWEEP_ARGS]
        simple = all(v is None for v in provided_knobs)

        cmd = [sys.executable, "-m", cls.SCRIPT_GEN_AND_BENCH, "--sweep", sweep]

        if simple:
            defaults = cls.SWEEP_CONFIGS.get(sweep, ())
            BaseCommand.append_args_from_specs(cmd, defaults)
            BaseCommand.append_args_from_namespace(cmd, args, cls.OPTIONAL_SWEEP_ARGS)
        else:
            BaseCommand.append_args_from_namespace(cmd, args, cls.SWEEP_ARGS)

        run_subprocess(cmd)
