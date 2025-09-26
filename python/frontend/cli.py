# python/frontend/cli.py
from __future__ import annotations

# stdlib
import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any
import subprocess

# third-party
# local
from python.core.circuits.errors import CircuitRunError
from python.core.utils.helper_functions import CircuitExecutionConfig, RunType

"""JSTprove CLI."""

# --- constants ---------------------------------------------------------------
DEFAULT_CIRCUIT_MODULE = "python.core.circuit_models.generic_onnx"
DEFAULT_CIRCUIT_CLASS = "GenericModelONNX"

BANNER_TITLE = r"""
  888888  .d8888b. 88888888888                                         
    "88b d88P  Y88b    888                                             
     888 Y88b.         888                                             
     888  "Y888b.      888  88888b.  888d888 .d88b.  888  888  .d88b.  
     888     "Y88b.    888  888 "88b 888P"  d88""88b 888  888 d8P  Y8b 
     888       "888    888  888  888 888    888  888 Y88  88P 88888888 
     88P Y88b  d88P    888  888 d88P 888    Y88..88P  Y8bd8P  Y8b.     
     888  "Y8888P"     888  88888P"  888     "Y88P"    Y88P    "Y8888  
   .d88P                    888                                        
 .d88P"                     888                                        
888P"                       888                                        
"""


class CLIError(Exception):
    """Base exception for known CLI errors."""


# --- ui helpers --------------------------------------------------------------
def print_header() -> None:
    """Print the CLI banner (no side-effects at import time)."""
    print(  # noqa: T201
        BANNER_TITLE
        + "\n"
        + "JSTprove — Verifiable ML by Inference Labs\n"
        + "Based on Polyhedra Network's Expander (GKR-based proving system)\n",
    )


# --- circuit helpers ---------------------------------------------------------
def _import_default_circuit() -> type[Any]:
    """Import the default Circuit class object."""
    mod = importlib.import_module(DEFAULT_CIRCUIT_MODULE)
    try:
        return getattr(mod, DEFAULT_CIRCUIT_CLASS)
    except (ImportError, ModuleNotFoundError) as e:
        msg = "Could not import default circuit module "
        f"'{DEFAULT_CIRCUIT_MODULE}': {e}"
        raise CLIError(msg) from e
    except AttributeError as e:
        msg = f"Default circuit class '{DEFAULT_CIRCUIT_CLASS}'"
        f" not found in '{DEFAULT_CIRCUIT_MODULE}'"
        raise CLIError(msg) from e


def _build_default_circuit(model_name_hint: str | None = None) -> None:
    """
    Instantiate the default Circuit class.

    Some circuit subclasses require a constructor arg (commonly `model_name` or `name`).
    We try a few common constructor signatures in order, falling back gracefully:

      1) cls(model_name=<name>)
      2) cls(name=<name>)
      3) cls(<name>)              # positional
      4) cls()                    # no-arg

    Args:
        model_name_hint: Optional human-friendly name (e.g., from model filename).

    Returns:
        An instance of the default Circuit subclass.

    Raises:
        SystemExit: if none of the constructor patterns work.
    """
    cls = _import_default_circuit()
    name = model_name_hint or "cli"

    # Try several constructor patterns commonly used in the codebase.
    for attempt in (
        lambda: cls(model_name=name),
        lambda: cls(name=name),
        lambda: cls(name),  # positional
        lambda: cls(),  # last resort
    ):
        try:
            return attempt()
        except TypeError:  # noqa: PERF203
            continue
    msg = f"Could not construct {cls.__name__} with/without name '{name}'"
    raise SystemExit(msg)


def _ensure_exists(path: str, kind: str = "file") -> None:
    """
    Fail fast if a required path is missing.

    Args:
        path: Path to check.
        kind: "file" or "dir" — controls the check performed.

    Raises:
        SystemExit: if the required file/dir does not exist.
    """
    p = Path(path)
    if kind == "file":
        if not p.is_file():
            msg = f"Required file not found: {path}"
            raise CLIError(msg)
        if not os.access(p, os.R_OK):
            msg = f"Cannot read file: {path}"
            raise CLIError(msg)
    elif kind == "dir":
        if not p.is_dir():
            msg = f"Required directory not found: {path}"
            raise CLIError(msg)
        if not os.access(p, os.X_OK):
            msg = f"Cannot access directory: {path}"
            raise CLIError(msg)


def _ensure_parent_dir(path: str) -> None:
    """
    Create parent directories for a file path if they don't exist.

    This is a no-op if the dirs already exist.

    Args:
        path: A file path whose parent directories should be ensured.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _run_compile(args: argparse.Namespace) -> None:
    _ensure_exists(args.model_path, "file")
    _ensure_parent_dir(args.circuit_path)

    # Instantiate the circuit. We use the model filename as a friendly hint.
    model_name_hint = Path(args.model_path).stem
    circuit = _build_default_circuit(model_name_hint)

    # Tell the converter exactly which ONNX file to load (legacy naming).
    circuit.model_file_name = args.model_path
    # Also set modern-ish aliases some subclasses/readers expect.
    circuit.onnx_path = args.model_path
    circuit.model_path = args.model_path

    # Compile: writes circuit + quantized model
    try:
        circuit.base_testing(
            CircuitExecutionConfig(
                run_type=RunType.COMPILE_CIRCUIT,
                circuit_path=args.circuit_path,
                dev_mode=True,
            ),
        )
    except CircuitRunError as e:
        raise CLIError(e) from e
    except Exception as e:
        msg = f"Process execution failed with args '{args.cmd}': {e}"
        raise CLIError(msg) from e
    print(  # noqa: T201
        f"[compile] done → circuit={args.circuit_path},",
    )


def _run_witness(args: argparse.Namespace) -> None:
    _ensure_exists(args.circuit_path, "file")
    _ensure_exists(args.input_path, "file")
    _ensure_parent_dir(args.output_path)
    _ensure_parent_dir(args.witness_path)

    circuit = _build_default_circuit("cli")

    # Witness: adjusts inputs (reshape/scale), computes outputs, writes witness
    try:
        circuit.base_testing(
            CircuitExecutionConfig(
                run_type=RunType.GEN_WITNESS,
                circuit_path=args.circuit_path,
                input_file=args.input_path,
                output_file=args.output_path,
                witness_file=args.witness_path,
            ),
        )
    except CircuitRunError as e:
        raise CLIError(e) from e
    except Exception as e:
        msg = f"Process execution failed with args '{args.cmd}': {e}"
        raise CLIError(msg) from e
    print(  # noqa: T201
        f"[witness] wrote witness → {args.witness_path} and outputs "
        f"→ {args.output_path}",
    )


def _run_prove(args: argparse.Namespace) -> None:
    _ensure_exists(args.circuit_path, "file")
    _ensure_exists(args.witness_path, "file")
    _ensure_parent_dir(args.proof_path)

    circuit = _build_default_circuit("cli")

    # Prove: witness → proof
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
        raise CLIError(e) from e
    except Exception as e:
        msg = f"Process execution failed with args '{args.cmd}': {e}"
        raise CLIError(msg) from e

    print(f"[prove] wrote proof → {args.proof_path}")  # noqa: T201


def _run_verify(args: argparse.Namespace) -> None:
    _ensure_exists(args.circuit_path, "file")
    _ensure_exists(args.input_path, "file")
    _ensure_exists(args.output_path, "file")
    _ensure_exists(args.witness_path, "file")
    _ensure_exists(args.proof_path, "file")

    circuit = _build_default_circuit("cli")

    # Verify: checks proof; some backends also emit verifier artifacts
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
        raise CLIError(e) from e
    except Exception as e:
        msg = f"Process execution failed with args '{args.cmd}': {e}"
        raise CLIError(msg) from e
    print(f"[verify] verification complete for proof → {args.proof_path}")  # noqa: T201


def _append_arg(cmd: list[str], flag: str, val: object | None) -> None:
    """Append '--flag val' only if val is not None/empty (for simple scalars/strings)."""
    if val is None:
        return
    if isinstance(val, str) and not val.strip():
        return
    cmd += [flag, str(val)]


def _run_bench(args: argparse.Namespace) -> None:
    """
    Run benchmarks:
      - depth/breadth: call python.scripts.gen_and_bench (existing sweeps)
      - lenet: run benchmark_runner on the repo's fixed LeNet model/input
    """
    sweep = args.sweep or args.mode
    if not sweep:
        raise CLIError(
            "Please specify --sweep {depth|breadth|lenet} or a positional mode."
        )

    # --- Fixed: LeNet quickstart model ---
    if sweep == "lenet":
        model = Path("python/models/models_onnx/lenet.onnx").resolve()
        inp = Path("python/models/inputs/lenet_input.json").resolve()
        # sensible defaults; allow overrides via --iterations/--results
        iterations = str(args.iterations if args.iterations is not None else 5)
        results = args.results or "benchmarking/lenet_fixed.jsonl"
        Path(results).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "python.scripts.benchmark_runner",
            "--model",
            str(model),
            "--input",
            str(inp),
            "--iterations",
            iterations,
            "--output",
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        rc = subprocess.run(cmd, text=True, env=env).returncode  # noqa: S603
        if rc != 0:
            raise CLIError(f"LeNet benchmark failed with exit code {rc}")
        return  # done

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
        _append_arg(cmd, "--onnx-dir", args.onnx_dir)
        _append_arg(cmd, "--inputs-dir", args.inputs_dir)
        _append_arg(cmd, "--tag", args.tag)
    else:
        _append_arg(cmd, "--depth-min", args.depth_min)
        _append_arg(cmd, "--depth-max", args.depth_max)
        _append_arg(cmd, "--input-hw", args.input_hw)
        _append_arg(cmd, "--arch-depth", args.arch_depth)
        _append_arg(cmd, "--input-hw-list", args.input_hw_list)
        _append_arg(cmd, "--iterations", args.iterations)
        _append_arg(cmd, "--results", args.results)
        _append_arg(cmd, "--onnx-dir", args.onnx_dir)
        _append_arg(cmd, "--inputs-dir", args.inputs_dir)
        _append_arg(cmd, "--pool-cap", args.pool_cap)
        _append_arg(cmd, "--stop-at-hw", args.stop_at_hw)
        _append_arg(cmd, "--conv-out-ch", args.conv_out_ch)
        _append_arg(cmd, "--fc-hidden", args.fc_hidden)
        _append_arg(cmd, "--n-actions", args.n_actions)
        _append_arg(cmd, "--tag", args.tag)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(cmd, text=True, env=env)  # noqa: S603
    if proc.returncode != 0:
        raise CLIError(f"Benchmark command failed with exit code {proc.returncode}")


def main(argv: list[str] | None = None) -> int:
    """
    Entry point for the JSTprove CLI.

    Flow:
      - Parse arguments (top-level options + subcommand).
      - Optionally print the banner.
      - Dispatch to the selected subcommand:
          * compile:  model → circuit + quantized model
          * witness:  inputs → outputs.json + witness.bin
          * prove:    witness → proof.bin
          * verify:   input/output/witness/proof → verification

    Returns:
      0 on success, 1 on handled error.
      Unhandled SystemExit is re-raised to preserve argparse semantics.
    """
    argv = sys.argv[1:] if argv is None else argv

    # --- argparse setup ------------------------------------------------------
    parser = argparse.ArgumentParser(
        prog="jst",
        description="ZKML CLI (compile, witness, prove, verify).",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress the startup banner.",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    # compile
    p_compile = sub.add_parser(
        "compile",
        aliases=["comp"],
        help="Compile a circuit (writes circuit + quantized model + weights).",
        allow_abbrev=False,
    )
    p_compile.add_argument(
        "-m",
        "--model-path",
        required=True,
        help="Path to the original ONNX model.",
    )
    p_compile.add_argument(
        "-c",
        "--circuit-path",
        required=True,
        help="Output path for the compiled circuit (e.g., circuit.txt).",
    )
    # witness
    p_wit = sub.add_parser(
        "witness",
        aliases=["wit"],
        help="Generate witness using a compiled circuit.",
        allow_abbrev=False,
    )
    p_wit.add_argument(
        "-c",
        "--circuit-path",
        required=True,
        help="Path to the compiled circuit.",
    )
    p_wit.add_argument("-i", "--input-path", required=True, help="Path to input JSON.")
    p_wit.add_argument(
        "-o",
        "--output-path",
        required=True,
        help="Path to write model outputs JSON.",
    )
    p_wit.add_argument(
        "-w",
        "--witness-path",
        required=True,
        help="Path to write witness.",
    )

    # prove
    p_prove = sub.add_parser(
        "prove",
        aliases=["prov"],
        help="Generate a proof from a circuit and witness.",
        allow_abbrev=False,
    )
    p_prove.add_argument(
        "-c",
        "--circuit-path",
        required=True,
        help="Path to the compiled circuit.",
    )
    p_prove.add_argument(
        "-w",
        "--witness-path",
        required=True,
        help="Path to an existing witness.",
    )
    p_prove.add_argument(
        "-p",
        "--proof-path",
        required=True,
        help="Path to write proof.",
    )

    # verify
    p_verify = sub.add_parser(
        "verify",
        aliases=["ver"],
        help="Verify a proof.",
        allow_abbrev=False,
    )
    p_verify.add_argument(
        "-c",
        "--circuit-path",
        required=True,
        help="Path to the compiled circuit.",
    )
    p_verify.add_argument(
        "-i",
        "--input-path",
        required=True,
        help="Path to input JSON.",
    )
    p_verify.add_argument(
        "-o",
        "--output-path",
        required=True,
        help="Path to expected outputs JSON.",
    )
    p_verify.add_argument(
        "-w",
        "--witness-path",
        required=True,
        help="Path to witness.",
    )
    p_verify.add_argument("-p", "--proof-path", required=True, help="Path to proof.")

    # bench (model generation + benchmarking)
    p_bench = sub.add_parser(
        "bench",
        help="Generate ONNX models and benchmark JSTprove (depth/breadth sweeps).",
        allow_abbrev=False,
    )

    p_bench.add_argument(
        "mode",
        choices=["depth", "breadth", "lenet"],
        nargs="?",
        help="Shorthand for --sweep when you just want defaults (e.g., 'jst bench depth').",
    )

    p_bench.add_argument(
        "--sweep",
        choices=["depth", "breadth", "lenet"],
        default=None,  # allow omission when using positional mode
        help="depth: vary conv depth; breadth: vary input H=W. If omitted, the positional MODE is used.",
    )

    # depth sweep options
    p_bench.add_argument("--depth-min", type=int, help="(depth) minimum conv depth")
    p_bench.add_argument("--depth-max", type=int, help="(depth) maximum conv depth")
    p_bench.add_argument("--input-hw", type=int, help="(depth) input H=W (e.g., 56)")

    # breadth sweep options
    p_bench.add_argument(
        "--arch-depth", type=int, help="(breadth) conv blocks at fixed topology"
    )
    p_bench.add_argument(
        "--input-hw-list",
        type=str,
        help="(breadth) sizes like '28,56,84,112' or '32:160:32'",
    )

    # common knobs
    p_bench.add_argument(
        "--iterations", type=int, help="E2E loops per model (default 3 in examples)"
    )
    p_bench.add_argument(
        "--results", help="Path to JSONL results (e.g., benchmarking/depth_sweep.jsonl)"
    )
    p_bench.add_argument(
        "--onnx-dir", help="Override ONNX output dir (else chosen by sweep)"
    )
    p_bench.add_argument(
        "--inputs-dir", help="Override inputs output dir (else chosen by sweep)"
    )
    p_bench.add_argument(
        "--pool-cap", type=int, help="Max pool blocks at start (Lenet-like: 2)"
    )
    p_bench.add_argument("--stop-at-hw", type=int, help="Allow pooling while H >= this")
    p_bench.add_argument("--conv-out-ch", type=int, help="Conv output channels")
    p_bench.add_argument("--fc-hidden", type=int, help="Fully-connected hidden size")
    p_bench.add_argument("--n-actions", type=int, help="Classifier outputs (classes)")
    p_bench.add_argument("--tag", help="Optional tag suffix for filenames")

    args = parser.parse_args(argv)

    # --- banner --------------------------------------------------------------
    if not args.no_banner and not os.environ.get("JSTPROVE_NO_BANNER"):
        print_header()

    # --- dispatch ------------------------------------------------------------
    try:
        if args.cmd == "compile":
            # Validate inputs and ensure output folders exist
            _run_compile(args)

        elif args.cmd == "witness":
            # Validate required files; ensure we can write outputs
            _run_witness(args)

        elif args.cmd == "prove":
            # Validate inputs; ensure we can create the proof file
            _run_prove(args)

        elif args.cmd == "verify":
            # Validate all inputs exist including quantized (only to hydrate shapes)
            _run_verify(args)

        elif args.cmd == "bench":
            _run_bench(args)
            from python.scripts import gen_and_bench

    # Preserve argparse/our own explicit exits
    except SystemExit:
        raise
    except CLIError as e:
        print(f"Error: {e}", file=sys.stderr)  # noqa: T201
        return 1
    # Convert unexpected exceptions to a clean non-zero exit
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)  # noqa: T201
        return 1
    else:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
