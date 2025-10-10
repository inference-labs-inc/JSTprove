from __future__ import annotations

import functools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import version as get_version
from pathlib import Path
from time import time
from typing import Any, TypeVar

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib

from python.core import PACKAGE_NAME
from python.core.utils.benchmarking_helpers import (
    end_memory_collection,
    start_memory_collection,
)
from python.core.utils.errors import (
    FileCacheError,
    MissingFileError,
    ProofBackendError,
    ProofSystemNotImplementedError,
)

F = TypeVar("F", bound=Callable[..., Any])
logger = logging.getLogger(__name__)


class RunType(Enum):
    END_TO_END = "end_to_end"
    COMPILE_CIRCUIT = "run_compile_circuit"
    GEN_WITNESS = "run_gen_witness"
    PROVE_WITNESS = "run_prove_witness"
    GEN_VERIFY = "run_gen_verify"


class ZKProofSystems(Enum):
    Expander = "Expander"


class ExpanderMode(Enum):
    PROVE = "prove"
    VERIFY = "verify"


@dataclass
class CircuitExecutionConfig:
    """Configuration for circuit execution operations."""

    run_type: RunType = RunType.END_TO_END
    witness_file: str | None = None
    input_file: str | None = None
    proof_file: str | None = None
    public_path: str | None = None
    verification_key: str | None = None
    circuit_name: str | None = None
    metadata_path: str | None = None
    architecture_path: str | None = None
    w_and_b_path: str | None = None
    output_file: str | None = None
    proof_system: ZKProofSystems = ZKProofSystems.Expander
    circuit_path: str | None = None
    quantized_path: str | None = None
    ecc: bool = True
    dev_mode: bool = False
    write_json: bool = False
    bench: bool = False


def filter_expander_output(stderr: str) -> str:
    """Keep Rust panic + MPI exit summary, drop system stack traces and notes."""
    lines = stderr.splitlines()
    filtered = []
    rust_panic_started = False

    for line in lines:
        # Start capturing Rust panic/assertion
        if "panicked at" in line or "assertion" in line.lower():
            rust_panic_started = True
            filtered.append(line)
            continue

        # Keep lines following Rust panic that are relevant
        if rust_panic_started:
            # Stop at system stack traces or abort messages
            if (
                re.match(r"\[\s*\d+\]", line)
                or "*** Process received signal ***" in line
            ):
                rust_panic_started = False
                continue
            filtered.append(line)

        # Always keep MPI exit summary
        if line.startswith("prterun noticed that process rank"):
            filtered.append(line)

    return "\n".join(filtered)


def extract_rust_error(stderr: str) -> str:
    """
    Extracts the Rust error message from stderr,
    handling both panics and normal error prints.
    """
    lines = stderr.splitlines()
    error_lines = []

    # Case 1: Rust panic
    capture = False
    for line in lines:
        if re.match(r"thread '.*' panicked at", line):
            capture = True
            continue
        if capture:
            if "stack backtrace:" in line.lower():
                break
            error_lines.append(line)
    if error_lines:
        return "\n".join(error_lines).strip()

    # Case 2: Non-panic error (just "Error: ...")
    for line in lines:
        if line.strip().startswith("Error:"):
            return line.strip()

    return ""


# Decorator to compute outputs once and store in temp folder
def compute_and_store_output(func: Callable) -> Callable:
    """Decorator that computes outputs once
    per circuit instance and stores in temp folder.
    Instead of using in-memory cache, uses files in temp folder.

    Args:
        func (Callable): Method that computes outputs to be cached.

    Returns:
        Callable: Wrapped function that reads/writes a caches.
    """

    @functools.wraps(func)
    def wrapper(self: object, *args: tuple, **kwargs: dict) -> object:
        # Define paths for storing outputs in temp folder
        temp_folder = getattr(self, "temp_folder", "temp")
        try:
            Path(temp_folder).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            msg = f"Could not create temp folder {temp_folder}: {e}"
            logger.exception(msg)
            raise FileCacheError(msg) from e

        output_cache_path = Path(temp_folder) / f"{self.name}_output_cache.json"

        # Check if cached output exists
        if output_cache_path.exists():
            msg = f"Loading cached outputs for {self.name} from {output_cache_path}"
            logger.info(msg)
            try:
                with output_cache_path.open() as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                msg = f"Error loading cached output: {e}"
                logger.warning(msg)
                # Continue to compute if loading fails

        # Compute outputs and cache them
        msg = f"Computing outputs for {self.name}..."
        logger.info(msg)
        output = func(self, *args, **kwargs)

        # Store output in temp folder
        try:
            with Path(output_cache_path).open("w") as f:
                json.dump(output, f)
            msg = f"Stored outputs in {output_cache_path}"
            logger.info(msg)
        except OSError as e:
            msg = f"Warning: Could not cache output to file: {e}"
            logger.warning(msg)

        return output

    return wrapper


# Decorator to prepare input/output files
def prepare_io_files(func: Callable) -> Callable:
    """Decorator that prepares input and output files.
    This allows the function to be called independently.

    Args:
        func (Callable): The function requiring prepared file paths.

    Returns:
        Callable: Wrapped function with prepared file paths injected into its arguments.
    """

    @functools.wraps(func)
    def wrapper(
        self: object,
        exec_config: CircuitExecutionConfig,
        *args: tuple,
        **kwargs: dict,
    ) -> object:

        def resolve_folder(
            key: str,
            file_attr: str | None = None,
            default: str = "",
        ) -> str:
            if getattr(exec_config, key, None):
                return getattr(exec_config, key)
            if file_attr and getattr(exec_config, file_attr, None):
                return str(Path(getattr(exec_config, file_attr)).parent)
            return getattr(self, key, default)

        input_folder = resolve_folder(
            "input_folder",
            "input_file",
            default="python/models/inputs",
        )
        output_folder = resolve_folder(
            "output_folder",
            "output_file",
            default="python/models/output",
        )
        proof_folder = resolve_folder(
            "proof_folder",
            "proof_file",
            default="python/models/proofs",
        )
        quantized_model_folder = resolve_folder(
            "quantized_folder",
            "quantized_path",
            default="python/models/quantized_model_folder",
        )
        weights_folder = resolve_folder(
            "weights_folder",
            default="python/models/weights",
        )
        circuit_folder = resolve_folder("circuit_folder", default="python/models/")

        proof_system = exec_config.proof_system or getattr(
            self,
            "proof_system",
            ZKProofSystems.Expander,
        )

        files = get_files(
            self.name,
            proof_system,
            {
                "input": input_folder,
                "proof": proof_folder,
                "circuit": circuit_folder,
                "weights": weights_folder,
                "output": output_folder,
                "quantized_model": quantized_model_folder,
            },
        )
        # Fill in any missing fields in exec_config with defaults from `files`
        exec_config.witness_file = exec_config.witness_file or files["witness_file"]
        exec_config.input_file = exec_config.input_file or files["input_file"]
        exec_config.proof_file = exec_config.proof_file or files["proof_path"]
        exec_config.public_path = exec_config.public_path or files["public_path"]
        exec_config.circuit_name = exec_config.circuit_name or files["circuit_name"]
        exec_config.metadata_path = exec_config.metadata_path or files["metadata_path"]
        exec_config.architecture_path = (
            exec_config.architecture_path or files["architecture_path"]
        )
        exec_config.w_and_b_path = exec_config.w_and_b_path or files["w_and_b_path"]
        exec_config.output_file = exec_config.output_file or files["output_file"]

        if exec_config.circuit_path:
            circuit_dir = Path(exec_config.circuit_path).parent
            name = Path(exec_config.circuit_path).stem
            exec_config.quantized_path = str(
                circuit_dir / f"{name}_quantized_model.onnx",
            )
            exec_config.metadata_path = str(
                circuit_dir / f"{name}_metadata.json",
            )
            exec_config.architecture_path = str(
                circuit_dir / f"{name}_architecture.json",
            )
            exec_config.w_and_b_path = str(
                circuit_dir / f"{name}_wandb.json",
            )
        else:
            exec_config.quantized_path = None

        # Store paths and data for use in the decorated function
        self._file_info = {
            "witness_file": exec_config.witness_file,
            "input_file": exec_config.input_file,
            "proof_file": exec_config.proof_file,
            "public_path": exec_config.public_path,
            "circuit_name": exec_config.circuit_name,
            "metadata_path": exec_config.metadata_path,
            "architecture_path": exec_config.architecture_path,
            "w_and_b_path": exec_config.w_and_b_path,
            "output_file": exec_config.output_file,
            "inputs": exec_config.input_file,
            "weights": exec_config.w_and_b_path,  # Changed to w_and_b_path
            "outputs": exec_config.output_file,
            "output": exec_config.output_file,
            "proof_system": exec_config.proof_system or proof_system,
            "model_path": getattr(exec_config, "model_path", None),
            "quantized_model_path": exec_config.quantized_path,
        }

        # Call the original function with the populated exec_config
        return func(self, exec_config, *args, **kwargs)

    return wrapper


def ensure_parent_dir(path: str) -> None:
    """Create parent directories for a given path if they don't exist.

    Args:
        path (str): Path for which to create parent directories.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def run_subprocess(cmd: list[str]) -> None:
    """Run a subprocess command and raise RuntimeError if it fails.

    Args:
        cmd (list[str]): The command to execute.

    Raises:
        RuntimeError: If the command exits with a non-zero return code.
    """
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(cmd, text=True, env=env, check=False)  # noqa: S603
    if proc.returncode != 0:
        msg = f"Command failed with exit code {proc.returncode}"
        raise RuntimeError(msg)


def to_json(inputs: dict[str, Any], path: str) -> None:
    """Write data to a JSON file.

    Args:
        inputs (dict[str, Any]): Data to be serialized.
        path (str): Path where the JSON file will be written.
    """
    ensure_parent_dir(path)
    with Path(path).open("w") as outfile:
        json.dump(inputs, outfile)


def read_from_json(public_path: str) -> dict[str, Any]:
    """Read data from a JSON file.

    Args:
        public_path (str): Path to the JSON file to read.

    Returns:
        dict[str, Any]: The data read from the JSON file.
    """
    with Path(public_path).open() as json_data:
        return json.load(json_data)


def run_cargo_command(
    binary_name: str,
    command_type: str,
    args: dict[str, str] | None = None,
    *,
    dev_mode: bool = False,
    bench: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a cargo command with the correct format based on the command type.

    Args:
        binary_name (str): Name of the Cargo binary.
        command_type (str): Command type (e.g., 'run_proof', 'run_compile_circuit').
        args (dict[str, str], optional): dictionary of CLI arguments. Defaults to None.
        circuit_path (str, optional):
            Path to the circuit file, used for copying the binary.
        dev_mode (bool, optional):
            If True, run with `cargo run --release` instead of prebuilt binary.
            Defaults to False.
        bench (bool, optional):
            If True, measure execution time and memory usage. Defaults to False.

    Raises:
        subprocess.CalledProcessError: If the Cargo command fails.

    Returns:
        subprocess.CompletedProcess[str]: Exit message from the subprocess.
    """
    try:
        version = get_version(PACKAGE_NAME)
        binary_name = binary_name + f"_{version}".replace(".", "-")
    except Exception:
        try:
            pyproject = tomllib.loads(Path("pyproject.toml").read_text())
            version = pyproject["project"]["version"]
            binary_name = binary_name + f"_{version}".replace(".", "-")
        except (FileNotFoundError, KeyError, tomllib.TOMLDecodeError):
            pass

    binary_path = None
    possible_paths = [
        f"./target/release/{binary_name}",
        Path(__file__).parent.parent / "binaries" / binary_name,
        Path(sys.prefix) / "bin" / binary_name,
    ]

    for path in possible_paths:
        if Path(path).exists():
            binary_path = str(path)
            break

    if not binary_path:
        binary_path = f"./target/release/{binary_name}"
    cmd = _build_command(
        binary_path=binary_path,
        command_type=command_type,
        args=args,
        dev_mode=dev_mode,
        binary_name=binary_name,
    )
    env = os.environ.copy()
    env["RUST_BACKTRACE"] = "1"

    msg = f"Running cargo command: {' '.join(cmd)}"
    print(msg)  # noqa: T201
    logger.info(msg)

    try:
        result = _run_subprocess_with_bench(
            cmd=cmd,
            env=env,
            bench=bench,
            binary_name=binary_name,
        )
        _handle_result(result=result, cmd=cmd)
    except OSError as e:
        msg = f"Failed to execute proof backend command '{cmd}': {e}"
        logger.exception(msg)
        raise ProofBackendError(msg) from e
    except subprocess.CalledProcessError as e:
        msg = f"Cargo command failed (return code {e.returncode}): {e.stderr}"
        logger.exception(msg)
        rust_error = extract_rust_error(e.stderr)
        msg = f"Rust backend error '{rust_error}'"
        raise ProofBackendError(msg, cmd) from e
    else:
        return result


def _build_command(
    binary_path: str,
    command_type: str,
    args: dict[str, str] | None,
    *,
    dev_mode: bool,
    binary_name: str,
) -> list[str]:
    """Build the command list for subprocess."""
    cmd = (
        ["cargo", "run", "--bin", binary_name, "--release"]
        if dev_mode or not Path(binary_path).exists()
        # dev_mode indicates that we want a recompile, this happens with compile
        # or if there is no executable already created, then we create a new one
        else [binary_path]
    )
    cmd.append(command_type)
    if args:
        for key, value in args.items():
            cmd.append(f"-{key}")
            if not (isinstance(value, bool) and value):
                cmd.append(str(value))
    return cmd


def _run_subprocess_with_bench(
    cmd: list[str],
    env: dict[str, str],
    binary_name: str,
    *,
    bench: bool,
) -> subprocess.CompletedProcess[str]:
    """Run the subprocess with optional benchmarking."""
    if bench:
        stop_event, monitor_thread, monitor_results = start_memory_collection(
            binary_name,
        )
    start_time = time()
    result = subprocess.run(  # noqa: S603
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    end_time = time()
    print("\n--- BENCHMARK RESULTS ---")  # noqa: T201
    print(f"Rust time taken: {end_time - start_time:.4f} seconds")  # noqa: T201
    msg = f"Rust command completed in {end_time - start_time:.4f} seconds"
    logger.info(msg)

    if bench:
        memory = end_memory_collection(stop_event, monitor_thread, monitor_results)
        msg = f"Rust subprocess memory: {memory['total']:.2f} MB"
        print(msg)  # noqa: T201
        logger.info(msg)

    print(result.stdout)  # noqa: T201
    logger.info(result.stdout)
    return result


def _handle_result(result: subprocess.CompletedProcess[str], cmd: list[str]) -> None:
    """Handle the subprocess result and raise errors if needed."""
    if result.returncode != 0:
        msg = f"Proving Backend failed (code {result.returncode}):\n{result.stderr}"
        logger.error(msg)
        msg = (
            f"Proving Backend command '{' '.join(cmd)}'"
            f" failed with code {result.returncode}:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        raise ProofBackendError(msg)


def _copy_binary_if_needed(
    binary_name: str,
    binary_path: str,
    *,
    dev_mode: bool,
) -> None:
    """Copy the binary if conditions are met."""
    src = f"./target/release/{binary_name}"
    if Path(src).exists() and (str(src) != str(binary_path)) and dev_mode:
        shutil.copy(src, binary_path)


def get_expander_file_paths(circuit_name: str) -> dict[str, str]:
    """Generate standard file paths for an Expander circuit.

    Args:
        circuit_name (str): The base name of the circuit.

    Returns:
        dict[str, str]:
            dictionary containing file paths with keys:
                circuit_file, witness_file, proof_file
    """
    return {
        "circuit_file": f"{circuit_name}_circuit.txt",
        "witness_file": f"{circuit_name}_witness.txt",
        "proof_file": f"{circuit_name}_proof.txt",
    }


def run_expander_raw(  # noqa: PLR0913, PLR0912, C901
    mode: ExpanderMode,
    circuit_file: str,
    witness_file: str,
    proof_file: str,
    pcs_type: str = "Hyrax",
    *,
    bench: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run the Expander executable directly using Cargo.

    Args:
        mode (ExpanderMode): Operation mode (PROVE or VERIFY).
        circuit_file (str): Path to the circuit definition file.
        witness_file (str): Path to the witness file.
        proof_file (str): Path to the proof file (input for verification,
                          output for proving).
        pcs_type (str, optional):
            Polynomial commitment scheme type ("Hyrax" or "Raw").
            Defaults to "Hyrax".
        bench (bool, optional):
            If True, collect runtime and memory benchmark data. Defaults to False.

    Returns:
        subprocess.CompletedProcess[str]: Exit message from the subprocess.
    """
    for file_path, label in [
        (circuit_file, "circuit_file"),
        (witness_file, "witness_file"),
        # For VERIFY mode, the proof_file is an input, not just an output
        (proof_file, "proof_file") if mode == ExpanderMode.VERIFY else (None, None),
    ]:
        if file_path and not Path(file_path).exists():
            msg = f"Missing file required for {label}"
            raise MissingFileError(msg, file_path)

    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C target-cpu=native"
    time_measure = "/usr/bin/time"

    expander_binary_path = None
    possible_paths = [
        "./Expander/target/release/expander-exec",
        Path(__file__).parent.parent / "binaries" / "expander-exec",
        Path(sys.prefix) / "bin" / "expander-exec",
    ]

    for path in possible_paths:
        if Path(path).exists():
            expander_binary_path = str(path)
            break

    if expander_binary_path:
        args = [
            time_measure,
            expander_binary_path,
            "-p",
            pcs_type,
        ]
    else:
        args = [
            time_measure,
            "mpiexec",
            "-n",
            "1",
            "cargo",
            "run",
            "--manifest-path",
            "Expander/Cargo.toml",
            "--bin",
            "expander-exec",
            "--release",
            "--",
            "-p",
            pcs_type,
        ]
    if mode == ExpanderMode.PROVE:
        args.append(mode.value)
        proof_command = "-o"
    else:
        args.append(mode.value)
        proof_command = "-i"

    args.extend(["-c", circuit_file])
    args.extend(["-w", witness_file])
    args.extend([proof_command, proof_file])

    try:
        if bench:
            stop_event, monitor_thread, monitor_results = start_memory_collection(
                "expander-exec",
            )
        start_time = time()
        result = subprocess.run(  # noqa: S603
            args,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        end_time = time()

        print("\n--- BENCHMARK RESULTS ---")  # noqa: T201
        print(f"Rust time taken: {end_time - start_time:.4f} seconds")  # noqa: T201

        if bench:
            memory = end_memory_collection(stop_event, monitor_thread, monitor_results)
            print(f"Rust subprocess memory: {memory['total']:.2f} MB")  # noqa: T201

        if result.returncode != 0:
            clean_stderr = filter_expander_output(result.stderr)
            msg = f"Expander {mode.value} failed:\n{clean_stderr}"
            logger.warning(msg)
            msg = f"Expander {mode.value} failed"
            raise ProofBackendError(
                msg,
                command=args,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=clean_stderr,
            )

        print(  # noqa: T201
            f"✅ expander-exec {mode.value} succeeded:\n{result.stdout}",
        )

        print(f"Time taken: {end_time - start_time:.4f} seconds")  # noqa: T201
    except OSError as e:
        msg = f"Failed to execute Expander {mode.value}: {e}"
        logger.exception(msg)
        raise ProofBackendError(
            msg,
            command=args,
        ) from e
    else:
        return result


def compile_circuit(  # noqa: PLR0913
    circuit_name: str,
    circuit_path: str,
    metadata_path: str,
    architecture_path: str,
    w_and_b_path: str,
    proof_system: ZKProofSystems = ZKProofSystems.Expander,
    *,
    dev_mode: bool = True,
    bench: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Compile a model into zk circuit

    Args:
        circuit_name (str): Name of the circuit.
        circuit_path (str):  Path to the circuit source file.
        proof_system (ZKProofSystems, optional):
            Proof system to use. Defaults to ZKProofSystems.Expander.
        dev_mode (bool, optional):
            If True, recompiles the rust binary (run in development mode).
            Defaults to True.
        bench (bool, optional):
            Whether or not to run benchmarking metrics. Defaults to False.

    Raises:
        NotImplementedError: If proof system is not supported.
    Returns:
        subprocess.CompletedProcess[str]: Exit message from the subprocess.
    """
    if proof_system == ZKProofSystems.Expander:
        # Extract the binary name from the circuit path
        binary_name = Path(circuit_name).name

        # Prepare arguments
        args = {
            "n": circuit_name,
            "c": circuit_path,
            "m": metadata_path,
            "a": architecture_path,
            "b": w_and_b_path,
        }
        # Run the command
        try:
            return run_cargo_command(
                binary_name=binary_name,
                command_type=RunType.COMPILE_CIRCUIT.value,
                args=args,
                dev_mode=dev_mode,
                bench=bench,
            )
        except ProofBackendError as e:
            warning = f"Warning: Compile operation failed: {e}."
            warning2 = f" Using binary: {binary_name}"
            logger.warning(warning)
            logger.warning(warning2)
            raise

    else:
        msg = f"Proof system {proof_system} not implemented"
        raise ProofSystemNotImplementedError(msg)


def generate_witness(  # noqa: PLR0913
    circuit_name: str,
    circuit_path: str,
    witness_file: str,
    input_file: str,
    output_file: str,
    metadata_path: str,
    proof_system: ZKProofSystems = ZKProofSystems.Expander,
    *,
    dev_mode: bool = False,
    bench: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Generate a witness file for a circuit.

    Args:
        circuit_name (str): Name of the circuit.
        circuit_path (str): Path to the circuit definition.
        witness_file (str): Path to the output witness file.
        input_file (str): Path to the input JSON file with private inputs.
        output_file (str): Path to the output JSON file with computed outputs.
        proof_system (ZKProofSystems, optional): Proof system to use.
            Defaults to ZKProofSystems.Expander.
        dev_mode (bool, optional):
            If True, recompiles the rust binary (run in development mode).
            Defaults to False.
        bench (bool, optional):
            If True, enable benchmarking. Defaults to False.

    Raises:
        NotImplementedError: If proof system is not supported.

    Returns:
        subprocess.CompletedProcess[str]: Exit message from the subprocess.
    """
    if proof_system == ZKProofSystems.Expander:
        # Extract the binary name from the circuit path
        binary_name = Path(circuit_name).name

        # Prepare arguments
        args = {
            "n": circuit_name,
            "c": circuit_path,
            "i": input_file,
            "o": output_file,
            "w": witness_file,
            "m": metadata_path,
        }
        # Run the command
        try:
            return run_cargo_command(
                binary_name=binary_name,
                command_type=RunType.GEN_WITNESS.value,
                args=args,
                dev_mode=dev_mode,
                bench=bench,
            )
        except ProofBackendError as e:
            warning = f"Warning: Witness generation failed: {e}"
            logger.warning(warning)
            raise
    else:
        msg = f"Proof system {proof_system} not implemented"
        raise ProofSystemNotImplementedError(msg)


def generate_proof(  # noqa: PLR0913
    circuit_name: str,
    circuit_path: str,
    witness_file: str,
    proof_file: str,
    metadata_path: str,
    proof_system: ZKProofSystems = ZKProofSystems.Expander,
    *,
    dev_mode: bool = False,
    ecc: bool = True,
    bench: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Generate proof for the witness.

    Args:
        circuit_name (str): Name of the circuit.
        circuit_path (str): Path to the circuit definition.
        witness_file (str): Path to the witness file.
        proof_file (str): Path to the output proof file.
        proof_system (ZKProofSystems, optional): Proof system to use.
            Defaults to ZKProofSystems.Expander.
        dev_mode (bool, optional):
            If True, recompiles the rust binary (run in development mode).
            Defaults to False.
        ecc (bool, optional):
            If true, run proof using ECC api, otherwise run directly through Expander.
            Defaults to True.
        bench (bool, optional):
            If True, enable benchmarking. Defaults to False.

    Raises:
        NotImplementedError: If proof system is not supported.

    Returns:
        subprocess.CompletedProcess[str]: Exit message from the subprocess.
    """
    if proof_system == ZKProofSystems.Expander:
        if ecc:
            # Extract the binary name from the circuit path
            binary_name = Path(circuit_name).name

            # Prepare arguments
            args = {
                "n": circuit_name,
                "c": circuit_path,
                "w": witness_file,
                "p": proof_file,
                "m": metadata_path,
            }

            # Run the command
            try:
                return run_cargo_command(
                    binary_name=binary_name,
                    command_type=RunType.PROVE_WITNESS.value,
                    args=args,
                    dev_mode=dev_mode,
                    bench=bench,
                )
            except ProofBackendError as e:
                warning = f"Warning: Proof generation failed: {e}"
                logger.warning(warning)
                raise
        else:
            return run_expander_raw(
                mode=ExpanderMode.PROVE,
                circuit_file=circuit_path,
                witness_file=witness_file,
                proof_file=proof_file,
                bench=bench,
            )
    else:
        msg = f"Proof system {proof_system} not implemented"
        raise ProofSystemNotImplementedError(msg)


def generate_verification(  # noqa: PLR0913
    circuit_name: str,
    circuit_path: str,
    input_file: str,
    output_file: str,
    witness_file: str,
    proof_file: str,
    metadata_path: str,
    proof_system: ZKProofSystems = ZKProofSystems.Expander,
    *,
    dev_mode: bool = False,
    ecc: bool = True,
    bench: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Verify a given proof.

    Args:
        circuit_name (str): Name of the circuit.
        circuit_path (str): Path to the circuit definition.
        input_file (str): Path to the input JSON file with public inputs.
        output_file (str): Path to the output JSON file with expected outputs.
        witness_file (str): Path to the witness file.
        proof_file (str): Path to the output proof file.
        proof_system (ZKProofSystems, optional): Proof system to use.
            Defaults to ZKProofSystems.Expander.
        dev_mode (bool, optional):
            If True, recompiles the rust binary (run in development mode).
            Defaults to False.
        ecc (bool, optional):
            If true, run proof using ECC api, otherwise run directly through Expander.
            Defaults to True.
        bench (bool, optional):
            If True, enable benchmarking. Defaults to False.

    Raises:
        NotImplementedError: If proof system is not supported.

    Returns:
        subprocess.CompletedProcess[str]: Exit message from the subprocess.
    """
    if proof_system == ZKProofSystems.Expander:
        if ecc:
            # Extract the binary name from the circuit path
            binary_name = Path(circuit_name).name

            # Prepare arguments
            args = {
                "n": circuit_name,
                "c": circuit_path,
                "i": input_file,
                "o": output_file,
                "w": witness_file,
                "p": proof_file,
                "m": metadata_path,
            }
            # Run the command
            try:
                return run_cargo_command(
                    binary_name=binary_name,
                    command_type=RunType.GEN_VERIFY.value,
                    args=args,
                    dev_mode=dev_mode,
                    bench=bench,
                )
            except ProofBackendError as e:
                warning = f"Warning: Verification generation failed: {e}"
                logger.warning(warning)
                raise
        else:
            return run_expander_raw(
                mode=ExpanderMode.VERIFY,
                circuit_file=circuit_path,
                witness_file=witness_file,
                proof_file=proof_file,
                bench=bench,
            )
    else:
        msg = f"Proof system {proof_system} not implemented"
        raise ProofSystemNotImplementedError(msg)


def run_end_to_end(  # noqa: PLR0913
    circuit_name: str,
    circuit_path: str,
    input_file: str,
    output_file: str,
    proof_system: ZKProofSystems = ZKProofSystems.Expander,
    *,
    demo: bool = False,
    dev_mode: bool = False,
    ecc: bool = True,
) -> int:
    """Run the full pipeline for proving and verifying a circuit.

    Steps:
        1. Compile the circuit.
        2. Generate a witness from inputs.
        3. Produce a proof from the witness.
        4. Verify the proof against inputs and outputs.

    Args:
        circuit_name (str): Name of the circuit.
        circuit_path (str): Path to the circuit definition.
        input_file (str): Path to the input JSON file with public inputs.
        output_file (str): Path to the output JSON file with expected outputs.
        proof_system (ZKProofSystems, optional):
            Proof system to use. Defaults to ZKProofSystems.Expander.
        demo (bool, optional):
            Run Demo mode, which limits prints, to clean only. Defaults to False.
        dev_mode (bool, optional):
            If True, recompiles the rust binary (run in development mode).
            Defaults to False.
        ecc (bool, optional):
            If true, run proof using ECC api, otherwise run directly through Expander.
            Defaults to True.

    Raises:
        NotImplementedError: If proof system is not supported.

    Returns:
        int: Exit code from the verification step (0 = success, non-zero = failure).
    """
    _ = demo
    if proof_system == ZKProofSystems.Expander:
        path = Path(circuit_path)
        base = str(path.with_suffix(""))  # filename without extension
        ext = path.suffix

        witness_file = f"{base}_witness{ext}"
        proof_file = f"{base}_proof.bin"
        compile_circuit(
            circuit_name,
            circuit_path,
            f"{base}_metadata.json",
            f"{base}_architecture.json",
            f"{base}_wandb.json",
            proof_system,
            dev_mode,
        )
        generate_witness(
            circuit_name,
            circuit_path,
            witness_file,
            input_file,
            output_file,
            f"{base}_metadata.json",
            proof_system,
            dev_mode,
        )
        generate_proof(
            circuit_name,
            circuit_path,
            witness_file,
            proof_file,
            f"{base}_metadata.json",
            proof_system,
            dev_mode,
            ecc,
        )
        return generate_verification(
            circuit_name,
            circuit_path,
            input_file,
            output_file,
            witness_file,
            proof_file,
            f"{base}_metadata.json",
            proof_system,
            dev_mode,
            ecc,
        )
    msg = f"Proof system {proof_system} not implemented"
    raise ProofSystemNotImplementedError(msg)


def get_files(
    name: str,
    proof_system: ZKProofSystems,
    folders: dict[str, str],
) -> dict[str, str]:
    """
    Generate file paths ensuring folders exist.

    Args:
        name (str): The base name for all generated files.
        proof_system (ZKProofSystems): The ZK proof system being used.
        folders (dict[str, str]):
            dictionary containing required folder paths with keys like:
            'input', 'proof', 'temp', 'circuit', 'weights', 'output', 'quantized_model'.

    Raises:
        NotImplementedError: If not implemented proof system is tried

    Returns:
        dict[str, str]: A dictionary mapping descriptive keys to file paths.
    """
    # Common file paths
    paths = {
        "input_file": str(Path(folders["input"]) / f"{name}_input.json"),
        "public_path": str(Path(folders["proof"]) / f"{name}_public.json"),
        "metadata_path": str(Path(folders["weights"]) / f"{name}_metadata.json"),
        "architecture_path": str(
            Path(folders["weights"]) / f"{name}_architecture.json",
        ),
        "w_and_b_path": str(Path(folders["weights"]) / f"{name}_w_and_b.json"),
        "output_file": str(Path(folders["output"]) / f"{name}_output.json"),
    }

    # Proof-system-specific files
    if proof_system == ZKProofSystems.Expander:
        paths.update(
            {
                "circuit_name": name,
                "witness_file": str(Path(folders["input"]) / f"{name}_witness.txt"),
                "proof_path": str(Path(folders["proof"]) / f"{name}_proof.bin"),
            },
        )
    else:
        msg = f"Proof system {proof_system} not implemented"
        raise ProofSystemNotImplementedError(msg)

    return paths
