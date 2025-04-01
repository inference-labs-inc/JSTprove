#!/usr/bin/env python3
"""
CLI for running circuit operations. Dynamically loads circuit modules, resolves file paths
using fuzzy matching, and supports listing available circuits.
"""

import argparse
import difflib
import importlib
import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

from python_testing.circuit_components.circuit_helpers import RunType
from python_testing.utils import helper_functions  # For monkey-patching run_expander_exec

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set PROJECT_ROOT since cli.py is in the project root (GravyTesting-Internal)
PROJECT_ROOT = Path(__file__).resolve().parent

# Cache for JSON files to avoid repeated directory scans
_JSON_FILES_CACHE: Optional[List[Path]] = None

# Default directories to search for the expander-exec binary if no exec_path is provided.
DEFAULT_EXEC_SEARCH_DIRS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "bin",
    PROJECT_ROOT / "target" / "release",
]

def _get_all_json_files() -> List[Path]:
    global _JSON_FILES_CACHE
    if _JSON_FILES_CACHE is None:
        _JSON_FILES_CACHE = list(PROJECT_ROOT.rglob("*.json"))
        print(f"[INFO] Cached JSON files from {PROJECT_ROOT}")
    return _JSON_FILES_CACHE

def find_file(filename: str, default_path: Optional[Path] = None) -> Path:
    """
    Finds a JSON file in the project root using exact or fuzzy matching.
    """
    if not filename.endswith(".json"):
        filename += ".json"

    if default_path:
        candidate = PROJECT_ROOT / default_path
        if candidate.is_file():
            print(f"[INFO] Found file at provided default path: {candidate}")
            return candidate

    all_json_files = _get_all_json_files()
    for path in all_json_files:
        if path.name == filename:
            print(f"[INFO] Found file: {path}")
            return path

    file_names = [p.name for p in all_json_files]
    close_matches = difflib.get_close_matches(filename, file_names, n=1, cutoff=0.6)
    if close_matches:
        fuzzy_match = close_matches[0]
        for path in all_json_files:
            if path.name == fuzzy_match:
                print(f"[WARNING] Could not find '{filename}', using close match '{fuzzy_match}'")
                return path

    raise FileNotFoundError(f"Could not find file '{filename}' (or a close match) under {PROJECT_ROOT}.")

def try_import(module_path: str, class_name: str):
    """
    Helper function: Tries to import module_path and return the attribute 'class_name'.
    Returns None on failure.
    """
    try:
        module = importlib.import_module(module_path)
        attr = getattr(module, class_name)
        print(f"[INFO] Successfully imported '{class_name}' from '{module_path}'")
        return attr
    except (ModuleNotFoundError, AttributeError):
        print(f"[WARNING] Failed to import '{class_name}' from '{module_path}'")
        return None

def load_circuit(circuit_name: str, class_name: str = "SimpleCircuit", search_path: Optional[str] = None):
    """
    Dynamically loads a circuit module and returns an instance of the specified circuit class.
    It tries the following in order:
      - python_testing.<search_path> (if provided)
      - python_testing.circuit_models
      - python_testing.circuit_components
    """
    base_paths = ["python_testing.circuit_models", "python_testing.circuit_components"]
    if search_path:
        # Prepend the user-specified search path
        base_paths.insert(0, f"python_testing.{search_path}")
        print(f"[INFO] Using custom search path: python_testing.{search_path}")

    for base in base_paths:
        module_path = f"{base}.{circuit_name}"
        print(f"[INFO] Attempting to load circuit from '{module_path}'")
        circuit_class = try_import(module_path, class_name)
        if circuit_class:
            print(f"[INFO] Loaded circuit class '{class_name}' from '{module_path}'")
            return circuit_class()
    raise ValueError(
        f"Could not load circuit '{circuit_name}' with class '{class_name}' from any known location."
    )

def resolve_file_paths(
    circuit_name: str, 
    input_override: Optional[str], 
    output_override: Optional[str], 
    pattern: Optional[str]
) -> Tuple[str, str]:
    """
    Resolves input and output file paths based on provided overrides or default patterns.
    """
    if pattern:
        input_filename = pattern.format(circuit=circuit_name)
        output_filename = input_filename.replace("input", "output") if "input" in input_filename else f"{circuit_name}_output.json"
    else:
        input_filename = f"{circuit_name}_input.json"
        output_filename = f"{circuit_name}_output.json"

    print(f"[INFO] Resolving file paths using input filename: {input_filename} and output filename: {output_filename}")
    input_path = str(find_file(input_override) if input_override else find_file(input_filename, Path("inputs") / input_filename))
    output_path = str(find_file(output_override) if output_override else find_file(output_filename, Path("output") / output_filename))
    print(f"[INFO] Resolved input path: {input_path}")
    print(f"[INFO] Resolved output path: {output_path}")
    return input_path, output_path

def is_circuit_file(path: Path) -> bool:
    """
    Checks whether a file contains a class that inherits from ZKModel or Circuit.
    """
    try:
        content = path.read_text()
    except Exception:
        return False
    return "class " in content and ( "(ZKModel):" in content or "(Circuit):" in content )

def list_available_circuits(search_path: Optional[str] = None):
    """
    Lists all available circuit files by searching for Python files that define a class inheriting
    from ZKModel or Circuit. By default, it searches in:
      - python_testing/circuit_components
      - python_testing/circuit_models
    The user can override this search directory using the --circuit_search_path flag.
    """
    print("[INFO] Listing available circuits...")
    if search_path:
        search_paths = [PROJECT_ROOT / search_path]
        print(f"[INFO] Using custom circuit search path: {PROJECT_ROOT / search_path}")
    else:
        search_paths = [
            PROJECT_ROOT / "python_testing" / "circuit_components",
            PROJECT_ROOT / "python_testing" / "circuit_models"
        ]
    circuit_files = {
        str(p.relative_to(PROJECT_ROOT))
        for base in search_paths if base.exists()
        for p in base.rglob("*.py")
        if is_circuit_file(p)
    }
    if circuit_files:
        print("Available circuit files:")
        for file in sorted(circuit_files):
            print(f"  {file}")
    else:
        print("No circuit files found.")

def resolve_exec_path(exec_input: Optional[str], circuit_name: str) -> Optional[str]:
    """
    Resolve the expander-exec binary path based on user input and default directories.
    If a filename isn't specified (or a default filename isn't found), also search for binaries whose name is of the form {circuit_name}.
    Returns the full path to the binary if found, or None if not found.
    """
    default_filename = "expander-exec"
    
    def search_in_directory(directory: Path, target_name: str) -> Optional[Path]:
        print(f"[INFO] Searching for '{target_name}' under directory: {directory}")
        if not directory.is_dir():
            print(f"[INFO] Directory {directory} does not exist or is not a directory.")
            return None
        # Look for an exact match.
        candidate = directory / target_name
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            print(f"[INFO] Found executable: {candidate}")
            return candidate
        # Fuzzy search in the directory.
        files = [f for f in directory.iterdir() if f.is_file()]
        file_names = [f.name for f in files]
        close_matches = difflib.get_close_matches(target_name, file_names, n=1, cutoff=0.6)
        if close_matches:
            fuzzy_candidate = directory / close_matches[0]
            if fuzzy_candidate.is_file() and fuzzy_candidate.stat().st_mode & 0o111:
                print(f"[INFO] Fuzzy match: found executable '{fuzzy_candidate}' for target '{target_name}'")
                return fuzzy_candidate
        print(f"[INFO] No match found for '{target_name}' in directory {directory}")
        return None

    # Case 1: If exec_input is provided.
    if exec_input:
        path_candidate = Path(exec_input)
        if path_candidate.is_absolute() or str(path_candidate).startswith("."):
            if path_candidate.is_file() and path_candidate.stat().st_mode & 0o111:
                print(f"[INFO] Using provided full path to executable: {path_candidate}")
                return str(path_candidate)
            elif path_candidate.is_dir():
                print(f"[INFO] Provided path is a directory. Searching inside: {path_candidate}")
                result = search_in_directory(path_candidate, default_filename)
                if result:
                    return str(result)
            else:
                print(f"[INFO] Provided exec_path '{exec_input}' is not an existing file; treating it as a filename.")
        else:
            print(f"[INFO] Provided exec_path '{exec_input}' is treated as filename. Searching in default directories.")
        for directory in DEFAULT_EXEC_SEARCH_DIRS:
            result = search_in_directory(directory, exec_input)
            if result:
                return str(result)
    
    # Case 2: No exec_input provided.
    print("[INFO] No exec_path provided. Searching for default executables in default directories.")
    # Try both the default filename and a binary named after the circuit.
    target_names = [default_filename, circuit_name]
    for target in target_names:
        print(f"[INFO] Searching for '{target}' in default directories.")
        for directory in DEFAULT_EXEC_SEARCH_DIRS:
            result = search_in_directory(directory, target)
            if result:
                return str(result)
    
    print("[INFO] No suitable executable binary found in specified or default directories.")
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Run circuit operations.")
    # Operation flags.
    parser.add_argument("--compile", action="store_true", help="Compile the circuit.")
    parser.add_argument("--gen_witness", action="store_true", help="Generate witness for the circuit.")
    parser.add_argument("--prove", action="store_true", help="Generate witness and proof.")
    parser.add_argument("--verify", action="store_true", help="Run verification.")
    parser.add_argument("--end_to_end", action="store_true", help="Run end-to-end circuit testing.")
    parser.add_argument("--all", action="store_true", help="Run all stages (compile_circuit, gen_witness, prove_witness, gen_verify).")
    parser.add_argument("--fresh_compile", action="store_true", help="Force fresh compilation of the circuit (sets dev_mode=True).")
    parser.add_argument("--ecc", action="store_true", help="Use ExpanderCompilerCollection (cargo) instead of expander-exec.")
    parser.add_argument("--exec_path", type=str, help="Path, filename, or directory for expander-exec binary.")

    # Listing and search path flag (used for both listing and dynamic loading).
    parser.add_argument("--list_circuits", action="store_true", help="List all available circuit files.")
    parser.add_argument("--circuit_search_path", type=str, help="Directory to search for circuits (relative to project root).")
    
    # File overrides and pattern.
    parser.add_argument("--circuit_path", type=str, help="Path to the circuit file (Rust or otherwise)")
    parser.add_argument("--input", type=str, help="Path to the input JSON file.")
    parser.add_argument("--output", type=str, help="Path to the output JSON file.")
    parser.add_argument("--witness", type=str, help="Optional path to witness file")
    parser.add_argument("--proof", type=str, help="Optional path to proof file")
    parser.add_argument("--pattern", type=str, help="Optional pattern for input/output filenames with '{circuit}' placeholder.")
    
    # Circuit module and class specification.
    parser.add_argument("--circuit", type=str, help="Name of the circuit module (under default packages).")
    parser.add_argument("--class", dest="class_name", type=str, default="SimpleCircuit",
                        help="Name of the circuit class to load. Defaults to 'SimpleCircuit'.")
    args = parser.parse_args()
    print("[INFO] Arguments parsed successfully.")
    return args

def get_run_operations(args) -> List[RunType]:
    """
    Returns a list of RunType operations based on the parsed arguments.
    """
    if args.all:
        return [RunType.COMPILE_CIRCUIT, RunType.GEN_WITNESS, RunType.PROVE_WITNESS, RunType.GEN_VERIFY]
    ops = []
    if args.compile:
        ops.append(RunType.COMPILE_CIRCUIT)
    if args.gen_witness:
        ops.append(RunType.GEN_WITNESS)
    if args.prove:
        ops.append(RunType.PROVE_WITNESS)
    if args.verify:
        ops.append(RunType.GEN_VERIFY)
    if args.end_to_end:
        ops.append(RunType.END_TO_END)
    print(f"[INFO] Run operations determined: {[op.value for op in ops]}")
    return ops

def main():
    print("[INFO] Starting CLI for circuit operations...")
    args = parse_args()
    
    # If listing circuits, perform the search and exit.
    if args.list_circuits:
        list_available_circuits(args.circuit_search_path)
        return

    if not args.circuit:
        raise ValueError("The --circuit argument is required unless using --list_circuits.")

    # If --ecc is NOT triggered, try to resolve the expander-exec binary.
    if not args.ecc:
        print("[INFO] --ecc not triggered; attempting to resolve expander-exec binary...")
        resolved_exec = resolve_exec_path(args.exec_path, args.circuit)
        if resolved_exec:
            print(f"[INFO] Resolved expander-exec binary: {resolved_exec}")
            # Monkey-patch helper_functions.run_expander_exec to use the resolved binary.
            def patched_run_expander_exec(mode: str, circuit_file: str, witness_file: str, proof_file: str):
                assert mode in {"prove", "verify"}
                binary = resolved_exec
                print(f"[INFO] Using custom expander-exec binary: {binary}")
                cmd = [binary, mode, "--circuit-file", circuit_file, "--witness-file", witness_file]
                if mode == "prove":
                    cmd += ["--output-proof-file", proof_file]
                else:
                    cmd += ["--input-proof-file", proof_file]
                print(f"[INFO] Running command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"❌ expander-exec {mode} failed:\n{result.stderr}")
                else:
                    print(f"✅ expander-exec {mode} succeeded:\n{result.stdout}")
            helper_functions.run_expander_exec = patched_run_expander_exec
        else:
            print("[INFO] No expander-exec binary found; falling back to ECC behavior.")
            args.ecc = True

    circuit_name = args.circuit
    print(f"[INFO] Loading circuit module '{circuit_name}'...")
    # Load the circuit module/class using the optional search path.
    circuit = load_circuit(circuit_name, args.class_name, args.circuit_search_path)
    print(f"[INFO] Circuit module '{circuit_name}' loaded successfully.")
    
    print("[INFO] Resolving file paths for input and output...")
    # Resolve file paths and assign them to the circuit instance.
    input_path, output_path = resolve_file_paths(circuit_name, args.input, args.output, args.pattern)
    circuit.input_path = input_path
    circuit.output_path = output_path

    run_operations = get_run_operations(args)
    if not run_operations:
        raise ValueError("No operation specified. Please specify at least one operation flag or use --all.")
    
    print("[INFO] Starting execution of operations...")
    # Execute each operation in order.
    for op in run_operations:
        print(f"[INFO] Executing operation: {op.value}")
        circuit.base_testing(
            run_type=op,
            dev_mode=args.fresh_compile,
            circuit_path=args.circuit_path,
            input_file=args.input,
            output_file=args.output,
            witness_file=args.witness,
            proof_file=args.proof,
            ecc=args.ecc
        )
        print(f"[INFO] Operation {op.value} completed.")
    
    print("[INFO] All operations completed successfully.")

if __name__ == "__main__":
    main()
