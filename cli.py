#!/usr/bin/env python3
"""
CLI for running circuit operations. Dynamically loads circuit modules and resolves file paths using fuzzy matching.
"""

import argparse
import difflib
import importlib
import logging
from pathlib import Path
from typing import Optional, Tuple

from python_testing.circuit_components.circuit_helpers import RunType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Since cli.py is now in the project root, set PROJECT_ROOT accordingly
PROJECT_ROOT = Path(__file__).resolve().parent  # GravyTesting-Internal

def find_file(filename: str, default_path: Optional[Path] = None) -> Path:
    """
    Finds a JSON file in the project root using exact or fuzzy matching.
    """
    if not filename.endswith(".json"):
        filename += ".json"

    if default_path:
        candidate = PROJECT_ROOT / default_path
        if candidate.is_file():
            return candidate

    all_json_files = list(PROJECT_ROOT.rglob("*.json"))
    for path in all_json_files:
        if path.name == filename:
            return path

    file_names = [p.name for p in all_json_files]
    close_matches = difflib.get_close_matches(filename, file_names, n=1, cutoff=0.6)
    if close_matches:
        fuzzy_match = close_matches[0]
        for path in all_json_files:
            if path.name == fuzzy_match:
                logger.warning(f"Could not find '{filename}', using close match '{fuzzy_match}'")
                return path

    raise FileNotFoundError(f"Could not find file '{filename}' (or a close match) under {PROJECT_ROOT}.")

def load_circuit(circuit_name: str, class_name: str = "SimpleCircuit"):
    """
    Dynamically loads a circuit module and returns an instance of the specified circuit class.
    """
    try:
        circuit_module = importlib.import_module(f"python_testing.circuit_models.{circuit_name}")
        circuit_class = getattr(circuit_module, class_name)
        return circuit_class()
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(f"Could not load circuit '{circuit_name}' with class '{class_name}': {e}")

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
        if "input" in input_filename:
            output_filename = input_filename.replace("input", "output")
        else:
            output_filename = f"{circuit_name}_output.json"
    else:
        input_filename = f"{circuit_name}_input.json"
        output_filename = f"{circuit_name}_output.json"

    if input_override:
        input_path = str(find_file(input_override))
    else:
        input_path = str(find_file(input_filename, Path("inputs") / input_filename))

    if output_override:
        output_path = str(find_file(output_override))
    else:
        output_path = str(find_file(output_filename, Path("output") / output_filename))

    return input_path, output_path

def list_available_circuits(search_path: Optional[str] = None):
    """
    Lists all available circuit files by searching for Python files that define a class inheriting
    from ZKModel or Circuit. By default, it searches in:
      - python_testing/circuit_components
      - python_testing/circuit_models
    The user can override this search directory using the --circuit_search_path flag.
    """
    search_paths = []
    if search_path:
        search_paths.append(PROJECT_ROOT / search_path)
    else:
        search_paths.append(PROJECT_ROOT / "python_testing" / "circuit_components")
        search_paths.append(PROJECT_ROOT / "python_testing" / "circuit_models")

    circuit_files = set()
    for base in search_paths:
        if base.exists():
            for path in base.rglob("*.py"):
                try:
                    content = path.read_text()
                except Exception:
                    continue
                if "class " in content and ( "(ZKModel):" in content or "(Circuit):" in content ):
                    circuit_files.add(str(path.relative_to(PROJECT_ROOT)))
        else:
            logger.warning(f"Search path {base} does not exist.")
    if circuit_files:
        print("Available circuit files:")
        for file in sorted(circuit_files):
            print(file)
    else:
        print("No circuit files found.")

def main():
    parser = argparse.ArgumentParser(description="Run circuit operations.")
    
    # Operation flags for various RunTypes
    parser.add_argument("--compile_circuit", action="store_true", help="Compile the circuit.")
    parser.add_argument("--gen_witness", action="store_true", help="Generate witness for the circuit.")
    parser.add_argument("--prove_witness", action="store_true", help="Generate witness and proof.")
    parser.add_argument("--gen_verify", action="store_true", help="Run verification.")
    parser.add_argument("--base_testing", action="store_true", help="Run base testing (prove and verify).")
    parser.add_argument("--end_to_end", action="store_true", help="Run end-to-end circuit testing.")
    parser.add_argument("--all", action="store_true", help="Run all stages (compile_circuit, gen_witness, prove_witness, gen_verify).")
    
    # Flag to list available circuit files, with optional search path override.
    parser.add_argument("--list_circuits", action="store_true", help="List all available circuit files.")
    parser.add_argument("--circuit_search_path", type=str, help="Directory to search for circuits (relative to project root).")
    
    # Optional file overrides and pattern
    parser.add_argument("--input", type=str, help="Path to the input JSON file.")
    parser.add_argument("--output", type=str, help="Path to the output JSON file.")
    parser.add_argument("--pattern", type=str, help="Optional pattern for input/output filenames with '{circuit}' placeholder.")
    
    # Circuit module and class specification
    parser.add_argument("--circuit", type=str, help="Name of the circuit module (under python_testing/circuit_models/).")
    parser.add_argument("--class", dest="class_name", type=str, default="SimpleCircuit", help="Name of the circuit class to load. Defaults to 'SimpleCircuit'.")

    args = parser.parse_args()
    
    # If the user requested to list circuits, perform the search and exit.
    if args.list_circuits:
        list_available_circuits(args.circuit_search_path)
        return
    
    if not args.circuit:
        parser.error("The --circuit argument is required unless using --list_circuits.")
    
    circuit_name = args.circuit
    
    # Load the circuit module/class
    circuit = load_circuit(circuit_name, args.class_name)
    
    # Resolve file paths
    input_path, output_path = resolve_file_paths(circuit_name, args.input, args.output, args.pattern)
    circuit.input_path = input_path
    circuit.output_path = output_path
    
    # Determine which operations to run based on flags
    run_operations = []
    if args.all:
        run_operations = [
            RunType.COMPILE_CIRCUIT,
            RunType.GEN_WITNESS,
            RunType.PROVE_WITNESS,
            RunType.GEN_VERIFY,
        ]
    else:
        if args.base_testing:
            run_operations.append(RunType.BASE_TESTING)
        if args.compile_circuit:
            run_operations.append(RunType.COMPILE_CIRCUIT)
        if args.gen_witness:
            run_operations.append(RunType.GEN_WITNESS)
        if args.prove_witness:
            run_operations.append(RunType.PROVE_WITNESS)
        if args.gen_verify:
            run_operations.append(RunType.GEN_VERIFY)
        if args.end_to_end:
            run_operations.append(RunType.END_TO_END)
    
    if not run_operations:
        parser.error("No operation specified. Please specify at least one operation flag or use --all.")
    
    # Execute each specified operation in order.
    for op in run_operations:
        circuit.base_testing(run_type=op)

if __name__ == "__main__":
    main()