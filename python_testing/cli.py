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

# Dynamically determine the root of the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # e.g., GravyTesting-Internal/


def find_file(filename: str, default_path: Optional[Path] = None) -> Path:
    """
    Finds a JSON file in the project root using exact or fuzzy matching.
    
    Args:
        filename (str): The filename to search for. If not ending with '.json', it will be appended.
        default_path (Optional[Path]): A default relative path to check first.
        
    Returns:
        Path: The resolved file path.
        
    Raises:
        FileNotFoundError: If the file cannot be found.
    """
    if not filename.endswith(".json"):
        filename += ".json"

    # Check default path first
    if default_path:
        candidate = PROJECT_ROOT / default_path
        if candidate.is_file():
            return candidate

    # Build list of all .json files
    all_json_files = list(PROJECT_ROOT.rglob("*.json"))

    # Exact match fallback
    for path in all_json_files:
        if path.name == filename:
            return path

    # Fuzzy match fallback
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
    
    Args:
        circuit_name (str): Name of the circuit module under python_testing.circuit_models.
        class_name (str): Name of the circuit class to load. Defaults to 'SimpleCircuit'.
        
    Returns:
        An instance of the circuit.
        
    Raises:
        ValueError: If the circuit module or class cannot be loaded.
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
    
    Args:
        circuit_name (str): Name of the circuit.
        input_override (Optional[str]): Override for input file path.
        output_override (Optional[str]): Override for output file path.
        pattern (Optional[str]): Optional pattern for input/output filenames with '{circuit}' placeholder.
    
    Returns:
        A tuple of (input_path, output_path) as strings.
    """
    if pattern:
        input_filename = pattern.format(circuit=circuit_name)  # assume pattern is for input
        # If "input" appears in the pattern, derive output filename by replacing it with "output"
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
        # Assume input files are in the "inputs" directory relative to the project root
        input_path = str(find_file(input_filename, Path("inputs") / input_filename))

    if output_override:
        output_path = str(find_file(output_override))
    else:
        # Assume output files are in the "output" directory relative to the project root
        output_path = str(find_file(output_filename, Path("output") / output_filename))

    return input_path, output_path


def main():
    parser = argparse.ArgumentParser(description="Run circuit operations.")

    # Operation flags
    parser.add_argument("--compile_circuit", action="store_true", help="Compile the circuit.")
    parser.add_argument("--prove_witness", action="store_true", help="Generate witness and proof.")
    parser.add_argument("--gen_verify", action="store_true", help="Run verification.")
    parser.add_argument("--all", action="store_true", help="Run all stages: compile circuit, prove witness, and generate verification.")

    # Optional file overrides and pattern
    parser.add_argument("--input", type=str, help="Path to the input JSON file.")
    parser.add_argument("--output", type=str, help="Path to the output JSON file.")
    parser.add_argument("--pattern", type=str, help="Optional pattern for input/output filenames with '{circuit}' placeholder.")

    # Required circuit argument and optional class name for flexibility
    parser.add_argument("--circuit", type=str, required=True, help="Name of the circuit module (under python_testing/circuit_models/).")
    parser.add_argument("--class", dest="class_name", type=str, default="SimpleCircuit", help="Name of the circuit class to load. Defaults to 'SimpleCircuit'.")

    args = parser.parse_args()
    circuit_name = args.circuit

    # Load the circuit
    circuit = load_circuit(circuit_name, args.class_name)

    # Resolve file paths
    input_path, output_path = resolve_file_paths(circuit_name, args.input, args.output, args.pattern)
    circuit.input_path = input_path
    circuit.output_path = output_path

    # Determine operations to run
    if args.all:
        compile_flag = prove_flag = verify_flag = True
    else:
        compile_flag = args.compile_circuit
        prove_flag = args.prove_witness
        verify_flag = args.gen_verify

    if compile_flag:
        circuit.base_testing(run_type=RunType.COMPILE_CIRCUIT)
    if prove_flag:
        circuit.base_testing(run_type=RunType.PROVE_WITNESS)
    if verify_flag:
        circuit.base_testing(run_type=RunType.GEN_VERIFY)


if __name__ == "__main__":
    main()