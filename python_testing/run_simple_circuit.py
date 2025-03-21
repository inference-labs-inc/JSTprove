import argparse
from python_testing.circuit_models.simple_circuit import SimpleCircuit
from python_testing.circuit_components.circuit_helpers import RunType

def main():
    parser = argparse.ArgumentParser(description="Run simple circuit operations.")
    
    # Run type flags
    parser.add_argument("--compile_circuit", action="store_true", help="Compile the circuit.")
    parser.add_argument("--prove_witness", action="store_true", help="Generate witness and proof.")
    parser.add_argument("--gen_verify", action="store_true", help="Run verification.")

    # Optional file arguments
    parser.add_argument("--input", type=str, help="Path to the input JSON file.")
    parser.add_argument("--output", type=str, help="Path to the output JSON file.")

    args = parser.parse_args()

    circuit = SimpleCircuit()

    # Optionally inject input/output paths
    if args.input:
        circuit.input_path = args.input
    if args.output:
        circuit.output_path = args.output

    # Run selected operations
    if args.compile_circuit:
        circuit.base_testing(run_type=RunType.COMPILE_CIRCUIT)
    if args.prove_witness:
        circuit.base_testing(run_type=RunType.PROVE_WITNESS)
    if args.gen_verify:
        circuit.base_testing(run_type=RunType.GEN_VERIFY)

if __name__ == "__main__":
    main()
