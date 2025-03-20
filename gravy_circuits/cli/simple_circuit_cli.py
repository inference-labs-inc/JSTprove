import argparse
import subprocess
import os

def run_rust_binary(input_path, output_path):
    """Runs the Rust binary using the correct output directory."""

    # Locate Rust executable (one level up)
    rust_executable = os.path.abspath(os.path.join("..", "target", "release", "simple_circuit.exe"))

    # Convert input/output paths relative to GravyTesting-Internal/
    input_path = os.path.abspath(os.path.join("..", input_path))
    output_path = os.path.abspath(os.path.join("..", output_path))  # Now correctly points to 'output/'

    # Print resolved paths for debugging
    print(f"\n🚀 Resolved Paths:")
    print(f"   Rust Executable: {rust_executable}")
    print(f"   Input File: {input_path}")
    print(f"   Output File: {output_path}\n")

    # Rust expects <input> <output>
    cmd = [rust_executable, input_path, output_path]

    print(f"Running Rust command: {' '.join(cmd)}")

    # Execute the Rust binary
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)  # Print Rust output
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Rust binary: {e}")
        print(e.stderr)
    except FileNotFoundError as e:
        print(f"❌ FileNotFoundError: {e}. Check if paths are correct.")

def main():
    parser = argparse.ArgumentParser(description="Python CLI for running the Rust proof system")
    parser.add_argument("input", help="The file to read circuit inputs from (e.g., inputs/example.json)")
    parser.add_argument("output", help="The file to write circuit outputs to (e.g., output/proof.json)")

    args = parser.parse_args()

    # Run Rust binary with correctly resolved paths
    run_rust_binary(args.input, args.output)

if __name__ == "__main__":
    main()
