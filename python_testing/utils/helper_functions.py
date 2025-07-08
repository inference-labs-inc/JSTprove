from pathlib import Path
import threading
from time import time, sleep
import torch
import json
import os
import functools
from typing import Dict, Any, Tuple, Optional
from python_testing.utils.run_proofs import ZKProofsCircom, ZKProofsExpander, ZKProofSystems
from enum import Enum
import subprocess
from python_testing.utils.benchmarking_helpers import end_memory_collection, monitor_subprocess_memory, start_memory_collection

class RunType(Enum):
    BASE_TESTING = 'base_testing'
    END_TO_END = 'end_to_end'
    COMPILE_CIRCUIT = 'compile_circuit'
    GEN_WITNESS = 'gen_witness'
    PROVE_WITNESS = 'prove_witness'
    GEN_VERIFY = 'gen_verify'

# Decorator to compute outputs once and store in temp folder
def compute_and_store_output(func):
    """
    Decorator that computes outputs once per circuit instance and stores in temp folder.
    Instead of using in-memory cache, uses files in temp folder.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Define paths for storing outputs in temp folder
        temp_folder = getattr(self, 'temp_folder', "temp")
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
            
        output_cache_path = os.path.join(temp_folder, f"{self.name}_output_cache.json")
        
        # Check if cached output exists
        if os.path.exists(output_cache_path):
            print(f"Loading cached outputs for {self.name} from {output_cache_path}")
            try:
                with open(output_cache_path, 'r') as f:
                    output = json.load(f)
                    return output
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading cached output: {e}")
                # Continue to compute if loading fails
        
        # Compute outputs and cache them
        print(f"Computing outputs for {self.name}...")
        output = func(self, *args, **kwargs)
        
        # Store output in temp folder
        try:
            with open(output_cache_path, 'w') as f:
                json.dump(output, f)
            print(f"Stored outputs in {output_cache_path}")
        except IOError as e:
            print(f"Warning: Could not cache output to file: {e}")
            
        return output
    
    return wrapper



# Decorator to prepare input/output files
def prepare_io_files(func):
    """
    Decorator that prepares input and output files.
    This allows the function to be called independently.
    """
    
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):

        def resolve_folder(key, file_key=None, default=""):
            if key in kwargs:
                return kwargs[key]
            if file_key in kwargs and kwargs[file_key] is not None:
                return str(Path(kwargs[file_key]).parent)
            return getattr(self, key, default)
        
        # TODO These may need to be fixed as I think the function brings in files not folders, too much default
        # Use provided values or defaults from instance

        # Fix this
        # input_folder = kwargs.get("input_folder") or getattr(self, 'input_folder', "inputs")
        # proof_folder = kwargs.get("proof_folder") or getattr(self, 'proof_folder', "analysis")
        # temp_folder = kwargs.get("temp_folder") or getattr(self, 'temp_folder', "temp")
        # circuit_folder = kwargs.get("circuit_folder") or getattr(self, 'circuit_folder', "")
        # weights_folder = kwargs.get("weights_folder") or getattr(self, 'weights_folder', "weights")
        # output_folder = kwargs.get("output_folder") or getattr(self, 'output_folder', "output")
        # quantized_model_folder = kwargs.get("quantized_folder") or getattr(self, 'quantized_path', "quantized_model_folder")
        # proof_system = kwargs.get("proof_system") or getattr(self, 'proof_system', ZKProofSystems.Expander)
        input_folder = resolve_folder("input_folder", "input_file", "inputs")
        output_folder = resolve_folder("output_folder", "output_file", "output")
        quantized_model_folder = resolve_folder("quantized_folder", "quantized_path", "quantized_model_folder")
        proof_folder = resolve_folder("proof_folder", default="analysis")
        temp_folder = resolve_folder("temp_folder", default="temp")
        weights_folder = resolve_folder("weights_folder", default="weights")
        circuit_folder = resolve_folder("circuit_folder", default="")

        proof_system = kwargs.get("proof_system") or getattr(self, 'proof_system', ZKProofSystems.Expander)
        run_type = kwargs.pop("run_type")

        # Get file paths
        witness_file, input_file, proof_path, public_path, verification_key, circuit_name, weights_path, output_file = get_files(
            input_folder, proof_folder, temp_folder, circuit_folder, weights_folder, 
            self.name, output_folder, quantized_model_folder, proof_system
        )
        if not kwargs.get("input_file", None) is None:
            input_file = kwargs["input_file"]
        kwargs.pop("input_file", None)
        if not kwargs.get("output_file", None) is None:
            output_file = kwargs["output_file"]
        kwargs.pop("output_file", None)
        if not kwargs.get("proof_file", None) is None:
            proof_path = kwargs["proof_file"]
        kwargs.pop("proof_file", None)
        if not kwargs.get("witness_file", None) is None:
            witness_file = kwargs["witness_file"]

        if not kwargs.get("witness_file", None) is None:
            witness_file = kwargs["witness_file"]
        kwargs.pop("witness_file", None)
        if not kwargs.get("circuit_path", None) is None:
            circuit_path = kwargs["circuit_path"]
        else:
            circuit_path = None
        kwargs.pop("circuit_path", None)
        

        # No functionality for the following couple outside of this.
        # For now they are hardcoded
        if not kwargs.get("model_path", None) is None:
            model_path = kwargs["model_path"]
        else:
            model_path = None
        if not kwargs.get("quantized_model_path", None) is None:
            quantized_model_path = kwargs["quantized_model_path"]
        else:
            if circuit_path:
                name = os.path.splitext(os.path.basename(circuit_path))[0]
                quantized_model_path = f"{quantized_model_folder}/{name}_quantized_model.pth"
            else:
                quantized_model_path = f"{quantized_model_folder}/quantized_model_{self.__class__.__name__}.pth"
        
        
        # Store paths and data for use in the decorated function
        file_info = {
            'witness_file': witness_file,
            'input_file': input_file,
            'proof_file': proof_path,
            'public_path': public_path,
            'verification_key': verification_key,
            'circuit_name': circuit_name,
            'weights_path': weights_path,
            'output_file': output_file,
            'inputs': input_file,
            'weights': weights_path,
            'outputs': output_file,
            'output': output_file,
            'proof_system': proof_system,
            'model_path':model_path,
            'quantized_model_path': quantized_model_path
        }
        
        # Store file_info in the instance
        self._file_info = file_info
        # if "circuit_name" in kwargs.keys():
        #     kwargs.pop("circuit_name")
        

        # print(kwargs)
        # Call the original function with all arguments including file info
        return func(self, run_type, 
                    witness_file, input_file, proof_path, public_path, 
                    verification_key, circuit_name, weights_path, output_file,
                    proof_system, circuit_path = circuit_path, *args, **kwargs)
    
    return wrapper

def to_json(inputs: Dict[str, Any], path: str) -> None:
    """Write data to a JSON file."""
    with open(path, 'w') as outfile:
        json.dump(inputs, outfile)
    
def read_from_json(public_path: str) -> Dict[str, Any]:
    """Read data from a JSON file."""
    with open(public_path) as json_data:
        d = json.load(json_data)
        return d

def run_cargo_command(binary_name, command_type, args=None, dev_mode = False, bench = False):
    """
    Run a cargo command with the correct format based on the command type.
    
    Args:
        binary_name: Name of the cargo binary
        command_type: Type of command (e.g., 'run_proof', 'run_compile_circuit')
        args: Additional arguments as a dictionary
        
    Returns:
        Process return code
    """
    # Build the command
    import subprocess
    
    # Base command
    if dev_mode:
        cmd = ['cargo', 'run', '--bin', binary_name, '--release']
    else:
        cmd = [f'./target/release/{binary_name}']
    env = os.environ.copy()
    env["RUST_BACKTRACE"] = "1"
    
    # Add command type
    cmd.append(command_type)
    
    # Add arguments
    if args:
        for key, value in args.items():
            if isinstance(value, bool) and value:
                cmd.append(f'-{key}')
            else:
                cmd.append(f'-{key}')
                cmd.append(str(value))
    
    print(f"Running cargo command: {' '.join(cmd)}")
    try:
        if bench:
            stop_event, monitor_thread, monitor_results = start_memory_collection(binary_name)
        start_time = time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env = env)
        end_time = time() 
        print("\n--- BENCHMARK RESULTS ---")
        print(f"Rust time taken: {end_time - start_time:.4f} seconds")

        if bench:
            memory = end_memory_collection(stop_event, monitor_thread, monitor_results)
            print(f"Rust subprocess memory: {memory['total']:.2f} MB")

        print(result.stdout)

        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        raise e

def prove_and_verify(witness_file, input_file, proof_path, public_path, verification_key, 
                    circuit_name, proof_system: ZKProofSystems = ZKProofSystems.Expander, 
                    output_file=None, demo=False, dev_mode = False, ecc = True) -> None:
    """Process ZK proof based on the proof system type."""
    if proof_system == ZKProofSystems.Expander:
        assert output_file is not None, "Output_file must be specified for Expander proof system"
        
        if ecc: 
            # Extract the binary name from the circuit path
            binary_name = os.path.basename(circuit_name)
            
            # Prepare arguments according to the expected format
            args = {
                'i': input_file,
                'o': output_file,
            }
            
            # Run the command
            try:
                run_cargo_command(binary_name, 'run_proof', args, dev_mode=False)
            except Exception as e:
                print(f"Warning: Could not complete prove_and_verify: {e}")
                print("This may be expected if the Rust binary is not available.")
                print(f"Input file has been written to: {input_file}")
                print(f"Output file has been written to: {output_file}")
        else:
            # Direct Expander call via expander-exec binary
            paths = get_expander_file_paths(circuit_name)
            run_expander_raw(
                mode="prove",
                circuit_file=paths["circuit_file"],
                witness_file=paths["witness_file"],
                proof_file=paths["proof_file"]
            )

            run_expander_raw(
                mode="verify",
                circuit_file=paths["circuit_file"],
                witness_file=paths["witness_file"],
                proof_file=paths["proof_file"]
            )
            
    elif proof_system == ZKProofSystems.Circom:
        raise NotImplementedError("Circom is not implemented")

        circuit = ZKProofsCircom(circuit_name)
        res = circuit.compile_circuit()
        circuit.compute_witness(witness_file, input_file, wasm=True, c=False)
        circuit.proof_setup(verification_key)
        circuit.proof(witness_file, proof_path, public_path)
        circuit.verify(verification_key, public_path, proof_path)
    else:
        raise NotImplementedError(f"Proof system {proof_system} not implemented")
    
def get_expander_file_paths(circuit_name: str):
    return {
        "circuit_file": f"{circuit_name}_circuit.txt",
        "witness_file": f"{circuit_name}_witness.txt",
        "proof_file":   f"{circuit_name}_proof.txt"
    }
    
# env RUSTFLAGS="-C target-cpu=native" mpiexec -n 1 cargo run --manifest-path Expander/Cargo.toml --bin expander-exec --release -- -p Raw prove -c slices/segment_4/segment_4_circuit.compiled -w slices/segment_4/segment_4_witness.compiled -o slices/segment_4/segment_4_proof.bin

def run_expander_exec(mode: str, circuit_file: str, witness_file: str, proof_file: str):
    assert mode in {"prove", "verify"}
    binary = "./expander-exec"  # or full path if needed

    args = [binary, mode, "--circuit-file", circuit_file, "--witness-file", witness_file]

    if mode == "prove":
        args += ["--output-proof-file", proof_file]
    else:
        args += ["--input-proof-file", proof_file]


    result = subprocess.run(args, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ expander-exec {mode} failed:\n{result.stderr}")
    else:
        print(f"✅ expander-exec {mode} succeeded:\n{result.stdout}")

#   env RUSTFLAGS="-C target-cpu=native" mpiexec -n 1 cargo run --manifest-path Expander/Cargo.toml --bin expander-exec --release -- -p Raw verify -c slices/segment_4/segment_4_circuit.compiled -w slices/segment_4/segment_4_witness.compiled -i slices/segment_4/segment_4_proof.bin


def run_expander_raw(mode: str, circuit_file: str, witness_file: str, proof_file: str, pcs_type: str = "Raw", bench = False):
    assert mode in {"prove", "verify"}

    pcs_type = "Raw" #or Hyrax
    # pcs_type = "Hyrax"

    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C target-cpu=native"
    time_measure = "/usr/bin/time" 
    time_flag = "-l"

    arg_1 = 'mpiexec' 
    arg_2 = '-n'
    arg_3 = '1'
    command = 'cargo' 
    command_2 = 'run'
    manifest_path = 'Expander/Cargo.toml'
    binary = 'expander-exec'

    args = [time_measure, time_flag, arg_1, arg_2, arg_3, command, command_2, '--manifest-path', manifest_path,'--bin', binary, '--release', '--', '-p', pcs_type]
    if mode == 'prove':
        args.append("prove")
        proof_command = '-o'
    else:
        args.append("verify")
        proof_command = '-i'


    args.append('-c')
    args.append(circuit_file)
    
    args.append('-w')
    args.append(witness_file)

    args.append(proof_command)
    args.append(proof_file)
    # TODO wrap and only run if benchmarking internally
    if bench:
        stop_event, monitor_thread, monitor_results = start_memory_collection("expander-exec")
    start_time = time()
    result = subprocess.run(args, env = env, capture_output=True, text=True)
    end_time = time() 

    print("\n--- BENCHMARK RESULTS ---")
    print(f"Rust time taken: {end_time - start_time:.4f} seconds")

    if bench:
        memory = end_memory_collection(stop_event, monitor_thread, monitor_results)
        print(f"Rust subprocess memory: {memory['total']:.2f} MB")
    # if bench:
    #     memory = end_memory_collection(stop_event, monitor_thread, monitor_results)
    #     print(memory)


    if result.returncode != 0:
        print(f"❌ expander-exec {mode} failed:\n{result.stderr}")

    else:
        print(f"✅ expander-exec {mode} succeeded:\n{result.stdout}")

    print(f"Time taken: {end_time - start_time:.4f} seconds")
    



def compile_circuit(circuit_name, circuit_path, proof_system: ZKProofSystems = ZKProofSystems.Expander, dev_mode = False, bench = False):
    """Compile a circuit."""

    if proof_system == ZKProofSystems.Expander:
        # Extract the binary name from the circuit path 
        binary_name = os.path.basename(circuit_name)
        
        # Prepare arguments
        args = {
            'n': circuit_name,
            'c': circuit_path,
        }
        # Run the command
        try:
            run_cargo_command(binary_name, 'run_compile_circuit', args, dev_mode, bench)
        except Exception as e:
            print(f"Warning: Compile operation failed: {e}")
            print(f"Using binary: {binary_name}")
            
    elif proof_system == ZKProofSystems.Circom:
        raise NotImplementedError("Circom is not implemented")
        circuit = ZKProofsCircom(circuit_name)
        res = circuit.compile_circuit()
    else:
        raise NotImplementedError(f"Proof system {proof_system} not implemented")

def generate_witness(circuit_name, circuit_path, witness_file, input_file, output_file, 
                    proof_system: ZKProofSystems = ZKProofSystems.Expander, dev_mode = False, bench = False):
    """Generate witness for a circuit."""
    if proof_system == ZKProofSystems.Expander:
        # Extract the binary name from the circuit path
        binary_name = os.path.basename(circuit_name)
        
        # Prepare arguments
        args = {
            'n': circuit_name,
            'c': circuit_path,
            'i': input_file,
            'o': output_file,
            'w': witness_file,
            'c': circuit_path
        }
        
        # Run the command
        try:
            run_cargo_command(binary_name, 'run_gen_witness', args, dev_mode, bench)
        except Exception as e:
            print(f"Warning: Witness generation failed: {e}")
            
    elif proof_system == ZKProofSystems.Circom:
        raise NotImplementedError("Circom is not implemented")
        circuit = ZKProofsCircom(circuit_name)
        circuit.compute_witness(witness_file, input_file, wasm=True, c=False)
    else:
        raise NotImplementedError(f"Proof system {proof_system} not implemented")


def generate_proof(circuit_name, circuit_path, witness_file, proof_file, 
                    proof_system: ZKProofSystems = ZKProofSystems.Expander, dev_mode = False, ecc = True, bench = False):
    """Generate witness for a circuit."""
    if proof_system == ZKProofSystems.Expander:
        if ecc:
            # Extract the binary name from the circuit path
            binary_name = os.path.basename(circuit_name)
            
            # Prepare arguments
            args = {
                'n': circuit_name,
                'c': circuit_path,
                'w': witness_file,
                'p': proof_file,
                'c': circuit_path
            }
            
            # Run the command
            try:
                run_cargo_command(binary_name, 'run_prove_witness', args, dev_mode, bench)
            except Exception as e:
                print(f"Warning: Proof generation failed: {e}")
        else:
            # Direct Expander call via expander-exec binary
            # paths = get_expander_file_paths(circuit_name)
            # run_expander_exec(
            #     mode="prove",
            #     circuit_file=circuit_path,
            #     witness_file=witness_file,
            #     proof_file=proof_file
            # )
            run_expander_raw(
                mode="prove",
                circuit_file=circuit_path,
                witness_file=witness_file,
                proof_file=proof_file,
                bench = bench
            )
            
    elif proof_system == ZKProofSystems.Circom:
        raise NotImplementedError("Circom is not implemented")
        circuit = ZKProofsCircom(circuit_name)
        circuit.proof(witness_file, proof_file, public_path="")
    else:
        raise NotImplementedError(f"Proof system {proof_system} not implemented")


def generate_verification(circuit_name, circuit_path, input_file, output_file, witness_file, proof_file, proof_system: ZKProofSystems = ZKProofSystems.Expander, dev_mode = False, ecc = True, bench = False):
    """Generate verification for a circuit."""
    if proof_system == ZKProofSystems.Expander:
        if ecc:
            # Extract the binary name from the circuit path
            binary_name = os.path.basename(circuit_name)
            
            # Prepare arguments
            args = {
                'n': circuit_name,
                'c': circuit_path,
                'i': input_file,
                'o': output_file,
                'w': witness_file,
                'p': proof_file,
                'c': circuit_path
            }
            
            # Run the command
            try:
                run_cargo_command(binary_name, 'run_gen_verify', args, dev_mode, bench)
            except Exception as e:
                print(f"Warning: Verification generation failed: {e}")
        else:
            # Direct Expander call via expander-exec binary
            # paths = get_expander_file_paths(circuit_name)
            # run_expander_exec(
            #     mode="verify",
            #     circuit_file=circuit_path,
            #     witness_file=witness_file,
            #     proof_file=proof_file
            # )
            run_expander_raw(
                mode="verify",
                circuit_file=circuit_path,
                witness_file=witness_file,
                proof_file=proof_file,
                bench = bench
            )
            
    elif proof_system == ZKProofSystems.Circom:
        raise NotImplementedError("Circom is not implemented")
    else:
        raise NotImplementedError(f"Proof system {proof_system} not implemented")

def run_end_to_end(circuit_name, circuit_path, input_file, output_file, 
                  proof_system: ZKProofSystems = ZKProofSystems.Expander, demo=False, dev_mode = False, ecc = True):
    """Run end-to-end proof."""
    if proof_system == ZKProofSystems.Expander:
        # Extract the binary name from the circuit path
        # binary_name = os.path.basename(circuit_name)
        
        # # Prepare arguments
        # args = {
        #     'c': circuit_path,
        #     'i': input_file,
        #     'o': output_file,
        # }
        
        # # Run the command
        # try:
        #     run_cargo_command(binary_name, 'run_end_to_end', args, dev_mode)
        # except Exception as e:
        #     print(f"Warning: End-to-end operation failed: {e}")

        base, ext = os.path.splitext(circuit_path)  # Split the filename and extension
        witness_file = f"{base}_witness{ext}"
        proof_file = f"{base}_proof{ext}"


        # d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")
        # d.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
        compile_circuit(circuit_name, circuit_path, proof_system, dev_mode)
        generate_witness(circuit_name, circuit_path, witness_file, input_file, output_file, proof_system, dev_mode)
        generate_proof(circuit_name, circuit_path, witness_file, proof_file, proof_system, dev_mode, ecc)
        generate_verification(circuit_name, circuit_path, input_file, output_file, witness_file, proof_file, proof_system, dev_mode, ecc)

            
    elif proof_system == ZKProofSystems.Circom:
        raise NotImplementedError("Circom is not implemented")
    else:
        raise NotImplementedError(f"Proof system {proof_system} not implemented")

def get_files(input_folder, proof_folder, temp_folder, circuit_folder, weights_folder, 
             name, output_folder, quantized_model_folder, proof_system):
    """Get file paths, creating folders as needed."""
    create_folder(input_folder)
    create_folder(proof_folder)
    create_folder(temp_folder)
    create_folder(output_folder)
    create_folder(weights_folder)
    create_folder(quantized_model_folder)


    
    input_file = os.path.join(input_folder, f"{name}_input.json")
    public_path = os.path.join(proof_folder, f"{name}_public.json")
    verification_key = os.path.join(temp_folder, f"{name}_verification_key.json")
    weights_path = os.path.join(weights_folder, f"{name}_weights.json")
    
    if proof_system == ZKProofSystems.Expander:
        circuit_name = os.path.join(circuit_folder, f"{name}")
        witness_file = os.path.join(f"{name}_witness.txt")
        proof_path = os.path.join(proof_folder, f"{name}_proof.bin")
    elif proof_system == ZKProofSystems.Circom:
        raise NotImplementedError("Circom is not implemented")
        circuit_name = os.path.join(circuit_folder, f"{name}.circom")
        witness_file = os.path.join(temp_folder, f"{name}_witness.wtns")
        proof_path = os.path.join(proof_folder, f"{name}_proof.json")
    else:
        raise NotImplementedError(f"Proof system {proof_system} not implemented")

    output_file = os.path.join(output_folder, f"{name}_output.json")
    return witness_file, input_file, proof_path, public_path, verification_key, circuit_name, weights_path, output_file

def create_folder(directory: str) -> None:
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)