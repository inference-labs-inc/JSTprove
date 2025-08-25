from pathlib import Path
from time import time
import json
import os
import functools
from typing import Callable, Dict, Any
from enum import Enum
import subprocess
from python.core.utils.benchmarking_helpers import end_memory_collection, start_memory_collection

class RunType(Enum):
    END_TO_END = 'end_to_end'
    COMPILE_CIRCUIT = 'compile_circuit'
    GEN_WITNESS = 'gen_witness'
    PROVE_WITNESS = 'prove_witness'
    GEN_VERIFY = 'gen_verify'

class ZKProofSystems(Enum):
    Expander = "Expander"

# Decorator to compute outputs once and store in temp folder
def compute_and_store_output(func: Callable) -> Callable:
    """Decorator that computes outputs once per circuit instance and stores in temp folder.
    Instead of using in-memory cache, uses files in temp folder.

    Args:
        func (Callable): Method that computes outputs to be cached.

    Returns:
        Callable: Wrapped function that reads/writes a caches.
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
def prepare_io_files(func: Callable) -> Callable:
    """Decorator that prepares input and output files.
    This allows the function to be called independently.

    Args:
        func (Callable): The function requiring prepared file paths.

    Returns:
        Callable: Wrapped function with prepared file paths injected into its arguments.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):

        def resolve_folder(key: str, file_key: str = None, default: str = ""):
            if key in kwargs:
                return kwargs[key]
            if file_key in kwargs and kwargs[file_key] is not None:
                return str(Path(kwargs[file_key]).parent)
            return getattr(self, key, default)
    
        
        # temp_folder = kwargs.get("temp_folder") or getattr(self, 'temp_folder', "python/models/temp")
        input_folder = resolve_folder("input_folder", "input_file", default = "python/models/inputs")
        output_folder = resolve_folder("output_folder", "output_file", default = "python/models/output")
        proof_folder = resolve_folder("proof_folder", "proof_file", default = "python/models/proofs")
        quantized_model_folder = resolve_folder("quantized_folder", "quantized_path", default = "python/models/quantized_model_folder")
        weights_folder = resolve_folder("weights_folder", default="python/models/weights")
        circuit_folder = resolve_folder("circuit_folder", default="python/models/")

        proof_system = kwargs.get("proof_system") or getattr(self, 'proof_system', ZKProofSystems.Expander)
        run_type = kwargs.pop("run_type")

        files = get_files(
            self.name,
            proof_system,
            {
                "input": input_folder,
                "proof": proof_folder,
                "circuit": circuit_folder,
                "weights": weights_folder,
                "output": output_folder,
                "quantized_model": quantized_model_folder
            }
        )
        # Set to defaults
        witness_file = files["witness_file"]
        input_file = files["input_file"]
        proof_path = files["proof_path"]
        public_path = files["public_path"]
        circuit_name = files["circuit_name"]
        weights_path = files["weights_path"]
        output_file = files["output_file"]

        witness_file = kwargs.pop("witness_file", files["witness_file"])
        input_file = kwargs.pop("input_file", files["input_file"])
        proof_path = kwargs.pop("proof_file", files["proof_path"])
        public_path = files["public_path"]
        circuit_name = files["circuit_name"]
        weights_path = files["weights_path"]
        output_file = kwargs.pop("output_file", files["output_file"])
        circuit_path = kwargs.pop("circuit_path", None)

        model_path = kwargs.pop("model_path", None)

        
        quantized_model_path = kwargs.pop("quantized_model_path", None)
        # No functionality for the following couple outside of this.
        # For now they are hardcoded
        if quantized_model_path is None:
            if circuit_path:
                name = Path(circuit_path).stem
                quantized_model_path = f"{quantized_model_folder}/{name}_quantized_model.onnx"
            else:
                quantized_model_path = f"{quantized_model_folder}/quantized_model_{self.__class__.__name__}.onnx"
        
        
        # Store paths and data for use in the decorated function
        self._file_info = {
            'witness_file': witness_file,
            'input_file': input_file,
            'proof_file': proof_path,
            'public_path': public_path,
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

        # Call the original function with all arguments including file info
        return func(self, run_type, 
                    witness_file, input_file, proof_path, public_path, 
                    "", circuit_name, weights_path, output_file,
                    proof_system, circuit_path = circuit_path, *args, **kwargs)
    
    return wrapper

def to_json(inputs: Dict[str, Any], path: str) -> None:
    """Write data to a JSON file.

    Args:
        inputs (Dict[str, Any]): Data to be serialized.
        path (str): Path where the JSON file will be written.
    """
    with open(path, 'w') as outfile:
        json.dump(inputs, outfile)
    
def read_from_json(public_path: str) -> Dict[str, Any]:
    """Read data from a JSON file."""
    with open(public_path) as json_data:
        d = json.load(json_data)
        return d

def run_cargo_command(binary_name: str,
                      command_type: str,
                      args: Dict[str, str]=None,
                      dev_mode: bool = False,
                      bench: bool = False
                      ) -> subprocess.CompletedProcess[str]:
    """Run a cargo command with the correct format based on the command type.

    Args:
        binary_name (str): Name of the Cargo binary.
        command_type (str): Command type (e.g., 'run_proof', 'run_compile_circuit').
        args (Dict[str, str], optional): dictionary of CLI arguments. Defaults to None.
        dev_mode (bool, optional): If True, run with `cargo run --release` instead of prebuilt binary. Defaults to False.
        bench (bool, optional): If True, measure execution time and memory usage. Defaults to False.

    Raises:
        subprocess.CalledProcessError: If the Cargo command fails.

    Returns:
        subprocess.CompletedProcess[str]: Exit message from the subprocess.
    """
    # Base command
    cmd = ['cargo', 'run', '--bin', binary_name, '--release'] if dev_mode else [f'./target/release/{binary_name}']
    env = os.environ.copy()
    env["RUST_BACKTRACE"] = "1"

    # Add command type
    cmd.append(command_type)

    # Add arguments
    if args:
        for key, value in args.items():
            cmd.append(f'-{key}')
            if not (isinstance(value, bool) and value):
                cmd.append(str(value))
    
    print(f"Running cargo command: {' '.join(cmd)}")
    
    try:
        
        if bench:
            stop_event, monitor_thread, monitor_results = start_memory_collection(binary_name)
        start_time = time()
        result = subprocess.run(cmd, check = True, capture_output = True, text = True, env = env)
        end_time = time() 
        print("\n--- BENCHMARK RESULTS ---")
        print(f"Rust time taken: {end_time - start_time:.4f} seconds")

        if bench:
            memory = end_memory_collection(stop_event, monitor_thread, monitor_results)
            print(f"Rust subprocess memory: {memory['total']:.2f} MB")

        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(f"Expander {command_type.value} failed:\n{result.stderr}")

        return result
    except subprocess.CalledProcessError as e:
        print(f"Cargo command failed (return code {e.returncode}): {e.stderr}")
        raise
    
def get_expander_file_paths(circuit_name: str) -> Dict[str, str]:
    """Generate standard file paths for an Expander circuit.

    Args:
        circuit_name (str): The base name of the circuit.

    Returns:
        Dict[str, str]: Dictionary containing file paths with keys: circuit_file, witness_file, proof_file
    """     
    return {
        "circuit_file": f"{circuit_name}_circuit.txt",
        "witness_file": f"{circuit_name}_witness.txt",
        "proof_file":   f"{circuit_name}_proof.txt"
    }
    

def run_expander_raw(mode: str,
                     circuit_file: str,
                     witness_file: str,
                     proof_file: str,
                     pcs_type: str = "Hyrax",
                     bench: bool = False
                     ) -> subprocess.CompletedProcess[str]:
    """Run the Expander executable directly using Cargo.

    Args:
        mode (str): Either "prove" or "verify", determining operation mode.
        circuit_file (str): Path to the circuit definition file.
        witness_file (str): Path to the witness file.
        proof_file (str): Path to the proof file (input for verification, 
                          output for proving).
        pcs_type (str, optional): Polynomial commitment scheme type ("Hyrax" or "Raw"). Defaults to "Hyrax".
        bench (bool, optional): If True, collect runtime and memory benchmark data. Defaults to False.

    Returns:
        subprocess.CompletedProcess[str]: Exit message from the subprocess.
    """
    assert mode in {"prove", "verify"}

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

    if result.returncode != 0:
        raise RuntimeError(f"Expander {mode} failed:\n{result.stderr}")

    else:
        print(f"✅ expander-exec {mode} succeeded:\n{result.stdout}")

    print(f"Time taken: {end_time - start_time:.4f} seconds")

    return result

def compile_circuit(circuit_name: str,
                    circuit_path: str,
                    proof_system: ZKProofSystems = ZKProofSystems.Expander,
                    dev_mode: bool = True,
                    bench: bool = False
                    ) -> subprocess.CompletedProcess[str]:
    """Compile a model into zk circuit

    Args:
        circuit_name (str): Name of the circuit.
        circuit_path (str):  Path to the circuit source file.
        proof_system (ZKProofSystems, optional): Proof system to use. Defaults to ZKProofSystems.Expander.
        dev_mode (bool, optional): If True, recompiles the rust binary (run in development mode). Defaults to True.
        bench (bool, optional): Whether or not to run benchmarking metrics. Defaults to False.

    Raises:
        NotImplementedError: If proof system is not supported.
    Returns:
        subprocess.CompletedProcess[str]: Exit message from the subprocess.
    """
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
            return run_cargo_command(binary_name, 'run_compile_circuit', args, dev_mode, bench)
        except Exception as e:
            print(f"Warning: Compile operation failed: {e}")
            print(f"Using binary: {binary_name}")
            
    else:
        raise NotImplementedError(f"Proof system {proof_system} not implemented")

def generate_witness(circuit_name: str,
                     circuit_path: str,
                     witness_file: str,
                     input_file: str,
                     output_file: str, 
                    proof_system: ZKProofSystems = ZKProofSystems.Expander,
                    dev_mode: bool = False,
                    bench: bool = False
                    ) -> subprocess.CompletedProcess[str]:
    """Generate a witness file for a circuit.

    Args:
        circuit_name (str): Name of the circuit.
        circuit_path (str): Path to the circuit definition.
        witness_file (str): Path to the output witness file.
        input_file (str): Path to the input JSON file with private inputs.
        output_file (str): Path to the output JSON file with computed outputs.
        proof_system (ZKProofSystems, optional): Proof system to use. Defaults to ZKProofSystems.Expander.
        dev_mode (bool, optional): If True, recompiles the rust binary (run in development mode). Defaults to False.
        bench (bool, optional): If True, enable benchmarking. Defaults to False.

    Raises:
        NotImplementedError: If proof system is not supported.

    Returns:
        subprocess.CompletedProcess[str]: Exit message from the subprocess.
    """    
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
        }
        # Run the command
        try:
            return run_cargo_command(binary_name, 'run_gen_witness', args, dev_mode, bench)
        except Exception as e:
            print(f"Warning: Witness generation failed: {e}")        
    else:
        raise NotImplementedError(f"Proof system {proof_system} not implemented")


def generate_proof(circuit_name: str,
                   circuit_path: str,
                   witness_file: str,
                   proof_file: str, 
                    proof_system: ZKProofSystems = ZKProofSystems.Expander,
                    dev_mode: bool = False,
                    ecc: bool = True,
                    bench: bool = False
                    ) -> subprocess.CompletedProcess[str]:
    """Generate proof for the witness.

    Args:
        circuit_name (str): Name of the circuit.
        circuit_path (str): Path to the circuit definition.
        witness_file (str): Path to the witness file.
        proof_file (str): Path to the output proof file.
        proof_system (ZKProofSystems, optional): Proof system to use. Defaults to ZKProofSystems.Expander.
        dev_mode (bool, optional): If True, recompiles the rust binary (run in development mode). Defaults to False.
        ecc (bool, optional): If true, run proof using ECC api, otherwise run directly through Expander. Defaults to True.
        bench (bool, optional): If True, enable benchmarking. Defaults to False.

    Raises:
        NotImplementedError: If proof system is not supported.

    Returns:
        subprocess.CompletedProcess[str]: Exit message from the subprocess.
    """    
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
            }
            
            # Run the command
            try:
                return run_cargo_command(binary_name, 'run_prove_witness', args, dev_mode, bench)
            except Exception as e:
                print(f"Warning: Proof generation failed: {e}")
        else:
            return run_expander_raw(
                mode="prove",
                circuit_file=circuit_path,
                witness_file=witness_file,
                proof_file=proof_file,
                bench = bench
            )
    else:
        raise NotImplementedError(f"Proof system {proof_system} not implemented")


def generate_verification(circuit_name: str,
                          circuit_path: str,
                          input_file: str,
                          output_file: str,
                          witness_file: str,
                          proof_file: str,
                          proof_system: ZKProofSystems = ZKProofSystems.Expander,
                          dev_mode: bool = False, 
                          ecc: bool = True,
                          bench: bool = False
                          ) -> subprocess.CompletedProcess[str]:
    """Verify a given proof.

    Args:
        circuit_name (str): Name of the circuit.
        circuit_path (str): Path to the circuit definition.
        input_file (str): Path to the input JSON file with public inputs.
        output_file (str): Path to the output JSON file with expected outputs.
        witness_file (str): Path to the witness file.
        proof_file (str): Path to the output proof file.
        proof_system (ZKProofSystems, optional): Proof system to use. Defaults to ZKProofSystems.Expander.
        dev_mode (bool, optional): If True, recompiles the rust binary (run in development mode). Defaults to False.
        ecc (bool, optional): If true, run proof using ECC api, otherwise run directly through Expander. Defaults to True.
        bench (bool, optional): If True, enable benchmarking. Defaults to False.

    Raises:
        NotImplementedError: If proof system is not supported.

    Returns:
        subprocess.CompletedProcess[str]: Exit message from the subprocess.
    """
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
            }
            # Run the command
            try:
                return run_cargo_command(binary_name, 'run_gen_verify', args, dev_mode, bench)
            except Exception as e:
                print(f"Warning: Verification generation failed: {e}")
        else:
            run_expander_raw(
                mode="verify",
                circuit_file=circuit_path,
                witness_file=witness_file,
                proof_file=proof_file,
                bench = bench
            )
    else:
        raise NotImplementedError(f"Proof system {proof_system} not implemented")

def run_end_to_end(circuit_name: str,
                   circuit_path: str,
                   input_file: str,
                   output_file: str, 
                  proof_system: ZKProofSystems = ZKProofSystems.Expander,
                  demo: bool = False,
                  dev_mode: bool = False,
                  ecc: bool = True
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
        proof_system (ZKProofSystems, optional): Proof system to use. Defaults to ZKProofSystems.Expander.
        demo (bool, optional): Run Demo mode, which limits prints, to clean only. Defaults to False.
        dev_mode (bool, optional): If True, recompiles the rust binary (run in development mode). Defaults to False.
        ecc (bool, optional): If true, run proof using ECC api, otherwise run directly through Expander. Defaults to True.

    Raises:
        NotImplementedError: If proof system is not supported.

    Returns:
        int: Exit code from the verification step (0 = success, non-zero = failure).
    """    
    if proof_system == ZKProofSystems.Expander:
        base, ext = os.path.splitext(circuit_path)  # Split the filename and extension
        witness_file = f"{base}_witness{ext}"
        proof_file = f"{base}_proof{ext}"
        compile_circuit(circuit_name, circuit_path, proof_system, dev_mode)
        generate_witness(circuit_name, circuit_path, witness_file, input_file, output_file, proof_system, dev_mode)
        generate_proof(circuit_name, circuit_path, witness_file, proof_file, proof_system, dev_mode, ecc)
        return generate_verification(circuit_name, circuit_path, input_file, output_file, witness_file, proof_file, proof_system, dev_mode, ecc)
    else:
        raise NotImplementedError(f"Proof system {proof_system} not implemented")

def get_files(
    name: str,
    proof_system: ZKProofSystems,
    folders: Dict[str, str],
) -> Dict[str, str]:
    """
    Generate file paths ensuring folders exist.

    Args:
        name (str): The base name for all generated files.
        proof_system (ZKProofSystems): The ZK proof system being used.
        folders (Dict[str, str]): Dictionary containing required folder paths with keys like:
                 'input', 'proof', 'temp', 'circuit', 'weights', 'output', 'quantized_model'.

    Raises:
        NotImplementedError: If not implemented proof system is tried

    Returns:
        Dict[str, str]: A dictionary mapping descriptive keys to file paths.
    """    
    # Ensure all provided folders exist
    for path in folders.values():
        create_folder(path)

    # Common file paths
    paths = {
        "input_file": os.path.join(folders["input"], f"{name}_input.json"),
        "public_path": os.path.join(folders["proof"], f"{name}_public.json"),
        # "verification_key": os.path.join(folders["temp"], f"{name}_verification_key.json"),
        "weights_path": os.path.join(folders["weights"], f"{name}_weights.json"),
        "output_file": os.path.join(folders["output"], f"{name}_output.json"),
    }

    # Proof-system-specific files
    if proof_system == ZKProofSystems.Expander:
        paths.update({
            "circuit_name": os.path.join(folders["circuit"], name),
            "witness_file": os.path.join(folders["input"], f"{name}_witness.txt"),
            "proof_path": os.path.join(folders["proof"], f"{name}_proof.bin"),
        })
    else:
        raise NotImplementedError(f"Proof system {proof_system} not implemented")

    return paths

def create_folder(directory: str) -> None:
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
