import subprocess
import logging
import sys
from typing import List
import json
import os
from abc import ABC
from enum import Enum

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

class ZKProofSystems(Enum):
    Circom = "Circom"
    Expander = "Expander"

class ZKProofsExpander():
    def __init__(self, circuit_file: str):
        assert isinstance(circuit_file, str)
        self.circuit_folder = "rust/ExpanderCompilerCollection"
        self.toml_path = f"{self.circuit_folder}/Cargo.toml"
        self.circuit_file = circuit_file



    def run_proof(self, input_file: str, output_file: str, demo = False, dev_mode = True):
        assert isinstance(input_file, str)
        assert isinstance(output_file, str)
        if dev_mode:
            executable_to_run = ["cargo", "run", "--bin", self.circuit_file, "--release", input_file, output_file]
        else:
            executable_to_run = [f"./target/release/{self.circuit_file}", input_file, output_file]
            

        res = ExecutableHelperFunctions.run_process(executable_to_run, die_on_error=True, demo=demo, cwd="rust")
        if res.returncode == 0:
            logging.info(f"Circuit Compiled with return code: {res.returncode}")

    def run_end_to_end(self, input_file: str, output_file: str, circuit_name:str,  demo = False, dev_mode = False):
        assert isinstance(input_file, str)
        assert isinstance(output_file, str)

        # Add path to toml
        if dev_mode:
            executable_to_run = ["cargo", "run", "--bin", self.circuit_file, "--release"]
        else:
            executable_to_run = [f"./target/release/{self.circuit_file}"]

        executable_to_run.append("run_proof")

        executable_to_run.append(f"-n {circuit_name}")

        executable_to_run.append("-i")
        executable_to_run.append(input_file)

        executable_to_run.append("-o")
        executable_to_run.append(output_file)

        res = ExecutableHelperFunctions.run_process(executable_to_run, die_on_error=True, demo=demo, cwd="rust")
        if res.returncode == 0:
            logging.info(f"Circuit Compiled with return code: {res.returncode}")

    def run_compile_circuit(self, circuit_name: str, demo = False, dev_mode = False):
        # Add path to toml
        if dev_mode:
            executable_to_run = ["cargo", "run", "--bin", self.circuit_file, "--release"]
        else:
            executable_to_run = [f"./target/release/{self.circuit_file}"]

        executable_to_run.append("run_compile_circuit")
        executable_to_run.append("-n")
        executable_to_run.append(circuit_name)

        res = ExecutableHelperFunctions.run_process(executable_to_run, die_on_error=True, demo=demo, cwd="rust")
        if res.returncode == 0:
            logging.info(f"Circuit Compiled with return code: {res.returncode}")

    def run_gen_witness(self, circuit_name: str, witness_name: str, input_file: str, output_file: str, demo = False, dev_mode = False):
        if dev_mode:
            executable_to_run = ["cargo", "run", "--bin", self.circuit_file, "--release"]
        else:
            executable_to_run = [f"./target/release/{self.circuit_file}"]

        executable_to_run.append("run_gen_witness")
        executable_to_run.append("-n")
        executable_to_run.append(circuit_name)

        executable_to_run.append("-i")
        executable_to_run.append(input_file)

        executable_to_run.append("-o")
        executable_to_run.append(output_file)



        res = ExecutableHelperFunctions.run_process(executable_to_run, die_on_error=True, demo=demo, cwd="rust")
        if res.returncode == 0:
            logging.info(f"Circuit Compiled with return code: {res.returncode}")

    def run_prove_witness(self, circuit_name: str, demo = False, dev_mode = False):
        if dev_mode:
            executable_to_run = ["cargo", "run", "--bin", self.circuit_file, "--release"]
        else:
            executable_to_run = [f"./target/release/{self.circuit_file}"]

        executable_to_run.append("run_prove_witness")
        executable_to_run.append("-n")
        executable_to_run.append(circuit_name)

        res = ExecutableHelperFunctions.run_process(executable_to_run, die_on_error=True, demo=demo, cwd="rust")
        if res.returncode == 0:
            logging.info(f"Circuit Compiled with return code: {res.returncode}")

    def run_gen_verify(self, circuit_name: str, demo = False, dev_mode = False):
        if dev_mode:
            executable_to_run = ["cargo", "run", "--bin", self.circuit_file, "--release"]
        else:
            executable_to_run = [f"./target/release/{self.circuit_file}"]


        executable_to_run.append("run_gen_verify")
        executable_to_run.append("-n")
        executable_to_run.append(circuit_name)




        res = ExecutableHelperFunctions.run_process(executable_to_run, die_on_error=True, demo=demo, cwd="rust")
        if res.returncode == 0:
            logging.info(f"Circuit Compiled with return code: {res.returncode}")
        
class ExecutableHelperFunctions():
    @staticmethod
    def filter_compiling_output(command):
        env = os.environ.copy()
        env["RUSTFLAGS"] = "-C target-cpu=native"
        # Run the command and get real-time output
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env = env, text=True)
        import re
        # Iterate over stdout and stderr, and print them to the command line
        for line in process.stdout:
            print(re.sub(r'\[.*?\]', '', line), end='')  # Print to the command line (stdout)
        process.wait()
        return process
    @staticmethod
    def run_process(executable: List[str], die_on_error:bool =True, shell = False, demo = False, cwd = None):
        env = os.environ.copy()
        env["RUSTFLAGS"] = "-C target-cpu=native"
        if demo:
            res = ExecutableHelperFunctions.filter_compiling_output(executable)
            return res
        try:
            # Capture output by setting capture_output=True
            if "pytest" in sys.modules:
                capture_output = True
            else:
                capture_output = False
            res = subprocess.run(executable, env = env, check=True, capture_output=capture_output, shell=shell, text=True, cwd=cwd)
            return res
        except subprocess.CalledProcessError as err:
            # Log the error message from stderr
            stderr_output = err.stderr if err.stderr else "No stderr output"
            logging.error(f"Error: {err} {stderr_output}")
            if die_on_error:
                # assert(err)
                raise
            return err
        
    @staticmethod
    def change_dir_run_process(directory:str, executable: List[str], die_on_error:bool =True, shell = False):
        wd = os.getcwd()
        os.chdir(directory)
        try:
            res = subprocess.run(executable, check=True, capture_output=True, shell=shell)
            os.chdir(wd)
            return res
        except subprocess.CalledProcessError as err:
            logging.error(f"{err} {err.stderr.decode('utf8')}")
            if die_on_error:
                sys.exit()
            os.chdir(wd)
            return err
        
    @staticmethod
    def run_process_and_communicate(executable: List[str], communication:str, die_on_error:bool =True, shell = False):
        try:
            res = subprocess.Popen(executable, encoding="Utf8")
            res.communicate(input = "a")
            return res
        except subprocess.CalledProcessError as err:
            logging.error(f"{err} {err.stderr.decode('utf8')}")
            logging.error(err.output.decode('utf8'))
            logging.debug(f"Running command: {' '.join(executable)}")
            if die_on_error:
                sys.exit()
            return err
        
    @staticmethod
    def generate_input(input_file: str, file_name: str, wasm: bool = True, c: bool = True):
        #Takes input saved in input_file and saves it in 
        folders = []
        if wasm:
            folders.append(f"{file_name[:-7]}_js")
        if c:
            folders.append(f"{file_name[:-7]}_cpp")
        input_data = ExecutableHelperFunctions.read_json(input_file)

        for folder in folders:
            file = os.path.join(folder, "input.json")
            ExecutableHelperFunctions.write_json(file, input_data)


    @staticmethod
    def read_json(file: str) -> json:
        with open(file) as f:
            d = json.load(f)
            return d
        
    @staticmethod
    def write_json(file:str, data: json):
        with open(file, 'w') as f:
            json.dump(data, f)



if __name__ == "__main__":
    pass