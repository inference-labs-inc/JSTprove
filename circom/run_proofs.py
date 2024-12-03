import subprocess
import logging
import sys
from typing import List
import json
import os
from abc import ABC

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

class ZKProofs(ABC):
    def compile_circuit(self):
        pass

    def compute_witness(self, witness_file:str, input_file: str):
        pass

    def proof(self, witness_path, output_proof_path, public_path):
        pass

    def verify(self, verification_key: str, public_path: str, proof_path: str):
        pass

    def proof_setup(self, verification_path:str):
        pass

class ZKProofsCircom(ZKProofs):
    def __init__(self, circuit_file: str):
        assert isinstance(circuit_file, str)
        self.circuit_file = circuit_file
        self.folder_js = f"{self.circuit_file[:-7]}_js"
        self.folder_c = f"{self.circuit_file[:-7]}_cpp"
        self.pot_file_name = "powersOfTau28_hez_final_21.ptau"
        

    def compile_circuit(self, wasm: bool = True, c: bool = True):
        executable_to_run = ["circom", self.circuit_file, "--r1cs"]
        if wasm:
            executable_to_run.append("--wasm")
        executable_to_run.append("--sym")
        if c:
            executable_to_run.append("--c")

        res = HelperFunctions.run_process(executable_to_run, die_on_error=False)
        if res.returncode == 0:
            logging.info(f"Circuit Compiled with return code: {res.returncode}")
    
    def compute_witness(self, witness_file: str, input_file: str, wasm: bool = True, c: bool = False):
        if wasm:
            ex_1 = ["node", os.path.join(self.folder_js,"generate_witness.js"),os.path.join(self.folder_js,f"{self.circuit_file[:-7]}.wasm"), input_file, witness_file]
            res = HelperFunctions.run_process(ex_1)
            if res.returncode == 0:
                logging.info(f"Generated witness in {witness_file} with return code: {res.returncode}")
        if c:
            #Currently locally got an error
            ex_1 = ["make"]
            ex_2 = [f"{self.circuit_file[:-7]}",input_file, witness_file]
            res = HelperFunctions.change_dir_run_process(self.folder_c, ex_1)
            if res.returncode == 0:
                logging.info(f"Witness -- Make complete")
            res = HelperFunctions.run_process(ex_2)
            if res.returncode == 0:
                logging.info(f"Generated witness in {witness_file} with return code: {res.returncode}")

    def proof(self, witness_path, output_proof_path, public_path):
        ex_1 = ["snarkjs", "groth16", "prove", f"{self.circuit_file[:-7]}_0001.zkey", witness_path, output_proof_path, public_path]
        res = HelperFunctions.run_process(ex_1)
        if res.returncode == 0:
            logging.info(f"Proof generated with exit code: {res.returncode}")
            try:
                logging.info(f"{res.stdout}")
            except:
                pass
            try:
                logging.info(f"{res.stderr}")
            except:
                pass
            try:
                logging.info(f"{res.output}")
            except:
                pass
            try:
                logging.info(f"{res.args}")
            except:
                pass
    
    def verify(self, verification_key, public_path, proof_path):
        ex_1 = ["snarkjs", "groth16", "verify", verification_key, public_path, proof_path]
        res = HelperFunctions.run_process(ex_1)
        if res.returncode == 0:
            logging.info(f"Verified with exit code: {res.returncode}")


    def proof_setup(self, verification_path):
        self.__powers_of_tau__(self.pot_file_name)
        self.__phase_two__(verification_path, self.pot_file_name)
    
    def __powers_of_tau__(self, pot_file_name):
        ex_1 = ["snarkjs", "powersoftau", "new", "bn128", "21", "pot12_0000.ptau", "-v"]
        comm_1 = ["aaa"]
        ex_2 = ["snarkjs", "powersoftau", "contribute", "pot12_0000.ptau", "pot12_0001.ptau", '--name="First contribution"', "-v"]
        comm_2 = ["aaa"]
        if not os.path.exists(pot_file_name):
            res = HelperFunctions.run_process_and_communicate(ex_1,comm_2)
            
        if not os.path.exists(pot_file_name):
            res = HelperFunctions.run_process_and_communicate(ex_2, comm_2)
            if res.returncode == 0:
                logging.info(f"Ran Powers of Tau")

    def __phase_two__(self, verification_path, pot_file_name):
        folder = f"{self.circuit_file[:-7]}_js"
        
        ex_1 = ["snarkjs", "powersoftau", "prepare", "phase2", "pot12_0001.ptau", pot_file_name, "-v"]
        ex_2 = ["snarkjs", "groth16", "setup", f"{self.circuit_file[:-7]}.r1cs", pot_file_name, f"{self.circuit_file[:-7]}_0000.zkey"]
        #May need contribution here
        ex_3 = ["snarkjs", "zkey", "contribute", f"{self.circuit_file[:-7]}_0000.zkey", f"{self.circuit_file[:-7]}_0001.zkey", '--name="1st Contributor Name"', "-v"]
        ex_4 = ["snarkjs", "zkey", "export", "verificationkey", f"{self.circuit_file[:-7]}_0001.zkey", verification_path]
        if not os.path.exists(pot_file_name): 
            res = HelperFunctions.run_process(ex_1)
            if res.returncode == 0:
                logging.info(f"Start Generation of Phase 2 with exit code: {res.returncode}")
        res = HelperFunctions.run_process(ex_2)
        if res.returncode == 0:
            logging.info(f"Generate zkey with exit code: {res.returncode}")
            logging.info(f"{res.args}")
            try:
                logging.info(f"{res.stderr}")
            except:
                pass
        res = HelperFunctions.run_process_and_communicate(ex_3,"a")
        if res.returncode == 0:
            logging.info(f"Contribute to phase 2 with exit code: {res.returncode}")
        res = HelperFunctions.run_process(ex_4)
        if res.returncode == 0:
            logging.info(f"Export verification key with exit code: {res.returncode}")

class ZKProofsExpander(ZKProofs):
    def __init__(self, circuit_file: str):
        assert isinstance(circuit_file, str)
        self.circuit_file = circuit_file
    
    def compile_circuit(self):
        pass

    def compute_witness(self, witness_file:str, input_file: str):
        pass

    def proof(self, witness_path, output_proof_path, public_path):
        pass

    def verify(self, verification_key: str, public_path: str, proof_path: str):
        pass

    def proof_setup(self, verification_path:str):
        pass
        
        





        

class HelperFunctions():
    @staticmethod
    def run_process(executable: List[str], die_on_error:bool =True, shell = False):
        try:
            res = subprocess.run(executable, check=True, capture_output=True, shell=shell)
            return res
        except subprocess.CalledProcessError as err:
            logging.error(f"{err} {err.stderr.decode('utf8')}")
            # logging.error(f"ARGS {err.args}")
            logging.error(err.output.decode('utf8'))
            if die_on_error:
                sys.exit()
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
        input_data = HelperFunctions.read_json(input_file)

        for folder in folders:
            file = os.path.join(folder, "input.json")
            HelperFunctions.write_json(file, input_data)


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

    witness_file = "witness.wtns"
    input_file = "input.json"
    proof_path = "proof.json"
    public_path = "public.json"
    verification_key = "verification_key.json"
    pot_file_name = "powersOfTau28_hez_final_21.ptau"


    circuit = ZKProofsCircom("multiplier2.circom")
    res = circuit.compile_circuit()
    # HelperFunctions.generate_input(input_file, circuit.circuit_file)
    circuit.compute_witness(witness_file,input_file, wasm = True, c = False)
    circuit.proof_setup(verification_key)
    circuit.proof(witness_file,proof_path, public_path)
    circuit.verify(verification_key, public_path, proof_path)
