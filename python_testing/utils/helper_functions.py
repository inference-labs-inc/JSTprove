import torch
import json
from typing import Dict
import os
from typing import Dict
from python_testing.utils.run_proofs import ZKProofsCircom, ZKProofsExpander, ZKProofSystems

def to_json(inputs: Dict[str, torch.Tensor], path: str):
    with open(path, 'w') as outfile:
        json.dump(inputs, outfile)
    
def read_outputs_from_json(public_path: str):
    with open(public_path) as json_data:
        d = json.load(json_data)
        json_data.close()
        return d

def prove_and_verify(witness_file, input_file, proof_path, public_path, verification_key, circuit_name, proof_system: ZKProofSystems = ZKProofSystems.Expander, output_file = None):
    if proof_system == ZKProofSystems.Expander:
        assert(output_file is not None, "Output_path must be specified")
        circuit = ZKProofsExpander(circuit_name)
        circuit.run_proof(input_file, output_file)

    elif proof_system == ZKProofSystems.Circom:
        circuit = ZKProofsCircom(circuit_name)
        res = circuit.compile_circuit()
        circuit.compute_witness(witness_file,input_file, wasm = True, c = False)
        circuit.proof_setup(verification_key)
        circuit.proof(witness_file,proof_path, public_path)
        circuit.verify(verification_key, public_path, proof_path)

def get_files(input_folder, proof_folder, temp_folder, circuit_folder, name, output_folder, proof_system):
    create_folder(input_folder)
    create_folder(proof_folder)
    create_folder(temp_folder)
    create_folder(output_folder)
    # self._create_folder(circuit_folder)

    witness_file = os.path.join(temp_folder,f"{name}_witness.wtns")
    input_file = os.path.join(input_folder,f"{name}_input.json")
    proof_path = os.path.join(proof_folder,f"{name}_proof.json")
    public_path = os.path.join(proof_folder,f"{name}_public.json")
    verification_key = os.path.join(temp_folder,f"{name}_verification_key.json")
    if proof_system == ZKProofSystems.Circom:
        circuit_name = os.path.join(circuit_folder,f"{name}.circom")
    elif proof_system == ZKProofSystems.Expander:
        circuit_name = os.path.join(circuit_folder,f"{name}")
    else:
        raise NotImplementedError

    output_file = os.path.join(output_folder,f"{name}_output.json")
    return witness_file,input_file,proof_path,public_path,verification_key,circuit_name, output_file

def create_folder(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        