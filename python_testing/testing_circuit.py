from circom.reward_fn import generate_sample_inputs
import torch
from torch import nn
import random
import matplotlib.pyplot as plt
import json
import ezkl
from typing import Dict, List, Tuple
import bittensor as bt
import os
import numpy as np
import sys
from typing import Dict
from python_testing.run_proofs import ZKProofsCircom, ZKProofsExpander, ZKProofSystems


class RewardTests():
    def __init__(self,maximum_score: torch.Tensor,
            previous_score: torch.Tensor,
            verified: torch.Tensor,
            proof_size: torch.Tensor,
            response_time: torch.Tensor,
            maximum_response_time: torch.Tensor,
            minimum_response_time: torch.Tensor,
            block_number: torch.Tensor,
            validator_uid: torch.Tensor,
            miner_uid: torch.Tensor):
        super().__init__()
        self.scaling = 100000000
        self.RATE_OF_DECAY = torch.tensor(0.4)
        self.RATE_OF_RECOVERY = torch.tensor(0.1)
        self.FLATTENING_COEFFICIENT = torch.tensor(0.9)
        self.PROOF_SIZE_THRESHOLD = torch.tensor(3648)
        self.PROOF_SIZE_WEIGHT = torch.tensor(0)
        # self.PROOF_SIZE_WEIGHT = torch.tensor(1)
        self.RESPONSE_TIME_WEIGHT = torch.tensor(1)
        self.MAXIMUM_RESPONSE_TIME_DECIMAL = torch.tensor(0.99)
        
        self.maximum_score = maximum_score
        self.previous_score = previous_score
        self.verified = verified
        self.proof_size = proof_size
        self.response_time = response_time
        self.maximum_response_time = maximum_response_time
        self.minimum_response_time = minimum_response_time
        self.block_number = block_number
        self.validator_uid = validator_uid
        self.miner_uid = miner_uid
    
    def _to_json(self, inputs: Dict[str, torch.Tensor], path: str):
        with open(path, 'w') as outfile:
            json.dump(inputs, outfile)
        
    def _read_outputs_from_json(self, public_path: str):
        with open(public_path) as json_data:
            d = json.load(json_data)
            json_data.close()
            return d
    
    def _prove_and_verify(self, witness_file, input_file, proof_path, public_path, verification_key, circuit_name, proof_system: ZKProofSystems = ZKProofSystems.Expander, output_file = None):
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

    def _get_files(self, input_folder, proof_folder, temp_folder, circuit_folder, name, output_folder, proof_system):
        self._create_folder(input_folder)
        self._create_folder(proof_folder)
        self._create_folder(temp_folder)
        self._create_folder(output_folder)
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
    
    def _create_folder(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def compare_values(self, python_value, circom_value, scaling):
        modulus = 21888242871839275222246405745257275088548364400416034343698204186575808495617
        if python_value < 0:
            circom_value = circom_value - modulus
            pass
        return abs(python_value - circom_value/scaling)<0.0000001
                                                
    
    def compare_values_ignore_case(self, python_value, circom_value):
        modulus = 21888242871839275222246405745257275088548364400416034343698204186575808495617
        if python_value < 0:
            # python_value * -1
            pass
        print(abs(abs(python_value) - abs(circom_value)))
        print((python_value - circom_value)<0.00000001)
        return abs(abs(python_value) - abs(circom_value))<0.000001
    
    def compare_values_case(self, python_value, circom_value):
        modulus = 21888242871839275222246405745257275088548364400416034343698204186575808495617
        if python_value < 0:
            # python_value * -1
            pass
        if python_value < 0:
            if circom_value == 1:
                return False
            elif circom_value == 0:
                return True
        if python_value>= 0 :
            if circom_value == 1:
                return True
            elif circom_value == 0:
                return False
        raise 

    def shifted_tan(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tan(
            torch.mul(
                torch.mul(torch.sub(x, torch.tensor(0.5)), torch.pi),
                self.FLATTENING_COEFFICIENT,
            )
        )

    def tan_shift_difference(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sub(self.shifted_tan(x), self.shifted_tan(torch.tensor(0.0)))

    def normalized_tangent_curve(self, x: torch.Tensor) -> torch.Tensor:
        return torch.div(
            self.tan_shift_difference(x), self.tan_shift_difference(torch.tensor(1.0))
        )
    
    def approximate_polynomial(self, x: torch.Tensor) -> torch.tensor:
        new_x = ((1/242000)*(20*x - 10)**5) + (1/242000*(20*x - 10)**3) + ((20/121)*x) + 0.417355371900826
        return new_x
    
    def test_reward_function(self, input_folder:str, proof_folder: str, temp_folder: str, circuit_folder:str, proof_system: ZKProofSystems, output_folder: str = None):
        name = "reward"
        witness_file, input_file, proof_path, public_path, verification_key, circuit_name, output_file = self._get_files(
            input_folder, proof_folder, temp_folder, circuit_folder, name, output_folder, proof_system)
        
        rate_of_change= torch.where(
            self.verified, self.RATE_OF_RECOVERY, self.RATE_OF_DECAY
        )

        response_time_normalized = torch.clamp(
            torch.div(
                torch.sub(self.response_time, self.minimum_response_time),
                torch.sub(self.maximum_response_time, self.minimum_response_time),
            ),
            0,
            self.MAXIMUM_RESPONSE_TIME_DECIMAL,
        )
        response_time_reward_metric = torch.mul(
            self.RESPONSE_TIME_WEIGHT,
            torch.sub(
                torch.tensor(1), self.approximate_polynomial(response_time_normalized)
            ),
        )
        # Original tan function
        # response_time_reward_metric = torch.mul(
        #     self.RESPONSE_TIME_WEIGHT,
        #     torch.sub(
        #         torch.tensor(1), self.normalized_tangent_curve(response_time_normalized)
        #     ),
        # )

        proof_size_reward_metric = torch.mul(
            self.PROOF_SIZE_WEIGHT,
            torch.clamp(
                self.proof_size / self.PROOF_SIZE_THRESHOLD, torch.tensor(0), torch.tensor(1)
            ),
        )

        calculated_score_fraction = torch.clamp(
            torch.sub(response_time_reward_metric, proof_size_reward_metric),
            torch.tensor(0),
            torch.tensor(1),
        )

        maximum_score = torch.mul(self.maximum_score, calculated_score_fraction)

        distance_from_score = torch.where(
            self.verified, torch.sub(maximum_score, self.previous_score), self.previous_score
        )
        change_in_score = torch.mul(rate_of_change, distance_from_score)
        
        new_score = torch.where(
            self.verified,
            self.previous_score + change_in_score,
            self.previous_score - change_in_score,
        )
         
        reward_fn_results = new_score

        inputs = {
            'RATE_OF_DECAY': int(self.scaling * self.RATE_OF_DECAY.item()),
            'RATE_OF_RECOVERY': int(self.scaling * self.RATE_OF_RECOVERY.item()),
            'FLATTENING_COEFFICIENT':int(self.scaling * self.FLATTENING_COEFFICIENT.item()),
            'PROOF_SIZE_WEIGHT': int(self.scaling * self.PROOF_SIZE_WEIGHT.item()),
            'PROOF_SIZE_THRESHOLD': int(self.scaling * self.PROOF_SIZE_THRESHOLD.item()),
            'RESPONSE_TIME_WEIGHT': int(self.scaling * self.RESPONSE_TIME_WEIGHT.item()),
            'MAXIMUM_RESPONSE_TIME_DECIMAL': int(self.scaling * self.MAXIMUM_RESPONSE_TIME_DECIMAL.item()),
            'maximum_score': [int(i) for i in torch.mul(self.maximum_score,self.scaling).tolist()],
            'previous_score': [int(i) for i in torch.mul(self.previous_score,self.scaling).tolist()],
            'verified': self.verified.int().tolist(),
            'proof_size': [int(i)*self.scaling for i in self.proof_size.tolist()],
            'response_time': [int(i) for i in torch.mul(self.response_time,self.scaling).tolist()],
            'maximum_response_time': [int(i) for i in torch.mul(self.maximum_response_time,self.scaling).tolist()],
            'minimum_response_time': [int(i) for i in torch.mul(self.minimum_response_time,self.scaling).tolist()],
            'block_number': self.block_number.int().tolist(),
            'validator_uid': self.validator_uid.int().tolist(),
            'miner_uid': self.miner_uid.int().tolist(),
            'scaling': self.scaling,
            }

        self._to_json(inputs, input_file)

        # Write output to json
        self._to_json(inputs, output_file)

        # #Run the circuit
        self._prove_and_verify(witness_file, input_file, proof_path, public_path, verification_key, circuit_name, proof_system, output_file)

        # # Check that the outputs match
        # output = self._read_outputs_from_json(public_path)
        # output = [int(t) for t in output]
        # x = 0
        # for i in range(len(self.block_number.tolist())):
        #     print(reward_fn_results[i].item(),output[i]/self.scaling)
        #     assert self.compare_values(reward_fn_results[i].item(),output[i], self.scaling)
        #     assert self.compare_values(self.block_number.tolist()[i], output[i+256],1)
        #     assert self.compare_values(self.miner_uid.tolist()[i], output[i+256*2],1)
        #     assert self.compare_values(self.validator_uid.tolist()[i], output[i+256*3],1)

        # print("Outputs match!")
    
if __name__ == "__main__":
    proof_system = ZKProofSystems.Expander
    proof_folder = "proofs"
    output_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    circuit_folder = ""
    inputs = generate_sample_inputs()
    reward = RewardTests(**inputs)
    reward.test_reward_function(input_folder,proof_folder, temp_folder, circuit_folder, proof_system, output_folder)

