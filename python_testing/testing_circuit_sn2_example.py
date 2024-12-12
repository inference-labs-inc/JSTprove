from circom.reward_fn import generate_sample_inputs
import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify


class RewardTests():
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self):
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

        inputs = generate_sample_inputs()
        
        self.maximum_score = torch.tensor(inputs["maximum_score"])
        self.previous_score = torch.tensor(inputs["previous_score"])
        self.verified = torch.tensor(inputs["verified"])
        self.proof_size = torch.tensor(inputs["proof_size"])
        self.response_time = torch.tensor(inputs["response_time"])
        self.maximum_response_time = torch.tensor(inputs["maximum_response_time"])
        self.minimum_response_time = torch.tensor(inputs["minimum_response_time"])
        self.block_number = torch.tensor(inputs["block_number"])
        self.validator_uid = torch.tensor(inputs["validator_uid"])
        self.miner_uid = torch.tensor(inputs["miner_uid"])
    
    def approximate_polynomial(self, x: torch.Tensor) -> torch.tensor:
        new_x = ((1/242000)*(20*x - 10)**5) + (1/242000*(20*x - 10)**3) + ((20/121)*x) + 0.417355371900826
        return new_x
    
    def test_reward_function(self, input_folder:str, proof_folder: str, temp_folder: str, circuit_folder:str, proof_system: ZKProofSystems, output_folder: str = None):
        # Setup
        name = "reward"
        witness_file, input_file, proof_path, public_path, verification_key, circuit_name, output_file = get_files(
            input_folder, proof_folder, temp_folder, circuit_folder, name, output_folder, proof_system)
        
        # Calculate circuit in python
        rate_of_change = torch.where(
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

        # Set inputs into json
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
        # Set output into json

        outputs = {
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

        to_json(inputs, input_file)

        # Write output to json
        to_json(outputs, output_file)

        # #Run the circuit
        prove_and_verify(witness_file, input_file, proof_path, public_path, verification_key, circuit_name, proof_system, output_file)
    
if __name__ == "__main__":
    proof_system = ZKProofSystems.Expander
    proof_folder = "proofs"
    output_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    circuit_folder = ""
    #Rework inputs to function
    reward = RewardTests()
    reward.test_reward_function(input_folder,proof_folder, temp_folder, circuit_folder, proof_system, output_folder)

