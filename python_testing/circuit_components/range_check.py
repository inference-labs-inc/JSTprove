import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import RunType
from python_testing.circuit_components.circuit_helpers import Circuit

BATCH_SIZE = 32
UPPER_BOUND = 999

class RangeCheckDemo(Circuit):
    def __init__(self):
        self.name = "range_check_demo"
        self.vals = torch.randint(low=0, high=UPPER_BOUND + 1, size=(BATCH_SIZE,))

    def get_inputs(self):
        return {"a_vec": self.vals}

    def get_outputs(self, _: dict):
        return torch.tensor([])   # no public outputs

    def format_inputs(self, inputs):
        return {"a_vec": inputs["a_vec"].tolist()}

    def format_outputs(self, _):
        return {}

if __name__ == "__main__":
    circ = RangeCheckDemo()
    circ.base_testing(
        run_type=RunType.COMPILE_CIRCUIT,
        dev_mode=True,
        circuit_folder="",
        input_folder="inputs",
        proof_folder="proofs",
        temp_folder="temp",
        output_folder="output",
        weights_folder="weights",
        proof_system=ZKProofSystems.Expander,
    )
    circ.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=True,
        circuit_folder="",
        input_folder="inputs",
        proof_folder="proofs",
        temp_folder="temp",
        output_folder="output",
        weights_folder="weights",
        proof_system=ZKProofSystems.Expander,
        write_json=True,
    )
