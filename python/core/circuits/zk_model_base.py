from typing import Optional


from python.core.circuits.base import Circuit, RunType
from python.core.utils.helper_functions import ZKProofSystems, prepare_io_files
from python.core.utils.general_layer_functions import GeneralLayerFunctions


class ZKModelBase(GeneralLayerFunctions, Circuit):
    """
    Abstract base class for Zero-Knowledge (ZK) ML models.

    This class provides a standard interface for ZK circuit ML models.
    Instantiates Circuit and GeneralLayerFunctions.

    Subclasses must implement the constructor to define the model's 
    architecture, layers, and circuit details.
    """
    def __init__(self):
        """Initialize the ZK model. Must be overridden by subclasses

        Raises:
            NotImplementedError: If called on the base class directly.
        """
        raise NotImplementedError("Must implement __init__")
    
    
    @prepare_io_files
    def base_testing(self, 
                    run_type: RunType = RunType.END_TO_END, 
                    witness_file: str = None,
                    input_file: str = None,
                    proof_file: str = None,
                    public_path: str = None, 
                    verification_key: str = None,
                    circuit_name: str = None,
                    weights_path: str = None,
                    output_file: str = None,
                    proof_system: ZKProofSystems = ZKProofSystems.Expander,
                    dev_mode: str = False,
                    ecc: str = True,
                    circuit_path: Optional[str] = None,
                    write_json: Optional[bool] = False, 
                    bench = False,
                    quantized_path = None):
        """Run the circuit in a specified mode (testing, proving, compiling, etc.).

        File path resolution is handled automatically by the `prepare_io_files` decorator.

        Args:
            run_type (RunType, optional): Type of run (compile_circuit, generate_witness, prove, verify). Defaults to RunType.END_TO_END.
            witness_file (str, optional): Path to witness file. Defaults to None.
            input_file (str, optional): Path to input JSON file. Defaults to None.
            proof_file (str, optional): Path to proof file. Defaults to None.
            public_path (str, optional): Path to public inputs file. Defaults to None.
            verification_key (str, optional): Path to verification key file. Defaults to None.
            circuit_name (str, optional): Name of the circuit. Defaults to None.
            weights_path (str, optional): Path to weights file. Defaults to None.
            output_file (str, optional): Path to output JSON file. Defaults to None.
            proof_system (ZKProofSystems, optional): Proof system identifier. Defaults to ZKProofSystems.Expander.
            dev_mode (str, optional):  Enable developer mode. Defaults to False.
            ecc (str, optional): Use ECC version of Expander for prove and verify. Defaults to True.
            circuit_path (Optional[str], optional): Path to compiled circuit file. Defaults to None.
            write_json (Optional[bool], optional): Whether to write inputs/outputs directly to JSON. Defaults to False.
            bench (bool, optional): Enable benchmarking mode. Defaults to False.
            quantized_path (_type_, optional): Path to quantized model file. Defaults to None.

        Raises:
            KeyError: If `_file_info` is not set by the decorator.
        """ 
        # TODO log
        print(f"Running circuit {run_type.value}")

        print(f"Circuit name: {circuit_name}, Circuit path: {circuit_path}")

        if not weights_path:
            weights_path = f"weights/{circuit_name}_weights.json"

        self.parse_proof_run_type(witness_file = witness_file, input_file = input_file, proof_path = proof_file, public_path = public_path, verification_key = verification_key, circuit_name = circuit_name, circuit_path = circuit_path, proof_system = proof_system, output_file = output_file, weights_path = weights_path, quantized_path =quantized_path, run_type = run_type, dev_mode = dev_mode, ecc = ecc, write_json = write_json, bench =bench)
