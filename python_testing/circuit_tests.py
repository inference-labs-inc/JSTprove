from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.convolution import Convolution, QuantizedConv
from python_testing.gemm import Gemm, QuantizedGemm
from python_testing.matrix_addition import MatrixAddition
from python_testing.matrix_hadamard_product import MatrixHadamardProduct
from python_testing.matrix_multiplication import MatrixMultiplication, QuantizedMatrixMultiplication
from python_testing.relu import ReLU, ConversionType
from python_testing.scaled_matrix_product_sum import ScaledMatrixProductSum
from python_testing.scaled_matrix_product import ScaledMatrixProduct






if __name__ == "__main__":
    proof_system = ZKProofSystems.Expander
    proof_folder = "analysis"
    output_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    weights_folder = "weights"
    circuit_folder = ""
    #Rework inputs to function
    print("Testing Convolution")
    test_circuit = Convolution()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    print("Testing Quantized Convolution")
    test_circuit = QuantizedConv()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)


    print("Testing gemm")
    test_circuit = Gemm()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    # print("Testing Quantized Gemm")
    # test_circuit = QuantizedGemm()
    # test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    print("Testing Matrix Addition")
    test_circuit = MatrixAddition()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    print("Testing Matrix Hadamard Product")
    test_circuit = MatrixHadamardProduct()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    print("Testing Matrix Multiplication")
    test_circuit = MatrixMultiplication()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    print("Testing Matrix Multiplication Quantized")
    test_circuit = QuantizedMatrixMultiplication()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    print("Testing ReLU")
    test_circuit = ReLU(ConversionType.TWOS_COMP)
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    print("Testing ReLU Dual")
    test_circuit = ReLU(ConversionType.DUAL_MATRIX)
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    print("Testing Scaled Matrix Product Sum")
    test_circuit = ScaledMatrixProductSum()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    print("Testing Scaled Matrix Product")
    test_circuit = ScaledMatrixProduct()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

