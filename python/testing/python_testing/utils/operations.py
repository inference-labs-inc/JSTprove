from python.testing.python_testing.circuit_models.simple_circuit import SimpleCircuit
from python.testing.python_testing.circuit_components.circuit_helpers import RunType

def test_independent_operations():
    """Test that operations can be called independently."""
    
    print("\n=== Testing independent operations ===")
    
    # Create a circuit instance
    circuit = SimpleCircuit()
    
    # Test compile operation independently
    print("\n1. Testing compile operation:")
    circuit.compile()
    
    # Test generate_witness operation independently
    print("\n2. Testing generate_witness operation:")
    circuit.generate_witness()
    
    # Test generate_verification operation independently
    print("\n3. Testing generate_verification operation:")
    circuit.generate_verification()
    
    # Test run_proof operation independently
    print("\n4. Testing run_proof operation:")
    circuit.run_proof()
    
    # Test run_end_to_end operation independently
    print("\n5. Testing run_end_to_end operation:")
    circuit.run_end_to_end()
    
    print("\n=== Independent operations test completed ===")

def test_multiple_circuits():
    """Test that multiple circuits can be run independently."""
    
    print("\n=== Testing multiple circuits ===")
    
    # Create two circuit instances with different inputs
    circuit1 = SimpleCircuit()
    circuit1.input_a = 10
    circuit1.input_b = 20
    circuit1.name = "circuit1"
    
    circuit2 = SimpleCircuit()
    circuit2.input_a = 30
    circuit2.input_b = 40
    circuit2.name = "circuit2"
    
    # Get outputs for both circuits
    print("\n1. Computing outputs for both circuits:")
    output1 = circuit1.get_outputs()
    output2 = circuit2.get_outputs()
    print(f"   Circuit1 output: {output1}")
    print(f"   Circuit2 output: {output2}")
    
    # Run operations on each circuit
    print("\n2. Running operations on circuit1:")
    circuit1.base_testing(RunType.COMPILE_CIRCUIT)
    
    print("\n3. Running operations on circuit2:")
    circuit2.base_testing(RunType.COMPILE_CIRCUIT)
    
    print("\n=== Multiple circuits test completed ===")

if __name__ == "__main__":
    test_independent_operations()
    test_multiple_circuits()