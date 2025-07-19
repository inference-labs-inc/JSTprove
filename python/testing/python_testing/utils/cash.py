import os
import json
from python.testing.python_testing.circuit_models.simple_circuit import SimpleCircuit
from python.testing.python_testing.circuit_components.circuit_helpers import RunType

def test_output_caching():
    """Test that outputs are computed only once and cached correctly."""
    
    # Create a circuit instance
    print("\n=== Testing output caching ===")
    circuit = SimpleCircuit()
    
    # Make sure temp directory exists and cache doesn't
    temp_dir = "temp"
    cache_file = os.path.join(temp_dir, f"{circuit.name}_output_cache.json")
    
    # Remove any existing cache file to ensure fresh test
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Removed existing cache file: {cache_file}")
    
    # First call - should compute and cache
    print("\n1. First call to get_outputs() - should compute:")
    output1 = circuit.get_outputs()
    print(f"   Output: {output1}")
    
    # Verify cache file exists
    print(f"\n   Checking for cache file: {cache_file}")
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_value = json.load(f)
        print(f"   Cache file found with value: {cached_value}")
    else:
        print(f"   ERROR: Cache file not found!")
    
    # Second call - should use cache
    print("\n2. Second call to get_outputs() - should use cache:")
    output2 = circuit.get_outputs()
    print(f"   Output: {output2}")
    
    # Run operations and ensure cache is used
    print("\n3. Running operations - should use cache for get_outputs():")
    circuit.base_testing(RunType.COMPILE_CIRCUIT)
    
    # Modify the cache file to test it's being used
    print("\n4. Modifying cache file to test it's being used:")
    test_value = 999
    with open(cache_file, 'w') as f:
        json.dump(test_value, f)
    print(f"   Cache file modified to: {test_value}")
    
    # Get outputs again - should use modified cache
    print("\n5. Call get_outputs() after modifying cache - should not use modified value:")
    output3 = circuit.get_outputs()
    print(f"   Output: {output3}")
    
    # Cleanup
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"\nRemoved test cache file: {cache_file}")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_output_caching()