### Code Refactoring Task

**Objective:**  
Refactor the codebase to improve organization and separation of concerns.

**Implementation Details:**

subtask 1. Python - specify files for inputs and create outputs in python
Create_outputs in python should also write to files
Generate witness should contain paths to inputs and outputs files

subtask 2. 
Separate python functions so output computation comes first, then the rest of the run is separate from output computation

**Files to Modify:**
1. `python/testing/core/simple_circuit.py`
2. `utils/helper_functions.py`
3. `circuit_components/circuit_helpers.py`

**Key Requirements:**
1. **Input/Output Management:**
   - Create dedicated functions for generating and managing input/output files
   - Ensure output computation is written to files
   - Include file paths in witness generation

2. **Function Separation:**
   - Isolate output computation from the main execution flow
   - Create distinct functions for different operations



**Implementation Approach:**
- Consider using the decorator pattern for better code organization
- Explore other suitable design patterns if applicable

**Testing:**
- Verify changes by running `simple_circuit.py`
- Ensure all functionality remains intact after refactoring

**Implementation Notes:**
- Maintain clear separation between file operations and core logic
- Ensure code is scalable and maintainable
- Keep the existing functionality while improving code structure

<!-- - what to do?

In helper_functions.py, create a function that generates input and output files and the second one for easily and scalably parse proofs.
I want to have simple_circuit.py and helper_functions.py to clean up and make them in 2 differenct functions.
 one for input_output file generation and the other one for the parsingproofs. both of these are in helper_functions.py in utils subdirectory.
  to test it run the simple_circuilt.py to test it out. -->

<!-- 
- Which files to  Alter?

1. python/testing/core/simple_circuit.py
2. utils/helper_function.py -->

<!-- 
- what is the best way to do this?

subtask 1. Python - specify files for inputs and create outputs in python
Create_outputs in python should also write to files
Generate witness should contain paths to inputs and outputs files

subtask 2. 
Separate python functions so output computation comes first, then the rest of the run is separate from output computation

 
- how to do this? 

  I think using the decorator pattern is the best way to do this.
  other design pattern that might be useful in this case is ?


  - testing?

run simple_circuit.py -->


**The Problem in Current Code**

1. Duplicate Computation: Every time base_testing() is called, it recalculates outputs = self.get_outputs() even if it's the exact same calculation
2. Fixed Structure: All the code is inside one method (base_testing), making it harder to modify just parts of the workflow
3. No Caching: There's no way to avoid recomputing outputs when calling different run types


**How Decorators Solve These Problems**

1. Selective Enhancement: Adding functionality to specific methods without changing their core logic
2. Automatic Caching: Ensuring get_outputs() runs only once per circuit instance
3. Cleaner Separation: Handling file I/O separately from business logic


# ZK Circuit Testing Framework Improvements

## Overview
This project enhances the ZK Circuit testing framework using Python decorators to optimize performance and improve workflow flexibility. The implementation addresses two key requirements:

1. Specifying input/output files in Python and ensuring witness generation has access to these paths
2. Separating output computation from the rest of the processing flow

## Implementation Approach

### Core Decorators

#### 1. `@compute_and_store_output` Decorator
Applied to the `get_outputs()` method to ensure outputs are computed only once per circuit instance:

```python
@compute_and_store_output
def get_outputs(self):
    # Expensive computation happens here
    return result
```

This decorator:
- Checks for cached output in the temp folder
- Computes output only if not cached
- Stores result in a cache file for future use
- Returns cached output on subsequent calls

#### 2. `@prepare_io_files` Decorator
Applied to the `base_testing()` method to handle file setup:

```python
@prepare_io_files
def base_testing(self, run_type=RunType.BASE_TESTING, ...):
    # Run the appropriate operation
    self.parse_proof_run_type(...)
```

This decorator:
- Creates necessary directories
- Generates consistent file paths
- Calls `get_outputs()` to compute outputs (with caching)
- Gets model parameters and writes to files
- Stores file paths for later use

### Requirement 1: Python File Handling

The implementation:
- Creates structured input files via `prepare_io_files` decorator
- Computes outputs in Python through the decorated `get_outputs()` method
- Automatically writes outputs to files using `to_json(outputs, output_file)`
- Ensures witness generation receives paths to both input and output files

### Requirement 2: Separating Output Computation

The implementation:
- Uses the `@compute_and_store_output` decorator to ensure outputs are computed first
- Caches results to prevent redundant computation
- Separates output computation from file preparation and proof operations
- Allows operation methods (`compile()`, `generate_witness()`, etc.) to be called independently

## Benefits

1. **Performance Optimization**: Outputs are computed only once per circuit instance
2. **Modular Design**: Each component has a single responsibility
3. **Workflow Flexibility**: Operations can be called in any order
4. **Maintainability**: Core logic is separate from cross-cutting concerns
5. **Error Resilience**: File operations continue even if cargo operations fail

## Example Usage

```python
# Create a circuit
circuit = SimpleCircuit()

# Compute outputs (happens once, cached for future use)
output = circuit.get_outputs()

# Run independent operations
circuit.compile()
circuit.generate_witness()
circuit.run_proof()

# Or run through base_testing
circuit.base_testing(RunType.COMPILE_CIRCUIT)
```

This implementation successfully separates output computation from proof operations while ensuring all necessary files are properly managed, meeting both key requirements.

## Original Requirements
[Include original requirements from task file here]

## Testing
1. Activate the virtual environment:
```bash
source venv/bin/activate
```

2. Run the test circuit:
```bash
python -m python/testing/corecore.simple_circuit
```

3. Verify that outputs are computed only once and files are generated correctly.