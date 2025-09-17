# Errors Guide
_For Python frontend + Rust backend codebase_

This document begins by explaining the errors raised by the Rust backend. Each section explains what the error means and the context in which it typically arises.

---

## 1. Layer Errors (`LayerError`)
Layer-specific validation failures.

| Error | Meaning |
|-------|---------|
| **MissingInput** | A required input tensor is absent when constructing the layer. |
| **ShapeMismatch** | The layer received input(s) with incompatible dimensions. |
| **MissingParameter** | A parameter required by the layer was not supplied. |
| **InvalidParameterValue** | The parameter exists but has an invalid or out-of-range value. |
| **UnsupportedConfig** | The layer configuration specifies a combination not currently supported by the circuit building backend. |
| **InvalidShape** | The shape defined for the layer is structurally invalid. |
| **UnknownOp** | An operator type in the architecture is unrecognized. |
| **Other** | A catch-all for miscellaneous issues, providing context in the attached message.

---

## 2. Array Conversion Errors (`ArrayConversionError`)
These errors occur when converting data structures into array representations.

| Error | Meaning |
|-------|---------|
| **InvalidArrayStructure** | The array provided does not match the expected dimensionality. |
| **InvalidNumber** | An array element is not numeric and thus cannot be parsed into a tensor. This indicates ill-formed numeric data. |
| **ShapeError** | A shape mismatch was detected while reshaping or aligning arrays. |
| **InvalidAxis** | An operation attempted to index along an axis that does not exist. |

---

## 3. Pattern Errors (`PatternError`)
These are **graph optimization errors** that occur when applying JSTprove's optimization rules to the computational graph.

| Error | Meaning |
|-------|---------|
| **InconsistentPattern** | The optimizer detected a discrepancy between the expected and actual structural patterns in the graph. This can break optimization passes. |
| **OutputMismatch** | The optimization pass produced a different set of outputs than expected. This usually points to semantic inconsistencies in graph rewriting. |
| **EmptyMatch** | The optimizer attempted to apply a pattern, but no corresponding layers were found. |
| **EmptyPattern** | An optimization pattern with no content was attempted to be constructed. |

---

## 4. Rescale Errors (`RescaleError`)
These occur during **numeric scaling and quantization operations**, typically in fixed-point or modular arithmetic.

| Error | Meaning |
|-------|---------|
| **ScalingExponentTooLargeError** | The scaling exponent is beyond what the target numeric type can represent. This prevents correct fixed-point scaling. |
| **ShiftExponentTooLargeError** | The bit-shift operation required exceeds the numeric type’s limits. This may indicate overly aggressive scaling parameters for the given model. |
| **BitDecompositionError** | An integer could not be decomposed into the requested number of bits. This usually indicates that a number is too large in the intermediate computation of the model. |
| **BitReconstructionError** | Reassembling an integer from its bit components failed. This usually indicates that a number is too large in the intermediate computation of the model. |

---

## 5. Build Errors (`BuildError`)
These errors occur during the **construction of layers and circuits** from definitions.

| Error | Meaning |
|-------|---------|
| **PatternMatcher** | A graph pattern used in circuit construction failed to apply. |
| **UnsupportedLayer** | A layer type is not recognized or has not been implemented. This indicates missing backend support for a frontend-requested feature. |
| **LayerBuild** | A failure occurred while constructing an individual layer. This is often due to invalid parameters, missing inputs, or unsupported configurations. |

---

## 6. Utils Errors (`UtilsError`)
These are utility errors and can be a part different parts of the backend, where parameters, inputs, and values are validated before being used in computation.

| Error | Meaning |
|-------|---------|
| **MissingParam** | A required parameter for a given layer is absent in the provided configuration. |
| **MissingInput** | The layer has been constructed without a required input connection. This usually surfaces when wiring up a computational graph and indicates an incomplete or inconsistent graph definition. |
| **ParseError**  | The backend attempted to parse a parameter from JSON but failed. |
| **ValueConversionError** | The system attempted to cast or convert a value into a type that is not representable (e.g., string → number, float → integer).|
| **InputDataLengthMismatch** | The number of inputs provided to a layer does not match its formal requirements. |
| **MissingTensor** | A weight tensor referenced by a layer cannot be found in the weight map. This usually signals that the weights file is incomplete or mismatched against the architecture definition. |
| **ValueTooLarge** | An integer or bitstring exceeds the maximum size supported by the backend for a given type. |
| **InvalidNumber**  | A JSON field was expected to be numeric but turned out to be a different type. |
| **GraphPatternError / ArrayConversionError / RescaleError / BuildError** | These errors are re-thrown from deeper subsystems and point to in the circuit graph. See the sections above. |

---

## 7. Circuit Errors (`CircuitError`)
High-level errors related to the entire circuit or architecture definition.

| Error | Meaning |
|-------|---------|
| **Layer** | Propagated `LayerError`. Indicates a problem inside a specific layer. |
| **UtilsError** | Propagated `UtilsError`. See the utils section for details. |
| **InvalidWeightsFormat** | The weights file could not be parsed as valid JSON or did not match expected schema. |
| **EmptyArchitecture** | No layers were provided in the architecture definition. |
| **GraphPatternError / ArrayConversionError / RescaleError / BuildError** | Errors bubbled up from subsystems (graph, array, scaling, build). |
| **Other** | A catch-all for miscellaneous circuit-level issues. |

---

## 8. CLI & Runtime Errors (`CliError`, `RunError`)
These are errors surfaced when interacting via the **command-line interface** or when executing circuits.

| Error | Meaning |
|-------|---------|
| **MissingArgument** | A required CLI argument was omitted. |
| **UnknownCommand** | An unrecognized CLI command was issued. |
| **Io** | A file or I/O operation failed (missing file, permissions, etc.). |
| **Json** | A JSON deserialization error occurred at runtime, often due to malformed input. |
| **Compile** | Circuit compilation failed. |
| **Serialize / Deserialize** | Errors in serializing or deserializing types. |
| **Witness** | Witness generation failed. |
| **Prove / Verify** | Proof generation or verification failed. |
| **Other** | Miscellaneous runtime errors. |

---

# Python Errors

## 1. CLI Errors (`CLIError`)

High-level errors raised when interacting via Python CLI utilities.

| Error      | Meaning |
|------------|---------|
| **CLIError** | Base exception for CLI-related errors. Indicates improper arguments, bad commands, or other user-facing CLI issues. Also wrapping verious other errors called through the cli.|

---

## 2. Circuit Definition & Configuration Errors (`CircuitError` and subclasses)

Errors related to **circuit definition, validation, and configuration**.

| Error | Meaning |
|-------|---------|
| **CircuitError** | Base class for all circuit-related errors. Usually wraps a human-readable message and optional structured details. |
| **CircuitConfigurationError** | Raised when the base circuit class is misconfigured (e.g., missing or invalid attributes such as input shape or scaling factor). Provides a list of missing attributes if known. |
| **CircuitInputError** | Raised when input validation fails. |
| **CircuitRunError** | Raised when a high-level circuit operation (compile, prove, verify, etc.) fails. |
| **CircuitFileError** | Raised when file-related circuit operations (read, write, access) fail. |
| **CircuitProcessingError** | Raised when data processing operations fail (e.g., scaling, reshaping, tensor manipulation). |

---

## 3. Quantization Errors (`QuantizationError` and subclasses)

Errors raised during **model quantization** — typically when converting or preparing ONNX models for backend execution.

| Error | Meaning |
|-------|---------|
| **QuantizationError** | Base class for all quantization errors. Always includes a generic JSTprove support message plus a specific description of the issue. |
| **InvalidParamError** | Raised when a node contains invalid or unsupported parameters during quantization. |
| **UnsupportedOpError** | Raised when an unsupported ONNX operation is encountered during quantization. Suggests reviewing supported layers. |
| **MissingHandlerError** | Raised when no registered handler exists for a particular operator type during quantization. |
| **InitializerNotFoundError** | Raised when a required initializer (typically weight or bias tensor) for a node is missing from the initializer map. |
| **HandlerImplementationError** | Raised when a quantization handler does not conform to the expected interface (e.g., missing `quantize` method, wrong return type). |
| **InvalidGraphError** | Raised when the ONNX graph is malformed or missing critical information. |
| **InvalidConfigError** | Raised when the quantization configuration itself is invalid or unsupported (e.g., bad scaling parameters or global settings). |

---

## 4. Model Conversion Errors (`ModelConversionError` and subclasses)

Errors raised while **loading, saving, analyzing, or converting** models (usually ONNX).

| Error | Meaning |
|-------|---------|
| **ModelConversionError** | Base class for all model conversion errors. Includes the model type and optional context. |
| **ModelLoadError** | Raised when an ONNX model cannot be loaded (e.g., file missing, corrupted, or incompatible). |
| **ModelSaveError** | Raised when saving a model to disk fails. |
| **InferenceError** | Raised when ONNX Runtime inference fails. May include the model path and reason. |
| **LayerAnalysisError** | Raised when analyzing layers of a model fails (e.g., unsupported layer format or corrupt metadata). |
| **IOInfoExtractionError** | Raised when extracting input/output tensor info from a model fails. |
| **InvalidModelError** | Raised when the ONNX model fails validation (`onnx.checker`) or other structural checks. |
| **SerializationError** | Raised when model data (e.g., tensors) cannot be serialized to the required format. |

---

## 5. Circuit Execution Errors (`CircuitExecutionError` and subclasses)

Errors raised during **runtime execution** of the circuit (proof generation, caching, system calls).

| Error | Meaning |
|-------|---------|
| **CircuitExecutionError** | Base exception for circuit execution-related errors. |
| **MissingFileError** | Raised when a required file cannot be found. Provides optional path. |
| **FileCacheError** | Raised when reading or writing cached output fails. Provides optional path. |
| **ProofBackendError** | Raised when a backend (interfacing with proving system) command fails. Includes command, exit code, stdout, and stderr for debugging. |
| **ProofSystemNotImplementedError** | Raised when a requested proof system is not implemented. |

---

## 6. Circuit Utility Errors (`CircuitUtilsError` and subclasses)

Errors from **utility functions** used in circuit construction, file parsing, or tensor manipulation.

| Error | Meaning |
|-------|---------|
| **CircuitUtilsError** | Base class for layer utility errors. |
| **InputFileError** | Raised when reading an input file fails. Includes file path and cause if known. |
| **MissingCircuitAttributeError** | Raised when a required attribute is missing or not set in a circuit definition. |
| **ShapeMismatchError** | Raised when attempting to reshape tensors into incompatible shapes. Includes expected and actual shapes. |

---

## Debugging Strategy
1. **Identify the error enum** – this indicates which subsystem failed (utils, layers, circuit, CLI, etc.).
2. **Inspect the detailed message** – most errors embed the specific layer, param, or input name.
3. **Trace frontend → backend inputs** – many errors originate from mismatched JSON schemas or invalid data passed in from Python.
4. **Check for shape and type mismatches** – tensor dimension issues are among the most common failures.
5. **Review logs** – backend logs will often reveal deeper stack traces that explain propagation of errors.
