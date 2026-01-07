# TestSpecBuilder Guide

The `TestSpecBuilder` class provides an API for constructing test specifications in a maintainable and scalable way. It is designed to assist in designing single layer onnx models for running a variety of tests on in a manner that is easily repeatable

This builder pattern ensures test cases are:

* Easy to read and write
* Consistently structured
* Flexible with overrides and metadata
* Uniform in the tests being performed (Additional tests can easily be created as well for custom situations)

All supported onnx layers should be tested within this framework

---

## Default Configuration and Layer Context

Each test specification is based on a **default layer configuration** provided by a corresponding `LayerConfigProvider` class (e.g., `ConvConfigProvider`).

The `get_config()` method of a provider defines the base layer setup:

* Default inputs
* Default attributes
* Default initializers

When writing tests, you override parts of this configuration using the builder methods. The default acts as the baseline for meaningful mutations.

### Example: Conv Layer Provider

```python
class ConvConfigProvider(BaseLayerConfigProvider):
    def get_config(self) -> LayerTestConfig:
        return LayerTestConfig(
            op_type="Conv",
            valid_inputs=["input", "conv_weight", "conv_bias"],
            valid_attributes={
                "strides": [1, 1],
                "kernel_shape": [3, 3],
                "dilations": [1, 1],
                "pads": [1, 1, 1, 1]
            },
            required_initializers={
                "conv_weight": np.random.randn(32, 16, 3, 3),
                "conv_bias": np.random.randn(32)
            },
            min_opset = 12,
            max_opset = 18
        )
```

### Why This Matters

By building on a shared base:

* You only specify **what changes** you want to make in the test
* Tests are more concise and easier to reason about
* Errors become clearer when tests deviate from the valid configuration



## Example Usage

```python
valid_test("basic")
    .description("Basic 2D convolution")
    .tags("basic", "2d")
    .build()

valid_test("different_padding")
    .description("Convolution with different padding")
    .override_attrs(pads=[2, 2, 2, 2], kernel_shape=[5, 5])
    .override_initializer("conv_weight", np.random.randn(32, 16, 5, 5))
    .tags("padding", "5x5_kernel")
    .build()

error_test("conv3d_unsupported")
    .description("3D convolution should raise error")
    .override_attrs(
        kernel_shape=[3, 3, 3],
        strides=[1, 1, 1],
        dilations=[1, 1, 1],
        pads=[1, 1, 1, 1, 1, 1]
    )
    .override_initializer("conv_weight", np.random.randn(32, 16, 3, 3, 3))
    .min_opset(13)
    .expects_error(InvalidParamError, "3D Conv is not currently supported")
    .tags("3d", "unsupported")
    .build()
```

---

## Builder Lifecycle

1. **Start with a test type**: Use `valid_test`, `error_test`, `e2e_test` or `edge_case_test`.
2. **Configure the spec** using builder methods like `.description()`, `.override_attrs()`, `.expects_error()`, etc.
3. **Finalize** the test using `.build()` to generate a `TestSpec`.

---

## Class: `TestSpecBuilder`

```python
TestSpecBuilder(name: str, spec_type: SpecType)
```

Constructs a new builder for a test case.

### Parameters:

* `name` – Name of the test case.
* `spec_type` – Type of the test (VALID, ERROR, E2E, or EDGE\_CASE).

---

## Builder Methods

All methods return `self` to allow method chaining (fluent interface). The default layer will be specified in the config file for the function. Any override specified using the functions below will override an aspect of the default layer

### `.description(desc: str)`

Sets a human-readable description for the test.

### `.override_attrs(**attrs)`

Overrides attributes in the test spec, e.g., `pads`, `kernel_shape`, `strides`, etc. to test new configurations.

### `.override_initializer(name: str, data: np.ndarray)`

Overrides an initializer (e.g., weights or biases) by providing a new tensor under the given name. This is useful as we may want to change the shape of the tensor for our new test.

### `.override_inputs(*inputs: str)`

Overrides input names for the test case (if needed).

### `.expects_error(error_type: type, match: str = None)`

Used only with `SpecType.ERROR`. Specifies the expected error type and optional error message match.

Tests will fail if called on a non-error spec.

### `.tags(*tags: str)`

Adds tags to help categorize or filter tests (e.g., "2d", "edge", "unsupported"). For specific or custom tests.

### `.skip(reason: str)`

Marks the test to be skipped with a given reason (e.g., "not yet supported").

### `.min_opset(opset: int)`

Sets the minimum opset version required for the test.

### `.max_opset(opset: int)`

Sets the maximum opset version required for the test.

### `.skip(reason: str)`

Marks the test to be skipped with a given reason (e.g., "not yet supported").

### `.build() -> TestSpec`

Validates and finalizes the `TestSpec`. Must be called to get the complete test definition.

<!-- Raises `ValueError` if:

* `SpecType.ERROR` is used but `.expects_error()` was not called. -->

---

## Convenience Functions

These functions simplify starting a test builder for a specific test type:

```python
valid_test(name: str) -> TestSpecBuilder
error_test(name: str) -> TestSpecBuilder
edge_case_test(name: str) -> TestSpecBuilder
```

* `valid_test` – For regular expected-to-pass tests.
* `error_test` – For tests where an error is the expected outcome.
* `edge_case_test` – For edge cases (e.g., minimal input, boundary behavior).
* `e2e_test` - For tests designed to test the end-2-end process of the layer provided, including

---

## Best Practices

* Always give your test a **clear name** and **description**.
* Use **tags** to group related tests or features.
* When using `error_test`, **always** call `.expects_error()` to specify the expected exception.
* Use `.override_attrs()` and `.override_initializer()` to tailor the test setup precisely.
* Chain methods fluently to keep test definitions compact and readable.
* Call `.build()` last to finalize the test object.

Note, each layer will be checked using `onnx.checker.check_model` to ensure that the layer is a valid onnx layer before the rest of the tests are run. If the layer fails this check, a suitable error will be thrown to help fix the layer, and the rest of the tests will be skipped
