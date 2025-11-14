from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from python.core.circuit_models.generic_onnx import GenericModelONNX
from python.core.utils.helper_functions import CircuitExecutionConfig, RunType
from python.tests.onnx_quantizer_tests.layers.base import SpecType
from python.tests.onnx_quantizer_tests.layers.factory import TestLayerFactory
from python.tests.onnx_quantizer_tests.layers_tests.base_test import BaseQuantizerTest

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from python.tests.onnx_quantizer_tests.layers.base import (
        LayerTestConfig,
        LayerTestSpec,
    )


class TestE2EQuantizer(BaseQuantizerTest):
    """End-to-end tests for ONNX quantizer layers."""

    __test__ = True

    @pytest.fixture
    def temp_quantized_model(self, tmp_path: Path) -> Generator[Path, None, None]:
        """Temporary path for quantized model."""
        path = tmp_path / "quantized_model.onnx"
        yield path
        if path.exists():
            path.unlink()

    @pytest.fixture
    def temp_circuit_path(self, tmp_path: Path) -> Generator[Path, None, None]:
        """Temporary path for circuit file."""
        path = tmp_path / "circuit.txt"
        yield path
        if path.exists():
            path.unlink()

    @pytest.fixture
    def temp_witness_file(self, tmp_path: Path) -> Generator[Path, None, None]:
        """Temporary path for witness file."""
        path = tmp_path / "witness.bin"
        yield path
        if path.exists():
            path.unlink()

    @pytest.fixture
    def temp_input_file(self, tmp_path: Path) -> Generator[Path, None, None]:
        """Temporary path for input JSON file."""
        path = tmp_path / "input.json"
        yield path
        if path.exists():
            path.unlink()

    @pytest.fixture
    def temp_output_file(self, tmp_path: Path) -> Generator[Path, None, None]:
        """Temporary path for output JSON file."""
        path = tmp_path / "output.json"
        yield path
        if path.exists():
            path.unlink()

    @pytest.fixture
    def temp_proof_file(self, tmp_path: Path) -> Generator[Path, None, None]:
        """Temporary path for proof file."""
        path = tmp_path / "proof.bin"
        yield path
        if path.exists():
            path.unlink()

    @pytest.mark.e2e
    @pytest.mark.parametrize(
        "test_case_data",
        TestLayerFactory.get_test_cases_by_type(SpecType.E2E),  # type: ignore[arg-type]
        ids=BaseQuantizerTest._generate_test_id,
    )
    def test_e2e_quantize_compile_witness_prove_verify(
        self,
        test_case_data: tuple[str, LayerTestConfig, LayerTestSpec],
        temp_quantized_model: Path,
        temp_circuit_path: Path,
        temp_witness_file: Path,
        temp_input_file: Path,
        temp_output_file: Path,
        temp_proof_file: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test end-to-end flow: quantize model, compile circuit,
        generate witness, prove, and verify."""
        layer_name, config, test_spec = test_case_data

        # Skip if validation failed or test is skipped
        self._check_validation_dependency(test_case_data)
        if test_spec.skip_reason:
            pytest.skip(f"{layer_name}_{test_spec.name}: {test_spec.skip_reason}")

        if layer_name == "Constant":
            pytest.skip(f"No e2e test for layer {layer_name} yet")

        # Create original model
        original_model = config.create_test_model(test_spec)

        # Save quantized model to temp location
        import onnx  # noqa: PLC0415

        onnx.save(original_model, str(temp_quantized_model))

        # Create GenericONNX model instance
        model = GenericModelONNX(model_name=str(temp_quantized_model))

        # Step 1: Compile circuit
        model.base_testing(
            CircuitExecutionConfig(
                run_type=RunType.COMPILE_CIRCUIT,
                dev_mode=False,
                circuit_path=str(temp_circuit_path),
            ),
        )

        # Verify circuit file exists
        assert (
            temp_circuit_path.exists()
        ), f"Circuit file not created for {layer_name}.{test_spec.name}"

        # Step 2: Generate witness
        model.base_testing(
            CircuitExecutionConfig(
                run_type=RunType.GEN_WITNESS,
                dev_mode=False,
                witness_file=temp_witness_file,
                circuit_path=str(temp_circuit_path),
                input_file=temp_input_file,
                output_file=temp_output_file,
                write_json=True,
            ),
        )
        # Verify witness and output files exist
        assert (
            temp_witness_file.exists()
        ), f"Witness file not generated for {layer_name}.{test_spec.name}"
        assert (
            temp_output_file.exists()
        ), f"Output file not generated for {layer_name}.{test_spec.name}"

        # Step 3: Prove
        model.base_testing(
            CircuitExecutionConfig(
                run_type=RunType.PROVE_WITNESS,
                dev_mode=False,
                witness_file=temp_witness_file,
                circuit_path=str(temp_circuit_path),
                input_file=temp_input_file,
                output_file=temp_output_file,
                proof_file=temp_proof_file,
            ),
        )

        # Verify proof file exists
        assert (
            temp_proof_file.exists()
        ), f"Proof file not generated for {layer_name}.{test_spec.name}"

        # Step 4: Verify
        model.base_testing(
            CircuitExecutionConfig(
                run_type=RunType.GEN_VERIFY,
                dev_mode=False,
                witness_file=temp_witness_file,
                circuit_path=str(temp_circuit_path),
                input_file=temp_input_file,
                output_file=temp_output_file,
                proof_file=temp_proof_file,
            ),
        )

        # Capture output and check for success indicators
        captured = capsys.readouterr()
        stdout = captured.out
        stderr = captured.err

        assert stderr == "", "Errors occurred during e2e test for "
        f"{layer_name}.{test_spec.name}: {stderr}"

        # Check for expected success messages (similar to circuit e2e tests)
        assert (
            "Witness Generated" in stdout
        ), f"Witness generation failed for {layer_name}.{test_spec.name}"
        assert "Proved" in stdout, f"Proving failed for {layer_name}.{test_spec.name}"
        assert (
            "Verified" in stdout
        ), f"Verification failed for {layer_name}.{test_spec.name}"
