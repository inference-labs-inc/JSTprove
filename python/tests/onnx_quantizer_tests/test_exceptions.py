import pytest

from python.core.model_processing.onnx_quantizer.exceptions import (
    REPORTING_URL,
    InvalidParamError,
    QuantizationError,
    UnsupportedOpError,
)


@pytest.mark.unit
def test_quantization_error_message() -> None:
    custom_msg = "Something went wrong."
    with pytest.raises(QuantizationError) as exc_info:
        raise QuantizationError(custom_msg)
    assert "This model is not supported by JSTprove." in str(exc_info.value)
    assert custom_msg in str(exc_info.value)

    assert REPORTING_URL in str(exc_info.value)

    assert "Submit model support requests via the JSTprove channel:" in str(
        exc_info.value,
    )


@pytest.mark.unit
def test_invalid_param_error_basic() -> None:
    with pytest.raises(InvalidParamError) as exc_info:
        raise InvalidParamError(
            node_name="Conv_1",
            op_type="Conv",
            message="Missing 'strides' attribute.",
        )
    err_msg = str(exc_info.value)
    assert "Invalid parameters in node 'Conv_1'" in err_msg
    assert "(op_type='Conv')" in err_msg
    assert "Missing 'strides' attribute." in err_msg
    assert "[Attribute:" not in err_msg
    assert "[Expected:" not in err_msg

    msg = ""

    with pytest.raises(QuantizationError) as exc_info_quantization:
        raise QuantizationError(msg)
    # Assert contains generic error message from quantization error
    assert str(exc_info_quantization.value) in err_msg


@pytest.mark.unit
def test_invalid_param_error_with_attr_and_expected() -> None:
    with pytest.raises(InvalidParamError) as exc_info:
        raise InvalidParamError(
            node_name="MaxPool_3",
            op_type="MaxPool",
            message="Kernel shape is invalid.",
            attr_key="kernel_shape",
            expected="a list of 2 positive integers",
        )
    err_msg = str(exc_info.value)
    assert "Invalid parameters in node 'MaxPool_3'" in err_msg
    assert "[Attribute: kernel_shape]" in err_msg
    assert "[Expected: a list of 2 positive integers]" in err_msg
    msg = ""

    with pytest.raises(QuantizationError) as exc_info_quantization:
        raise QuantizationError(msg)
    # Assert contains generic error message from quantization error
    assert str(exc_info_quantization.value) in err_msg


@pytest.mark.unit
def test_unsupported_op_error_with_node() -> None:
    with pytest.raises(UnsupportedOpError) as exc_info:
        raise UnsupportedOpError(op_type="Resize", node_name="Resize_42")
    err_msg = str(exc_info.value)
    assert "Unsupported op type: 'Resize'" in err_msg
    assert "in node 'Resize_42'" in err_msg
    assert "documentation for supported layers" in err_msg
    msg = ""

    with pytest.raises(QuantizationError) as exc_info_quantization:
        raise QuantizationError(msg)
    # Assert contains generic error message from quantization error
    assert str(exc_info_quantization.value) in err_msg


@pytest.mark.unit
def test_unsupported_op_error_without_node() -> None:
    with pytest.raises(UnsupportedOpError) as exc_info:
        raise UnsupportedOpError(op_type="Upsample")
    err_msg = str(exc_info.value)
    assert "Unsupported op type: 'Upsample'" in err_msg
    assert "in node" not in err_msg
    msg = ""

    with pytest.raises(QuantizationError) as exc_info_quantization:
        raise QuantizationError(msg)
    # Assert contains generic error message from quantization error
    assert str(exc_info_quantization.value) in err_msg
