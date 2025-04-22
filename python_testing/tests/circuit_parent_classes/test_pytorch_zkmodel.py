import pytest
from unittest.mock import patch, MagicMock
import sys
# from python_testing.utils.pytorch_helpers import 
sys.modules.pop("python_testing.utils.pytorch_helpers", None)


with patch('python_testing.utils.helper_functions.prepare_io_files', lambda f: f):  # MUST BE BEFORE THE UUT GETS IMPORTED ANYWHERE!
    from python_testing.utils.pytorch_helpers import ZKModel, RunType, ZKProofSystems



# ---------- __init__ ----------

def test_init_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        ZKModel()


# ---------- base_testing ----------

def test_base_testing_calls_parse_proof_run_type():
    class TestZK(ZKModel):
        def __init__(self):
            pass
        def parse_proof_run_type(self, *args, **kwargs):
            self.called_with = args, kwargs

    model = TestZK()
    model._file_info = {"weights": "dummy"}  # simulate decorator behavior

    model.base_testing(
        run_type=RunType.BASE_TESTING,
        witness_file="witness.wtns",
        input_file="input.json",
        proof_file="proof.json",
        public_path="public.json",
        verification_key="vk.key",
        circuit_name="test_circuit",
        output_file="out.json",
        dev_mode=True
    )

    args, kwargs = model.called_with
    assert "witness.wtns" in args
    assert RunType.BASE_TESTING in args


@patch.object(ZKModel, 'parse_proof_run_type')
def test_base_testing_uses_default_weights_path(mock_parse):
    class TestZK(ZKModel):
        def __init__(self):
            pass

    model = TestZK()
    model.base_testing(circuit_name="test_model")

    # ensure circuit_name propagates correctly
    args, kwargs = mock_parse.call_args
    assert "test_model" in args
    assert RunType.BASE_TESTING in args


# ---------- Inheritance ----------

@patch("python_testing.utils.pytorch_helpers.torch.save")
def test_inherits_save_model(mock_save):
    class TestZK(ZKModel):
        def __init__(self):
            self.model = MagicMock()
            self.model.state_dict.return_value = {"weights": 123}

    model = TestZK()
    model.save_model("zk.pt")
    mock_save.assert_called_once_with({"weights": 123}, "zk.pt")


@patch("python_testing.utils.pytorch_helpers.torch.load", return_value={"weights": 123})
def test_inherits_load_model(mock_load):
    class TestZK(ZKModel):
        def __init__(self):
            self.model = MagicMock()

    model = TestZK()
    model.load_model("zk.pt")
    model.model.load_state_dict.assert_called_once_with({"weights": 123})
