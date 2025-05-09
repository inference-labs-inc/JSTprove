import argparse
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from cli import (
    PROJECT_ROOT, parse_args, get_run_operations, find_file, load_circuit, resolve_file_paths
)
from python_testing.circuit_components.circuit_helpers import RunType


# ---------- parse_args ----------
def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["cli.py", "--circuit", "my_circuit", "--compile"])
    args = parse_args()
    assert args.circuit == "my_circuit"
    assert args.compile is True
    assert args.class_name == "SimpleCircuit"


# ---------- get_run_operations ----------
@pytest.mark.parametrize("flags,expected", [
    (["--all"], [RunType.COMPILE_CIRCUIT, RunType.GEN_WITNESS, RunType.PROVE_WITNESS, RunType.GEN_VERIFY]),
    (["--compile"], [RunType.COMPILE_CIRCUIT]),
    (["--gen_witness"], [RunType.GEN_WITNESS]),
    (["--prove"], [RunType.PROVE_WITNESS]),
    (["--verify"], [RunType.GEN_VERIFY]),
    (["--end_to_end"], [RunType.END_TO_END]),
])
def test_get_run_operations(monkeypatch, flags, expected):
    monkeypatch.setattr("sys.argv", ["cli.py", "--circuit", "foo"] + flags)
    args = parse_args()
    result = get_run_operations(args)
    assert result == expected


# ---------- find_file ----------
def test_find_file_appends_json_extension():
    result = find_file("data")
    assert result == "data.json"


@patch("cli.Path.is_file", return_value=True)
def test_find_file_returns_valid_default_path(mock_isfile):
    path = Path("inputs/some_file.json")
    result = find_file("ignored_name", default_path=path)
    assert result == PROJECT_ROOT / path
    mock_isfile.assert_called_once()


@patch("cli.Path.is_file", return_value=False)
def test_find_file_returns_filename_if_default_path_missing(mock_isfile):
    result = find_file("myfile", default_path=Path("fake/path.json"))
    assert result == "myfile.json"
    mock_isfile.assert_called_once()


def test_find_file_no_default_path():
    result = find_file("model_output")
    assert result == "model_output.json"


# ---------- resolve_file_paths ----------
@patch("cli.find_file")
def test_resolve_file_paths_with_overrides(mock_find):
    mock_find.side_effect = lambda x, default=None: Path(f"/resolved/{x}")
    input_path, output_path = resolve_file_paths("my_circuit", "input.json", "output.json", None)
    assert input_path == "/resolved/input.json"
    assert output_path == "/resolved/output.json"


@patch("cli.find_file")
def test_resolve_file_paths_with_pattern(mock_find):
    mock_find.side_effect = lambda x, default=None: Path(f"/matched/{x}")
    i, o = resolve_file_paths("cnn", None, None, "{circuit}_input.json")
    assert i == "/matched/cnn_input.json"
    assert o == "/matched/cnn_output.json"


# ---------- load_circuit ----------
@patch("cli.importlib.import_module")
def test_load_circuit_success(mock_import):
    mock_module = MagicMock()
    mock_import.return_value = mock_module
    mock_module.SimpleCircuit = lambda: "instance"
    result = load_circuit("my_mod", "SimpleCircuit")
    assert result == "instance"


@patch("cli.importlib.import_module", side_effect=ModuleNotFoundError)
def test_load_circuit_fail(mock_import):
    with pytest.raises(ValueError):
        load_circuit("bad_mod", "FakeClass")


# ---------- main ----------
@patch("cli.parse_args")
@patch("cli.list_available_circuits")
def test_main_lists_and_exits(mock_list, mock_args):
    import cli
    mock_args.return_value = argparse.Namespace(
        list_circuits=True,
        circuit=None,
        circuit_search_path=None
    )
    cli.main()
    mock_list.assert_called_once()


@patch("cli.parse_args")
@patch("cli.load_circuit")
@patch("cli.resolve_file_paths")
@patch("cli.get_run_operations", return_value=[RunType.COMPILE_CIRCUIT])
def test_main_executes_operations(mock_ops, mock_paths, mock_load, mock_args):
    import cli
    mock_args.return_value = argparse.Namespace(
        list_circuits=False,
        circuit="mycircuit",
        class_name="SimpleCircuit",
        circuit_search_path=None,
        input="in.json",
        output="out.json",
        pattern=None,
        compile=True,
        gen_witness=False,
        prove=False,
        verify=False,
        end_to_end=False,
        all=False,
        fresh_compile=True,
        circuit_path="cp.txt",
        witness=None,
        proof=None,
        ecc = None,
        bench = None
    )
    circuit_instance = MagicMock()
    mock_load.return_value = circuit_instance
    mock_paths.return_value = ("input.json", "output.json")

    cli.main()

    circuit_instance.base_testing.assert_called_once_with(
        run_type=RunType.COMPILE_CIRCUIT,
        dev_mode=True,
        circuit_path="cp.txt",
        input_file="in.json",
        output_file="out.json",
        witness_file=None,
        proof_file=None,
        ecc = None,
        bench = None
    )
