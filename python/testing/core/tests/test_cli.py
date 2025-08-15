# python/testing/core/tests/test_cli.py

from __future__ import annotations
import json
from pathlib import Path
import pytest

from python.frontend.cli import main


# ----- helpers -----

REPO_ROOT = Path(__file__).resolve().parents[4]

def _runner_exists() -> bool:
    return (REPO_ROOT / "target" / "release" / "onnx_generic_circuit").exists()

def _artifacts(tmp_path: Path, name: str = "doom") -> dict[str, Path]:
    base = tmp_path / "artifacts" / name
    base.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "circuit": base / "circuit.txt",
        "quant": base / "quantized.onnx",
        "input": REPO_ROOT / "python_testing" / "models" / "inputs" / "doom_input.json",
        "output": base / "output.json",
        "witness": base / "witness.bin",
        "proof": base / "proof.bin",
        "onnx": REPO_ROOT / "python" / "models" / "models_onnx" / "doom.onnx",
    }


# ----- unit: parser behavior (no heavy work) -----

@pytest.mark.unit
def test_unknown_subcommand_errors():
    # not an alias and abbrev disabled → argparse should SystemExit
    with pytest.raises(SystemExit):
        main(["--no-banner", "compil"])  # typo

@pytest.mark.unit
def test_known_aliases_are_accepted_but_require_args():
    # 'comp' is our alias; missing required flags should error (parser works)
    with pytest.raises(SystemExit):
        main(["--no-banner", "comp"])


# ----- integration: end-to-end flows (skip if runner missing) -----

pytestmark_integration = pytest.mark.integration
skip_no_runner = pytest.mark.skipif(not _runner_exists(), reason="runner binary not built")

@skip_no_runner
@pytestmark_integration
def test_cli_compile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # ensure reshaped input etc. land in tmp dir
    p = _artifacts(tmp_path)

    rc = main([
        "--no-banner",
        "compile",
        "-m", str(p["onnx"]),
        "-c", str(p["circuit"]),
        "-q", str(p["quant"]),
    ])
    assert rc == 0
    assert p["circuit"].exists()
    assert p["quant"].exists()


@skip_no_runner
@pytestmark_integration
def test_cli_witness(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = _artifacts(tmp_path)

    assert main(["--no-banner", "compile", "-m", str(p["onnx"]), "-c", str(p["circuit"]), "-q", str(p["quant"])]) == 0

    rc = main([
        "--no-banner",
        "witness",
        "-c", str(p["circuit"]),
        "-q", str(p["quant"]),
        "-i", str(p["input"]),
        "-o", str(p["output"]),
        "-w", str(p["witness"]),
    ])
    assert rc == 0
    assert p["witness"].exists()
    assert p["output"].exists()
    # output JSON is valid
    json.load(open(p["output"]))


@skip_no_runner
@pytestmark_integration
def test_cli_prove_and_verify(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = _artifacts(tmp_path)

    assert main(["--no-banner", "compile", "-m", str(p["onnx"]), "-c", str(p["circuit"]), "-q", str(p["quant"])]) == 0
    assert main(["--no-banner", "witness", "-c", str(p["circuit"]), "-q", str(p["quant"]),
                 "-i", str(p["input"]), "-o", str(p["output"]), "-w", str(p["witness"])]) == 0

    # prove
    assert main(["--no-banner", "prove",
                 "-c", str(p["circuit"]), "-w", str(p["witness"]), "-p", str(p["proof"])]) == 0
    assert p["proof"].exists()

    # verify (requires -q to hydrate input shapes)
    assert main(["--no-banner", "verify",
                 "-c", str(p["circuit"]), "-q", str(p["quant"]),
                 "-i", str(p["input"]), "-o", str(p["output"]),
                 "-w", str(p["witness"]), "-p", str(p["proof"])]) == 0
