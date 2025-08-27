# python/frontend/cli.py

# stdlib
import argparse
import importlib
import os
import sys
from pathlib import Path

# third-party
import onnx

# local
from python.core.circuits.base import RunType
from python.core.model_processing.onnx_custom_ops.onnx_helpers import get_input_shapes

"""JSTProve CLI."""

# --- constants ---------------------------------------------------------------
DEFAULT_CIRCUIT_MODULE = "python.core.circuit_models.generic_onnx"
DEFAULT_CIRCUIT_CLASS  = "GenericModelONNX"

BANNER_TITLE = r"""
         _/    _/_/_/  _/_/_/_/_/  _/_/_/                                             
        _/  _/            _/      _/    _/  _/  _/_/    _/_/    _/      _/    _/_/    
       _/    _/_/        _/      _/_/_/    _/_/      _/    _/  _/      _/  _/_/_/_/   
_/    _/        _/      _/      _/        _/        _/    _/    _/  _/    _/          
 _/_/    _/_/_/        _/      _/        _/          _/_/        _/        _/_/_/
"""

# --- ui helpers --------------------------------------------------------------
def print_header() -> None:
    """Print the CLI banner (no side-effects at import time)."""
    print(
        BANNER_TITLE
        + "\n"
        + "JSTProve — Verifiable ML by Inference Labs\n"
        + "Based on Polyhedra Network's Expander (GKR-based proving system)\n"
    )

# --- circuit helpers ---------------------------------------------------------
def _import_default_circuit():
    """Import the default Circuit class object."""
    mod = importlib.import_module(DEFAULT_CIRCUIT_MODULE)
    try:
        return getattr(mod, DEFAULT_CIRCUIT_CLASS)
    except AttributeError as e:
        raise SystemExit(
            f"Default circuit class '{DEFAULT_CIRCUIT_CLASS}' not found in '{DEFAULT_CIRCUIT_MODULE}'"
        ) from e

def _build_default_circuit(model_name_hint: str | None = None):
    """
    Instantiate the default Circuit class.

    Some circuit subclasses require a constructor arg (commonly `model_name` or `name`).
    We try a few common constructor signatures in order, falling back gracefully:

      1) cls(model_name=<name>)
      2) cls(name=<name>)
      3) cls(<name>)              # positional
      4) cls()                    # no-arg

    Args:
        model_name_hint: Optional human-friendly name (e.g., from model filename).

    Returns:
        An instance of the default Circuit subclass.

    Raises:
        SystemExit: if none of the constructor patterns work.
    """
    cls = _import_default_circuit()
    name = (model_name_hint or "cli")

    # Try several constructor patterns commonly used in the codebase.
    for attempt in (
        lambda: cls(model_name=name),
        lambda: cls(name=name),
        lambda: cls(name),        # positional
        lambda: cls(),            # last resort
    ):
        try:
            return attempt()
        except TypeError:
            continue
    raise SystemExit(f"Could not construct {cls.__name__} with/without name '{name}'")


def _ensure_exists(path: str, kind: str = "file"):
    """
    Fail fast if a required path is missing.

    Args:
        path: Path to check.
        kind: "file" or "dir" — controls the check performed.

    Raises:
        SystemExit: if the required file/dir does not exist.
    """
    p = Path(path)
    if kind == "file" and not p.is_file():
        raise SystemExit(f"Required {kind} not found: {path}")
    if kind == "dir" and not p.is_dir():
        raise SystemExit(f"Required {kind} not found: {path}")


def _ensure_parent_dir(path: str):
    """
    Create parent directories for a file path if they don't exist.

    This is a no-op if the dirs already exist.

    Args:
        path: A file path whose parent directories should be ensured.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    """
    Entry point for the JSTProve CLI.

    Flow:
      - Parse arguments (top-level options + subcommand).
      - Optionally print the banner.
      - Dispatch to the selected subcommand:
          * compile:  model → circuit + quantized model
          * witness:  inputs → outputs.json + witness.bin
          * prove:    witness → proof.bin
          * verify:   input/output/witness/proof → verification

    Returns:
      0 on success, 1 on handled error. Unhandled SystemExit is re-raised to preserve argparse semantics.
    """
    argv = sys.argv[1:] if argv is None else argv

    # --- argparse setup ------------------------------------------------------
    parser = argparse.ArgumentParser(
        prog="jstprove",
        description="ZKML CLI (compile, witness, prove, verify).",
        allow_abbrev=False
    )
    parser.add_argument("--no-banner", action="store_true", help="Suppress the startup banner.")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # compile
    p_compile = sub.add_parser(
        "compile", aliases=["comp"],
        help="Compile a circuit (writes circuit + quantized model + weights).",
        allow_abbrev=False
    )
    p_compile.add_argument("-m", "--model-path",     required=True, help="Path to the original ONNX model.")
    p_compile.add_argument("-c", "--circuit-path",   required=True, help="Output path for the compiled circuit (e.g., circuit.txt).")
    p_compile.add_argument("-q", "--quantized-path", required=True, help="Output path for the quantized model.")

    # witness
    p_wit = sub.add_parser(
        "witness", aliases=["wit"],
        help="Generate witness using a compiled circuit.",
        allow_abbrev=False
    )
    p_wit.add_argument("-c", "--circuit-path",   required=True, help="Path to the compiled circuit.")
    p_wit.add_argument("-q", "--quantized-path", required=True, help="Path to the quantized model (ONNX).")
    p_wit.add_argument("-i", "--input-path",     required=True, help="Path to input JSON.")
    p_wit.add_argument("-o", "--output-path",    required=True, help="Path to write model outputs JSON.")
    p_wit.add_argument("-w", "--witness-path",   required=True, help="Path to write witness.")

    # prove
    p_prove = sub.add_parser(
        "prove", aliases=["prov"],
        help="Generate a proof from a circuit and witness.",
        allow_abbrev=False
    )
    p_prove.add_argument("-c", "--circuit-path", required=True, help="Path to the compiled circuit.")
    p_prove.add_argument("-w", "--witness-path", required=True, help="Path to an existing witness.")
    p_prove.add_argument("-p", "--proof-path",   required=True, help="Path to write proof.")

    # verify
    p_verify = sub.add_parser(
        "verify", aliases=["ver"],
        help="Verify a proof.",
        allow_abbrev=False
    )
    p_verify.add_argument("-c", "--circuit-path",   required=True, help="Path to the compiled circuit.")
    p_verify.add_argument("-i", "--input-path",     required=True, help="Path to input JSON.")
    p_verify.add_argument("-o", "--output-path",    required=True, help="Path to expected outputs JSON.")
    p_verify.add_argument("-w", "--witness-path",   required=True, help="Path to witness.")
    p_verify.add_argument("-p", "--proof-path",     required=True, help="Path to proof.")
    p_verify.add_argument("-q", "--quantized-path", required=True, help="Path to the quantized ONNX (used to infer input shapes).")

    args = parser.parse_args(argv)

    # --- banner --------------------------------------------------------------
    if not args.no_banner and not os.environ.get("JSTPROVE_NO_BANNER"):
        print_header()

    # --- dispatch ------------------------------------------------------------
    try:
        if args.cmd == "compile":
            # Validate inputs and ensure output folders exist
            _ensure_exists(args.model_path, "file")
            _ensure_parent_dir(args.circuit_path)
            _ensure_parent_dir(args.quantized_path)

            # Instantiate the circuit. We use the model filename as a friendly hint.
            model_name_hint = Path(args.model_path).stem
            circuit = _build_default_circuit(model_name_hint)

            # Tell the converter exactly which ONNX file to load (legacy-friendly naming).
            setattr(circuit, "model_file_name", args.model_path)
            # Also set modern-ish aliases some subclasses/readers expect.
            setattr(circuit, "onnx_path", args.model_path)
            setattr(circuit, "model_path", args.model_path)

            # Compile: writes circuit + quantized model
            circuit.base_testing(
                run_type=RunType.COMPILE_CIRCUIT,
                circuit_path=args.circuit_path,
                quantized_path=args.quantized_path,
                dev_mode=True,
            )
            print(f"[compile] done → circuit={args.circuit_path}, quantized={args.quantized_path}")

        elif args.cmd == "witness":
            # Validate required files; ensure we can write outputs
            _ensure_exists(args.circuit_path, "file")
            _ensure_exists(args.quantized_path, "file")
            _ensure_exists(args.input_path, "file")
            _ensure_parent_dir(args.output_path)
            _ensure_parent_dir(args.witness_path)

            circuit = _build_default_circuit("cli")

            # Witness: adjusts inputs (reshape/scale), computes outputs, writes witness
            circuit.base_testing(
                run_type=RunType.GEN_WITNESS,
                circuit_path=args.circuit_path,
                quantized_path=args.quantized_path,
                input_file=args.input_path,
                output_file=args.output_path,
                witness_file=args.witness_path,
            )
            print(f"[witness] wrote witness → {args.witness_path} and outputs → {args.output_path}")

        elif args.cmd == "prove":
            # Validate inputs; ensure we can create the proof file
            _ensure_exists(args.circuit_path, "file")
            _ensure_exists(args.witness_path, "file")
            _ensure_parent_dir(args.proof_path)

            circuit = _build_default_circuit("cli")

            # Prove: witness → proof
            circuit.base_testing(
                run_type=RunType.PROVE_WITNESS,
                circuit_path=args.circuit_path,
                witness_file=args.witness_path,
                proof_file=args.proof_path,
                ecc=False,
            )
            print(f"[prove] wrote proof → {args.proof_path}")

        elif args.cmd == "verify":
            # Validate all inputs exist including quantized (used only to hydrate shapes)
            _ensure_exists(args.circuit_path, "file")
            _ensure_exists(args.input_path, "file")
            _ensure_exists(args.output_path, "file")
            _ensure_exists(args.witness_path, "file")
            _ensure_exists(args.proof_path, "file")
            _ensure_exists(args.quantized_path, "file")

            circuit = _build_default_circuit("cli")

            # Hydrate shapes so adjust_inputs() can reshape consistently.
            # Prefer the circuit's loader; fall back to direct ONNX parsing.
            if hasattr(circuit, "load_quantized_model"):
                circuit.load_quantized_model(args.quantized_path)
            else:
                # fallback: infer from ONNX directly
                m = onnx.load(args.quantized_path)
                shapes = get_input_shapes(m)  # dict of input_name -> shape
                if len(shapes) == 1:
                    circuit.input_shape = [s if s > 0 else 1 for s in next(iter(shapes.values()))]
                else:
                    raise SystemExit(
                        "verify needs load_quantized_model or a single-input model to infer shape"
                    )

            # Verify: checks proof; some backends also emit verifier artifacts
            circuit.base_testing(
                run_type=RunType.GEN_VERIFY,
                circuit_path=args.circuit_path,
                input_file=args.input_path,
                output_file=args.output_path,
                witness_file=args.witness_path,
                proof_file=args.proof_path,
                ecc=False,
            )
            print(f"[verify] verification complete for proof → {args.proof_path}")

        return 0

    # Preserve argparse/our own explicit exits
    except SystemExit:
        raise
    # Convert unexpected exceptions to a clean non-zero exit
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
