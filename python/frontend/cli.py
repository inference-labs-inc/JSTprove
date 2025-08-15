# python/frontend/cli.py

import argparse
import importlib
import sys
from pathlib import Path
from python.testing.core.utils.helper_functions import RunType
import onnx
from python.testing.core.utils.onnx_helpers import get_input_shapes
import os

DEFAULT_CIRCUIT_DOTTED = "python.testing.core.circuit_models.generic_onnx:GenericModelONNX"

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except Exception:  # colorama optional
    class _No:
        def __getattr__(self, _): return ""
    Fore = Style = _No()

BANNER_TITLE = r"""
         _/    _/_/_/  _/_/_/_/_/  _/_/_/                                             
        _/  _/            _/      _/    _/  _/  _/_/    _/_/    _/      _/    _/_/    
       _/    _/_/        _/      _/_/_/    _/_/      _/    _/  _/      _/  _/_/_/_/   
_/    _/        _/      _/      _/        _/        _/    _/    _/  _/    _/          
 _/_/    _/_/_/        _/      _/        _/          _/_/        _/        _/_/_/     
"""

def print_header():
    subtitle = "Verifiable ML by Inference Labs"
    footer = "Based on Polyhedra Network's Expander (GKR-based proving system)"
    print(
        f"{Fore.CYAN}{BANNER_TITLE}{Style.RESET_ALL}"
        f"{Fore.YELLOW}JSTProve{Style.RESET_ALL} — {subtitle}\n"
        f"{Fore.WHITE}{footer}{Style.RESET_ALL}\n"
    )

def _import_default_circuit():
    mod_name, cls_name = DEFAULT_CIRCUIT_DOTTED.split(":", 1)
    mod = importlib.import_module(mod_name)
    try:
        return getattr(mod, cls_name)
    except AttributeError as e:
        raise SystemExit(f"Default circuit class '{cls_name}' not found in '{mod_name}'") from e

def _build_default_circuit(model_name_hint: str | None = None):
    """
    GenericModelONNX requires a constructor arg (e.g., model_name).
    Use a safe hint if provided; otherwise a neutral placeholder.
    """
    cls = _import_default_circuit()
    name = (model_name_hint or "cli")
    # try common constructor patterns
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
    p = Path(path)
    if kind == "file" and not p.is_file():
        raise SystemExit(f"Required {kind} not found: {path}")
    if kind == "dir" and not p.is_dir():
        raise SystemExit(f"Required {kind} not found: {path}")

def _ensure_parent_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(
        prog="jstprove",
        description="ZK-ML CLI (compile, witness, prove, verify)."
    )
    parser.add_argument(
    "--no-banner", action="store_true",
    help="Suppress the startup banner."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # compile
    p_compile = sub.add_parser("compile", help="Compile a circuit (writes circuit + quantized model + weights).")
    p_compile.add_argument("--model-path", required=True, help="Path to the original ONNX model.")
    p_compile.add_argument("--circuit-path", required=True, help="Output path for the compiled circuit (e.g., circuit.txt).")
    p_compile.add_argument("--quantized-path", required=True, help="Output path for the quantized model.")

    # witness
    p_wit = sub.add_parser("witness", help="Generate witness using a compiled circuit.")
    p_wit.add_argument("--circuit-path", required=True, help="Path to the compiled circuit.")
    p_wit.add_argument("--quantized-path", required=True, help="Path to the quantized model.")
    p_wit.add_argument("--input-path", required=True, help="Path to input JSON.")
    p_wit.add_argument("--output-path", required=True, help="Path to write model outputs JSON.")
    p_wit.add_argument("--witness-path", required=True, help="Path to write witness.")

    # prove
    p_prove = sub.add_parser("prove", help="Generate a proof from a circuit and witness.")
    p_prove.add_argument("--circuit-path", required=True, help="Path to the compiled circuit.")
    p_prove.add_argument("--witness-path", required=True, help="Path to an existing witness.")
    p_prove.add_argument("--proof-path", required=True, help="Path to write proof.")

    # verify
    p_verify = sub.add_parser("verify", help="Verify a proof.")
    p_verify.add_argument("--circuit-path", required=True, help="Path to the compiled circuit.")
    p_verify.add_argument("--input-path", required=True, help="Path to input JSON.")
    p_verify.add_argument("--output-path", required=True, help="Path to expected outputs JSON.")
    p_verify.add_argument("--witness-path", required=True, help="Path to witness.")
    p_verify.add_argument("--proof-path", required=True, help="Path to proof.")
    p_verify.add_argument("--quantized-path", required=True, help="Path to the quantized ONNX (used to infer input shapes).")

    args = parser.parse_args(argv)

    if not args.no_banner and not os.environ.get("JSTPROVE_NO_BANNER"):
        print_header()

    try:
        if args.cmd == "compile":
            _ensure_exists(args.model_path, "file")
            _ensure_parent_dir(args.circuit_path)
            _ensure_parent_dir(args.quantized_path)

            # Instantiate the default circuit; hint its name from the model file
            model_name_hint = Path(args.model_path).stem
            circuit = _build_default_circuit(model_name_hint)

            # Make sure ONNXConverter sees the explicit model path
            setattr(circuit, "model_file_name", args.model_path)
            # Harmless for other codepaths:
            setattr(circuit, "onnx_path", args.model_path)
            setattr(circuit, "model_path", args.model_path)

            circuit.base_testing(
                run_type=RunType.COMPILE_CIRCUIT,
                circuit_path=args.circuit_path,
                quantized_path=args.quantized_path,
            )
            print(f"[compile] done → circuit={args.circuit_path}, quantized={args.quantized_path}")

        elif args.cmd == "witness":
            _ensure_exists(args.circuit_path, "file")
            _ensure_exists(args.quantized_path, "file")
            _ensure_exists(args.input_path, "file")
            _ensure_parent_dir(args.output_path)
            _ensure_parent_dir(args.witness_path)

            # We don't need a real model name here; a neutral one is fine
            circuit = _build_default_circuit("cli")

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
            _ensure_exists(args.circuit_path, "file")
            _ensure_exists(args.witness_path, "file")
            _ensure_parent_dir(args.proof_path)

            circuit = _build_default_circuit("cli")

            circuit.base_testing(
                run_type=RunType.PROVE_WITNESS,
                circuit_path=args.circuit_path,
                witness_file=args.witness_path,
                proof_file=args.proof_path,
            )
            print(f"[prove] wrote proof → {args.proof_path}")

        elif args.cmd == "verify":
            _ensure_exists(args.circuit_path, "file")
            _ensure_exists(args.input_path, "file")
            _ensure_exists(args.output_path, "file")
            _ensure_exists(args.witness_path, "file")
            _ensure_exists(args.proof_path, "file")
            _ensure_exists(args.quantized_path, "file")

            circuit = _build_default_circuit("cli")

            # hydrate shapes so adjust_inputs() works
            if hasattr(circuit, "load_quantized_model"):
                circuit.load_quantized_model(args.quantized_path)
            else:
                # fallback: infer from ONNX directly
                m = onnx.load(args.quantized_path)
                shapes = get_input_shapes(m)  # dict of input_name -> shape
                if len(shapes) == 1:
                    circuit.input_shape = [s if s > 0 else 1 for s in next(iter(shapes.values()))]
                else:
                    raise SystemExit("verify needs load_quantized_model or a single-input model to infer shape")

            circuit.base_testing(
                run_type=RunType.GEN_VERIFY,
                circuit_path=args.circuit_path,
                input_file=args.input_path,
                output_file=args.output_path,
                witness_file=args.witness_path,
                proof_file=args.proof_path,
            )
            print(f"[verify] verification complete for proof → {args.proof_path}")

        return 0

    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
