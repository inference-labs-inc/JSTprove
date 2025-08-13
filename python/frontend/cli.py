# python/frontend/cli.py

import argparse
import importlib
import sys
from pathlib import Path

from python.testing.core.utils.helper_functions import RunType


def _load_circuit(dotted: str):
    """Load a Circuit subclass from 'package.module:ClassName'."""
    if ":" not in dotted:
        raise SystemExit("--circuit must be 'package.module:ClassName'")
    mod_name, cls_name = dotted.split(":", 1)
    mod = importlib.import_module(mod_name)
    try:
        cls = getattr(mod, cls_name)
    except AttributeError as e:
        raise SystemExit(f"Class '{cls_name}' not found in module '{mod_name}'") from e
    return cls()


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
    sub = parser.add_subparsers(dest="cmd", required=True)

    # compile
    p_compile = sub.add_parser("compile", help="Compile a circuit (writes circuit + quantized model + weights).")
    p_compile.add_argument("--circuit", required=True,
                           help="Dotted path to Circuit subclass, e.g. "
                                "python.testing.core.circuit_models.generic_onnx:GenericONNXCircuit")
    p_compile.add_argument("--name", required=True, help="Logical circuit/model name.")
    p_compile.add_argument("--circuit-path", required=True, help="Output path for the compiled circuit (e.g., circuit.txt).")
    p_compile.add_argument("--model-path", required=True, help="Path to the original model to be circuitized.")
    p_compile.add_argument("--quantized-path", required=True, help="Output path for the quantized model.")

    # witness
    p_wit = sub.add_parser("witness", help="Generate witness using a compiled circuit.")
    p_wit.add_argument("--circuit", required=True, help="Dotted path to Circuit subclass.")
    p_wit.add_argument("--name", required=True, help="Logical circuit/model name.")
    p_wit.add_argument("--circuit-path", required=True, help="Path to the compiled circuit.")
    p_wit.add_argument("--quantized-path", required=True, help="Path to the quantized model.")
    p_wit.add_argument("--input-path", required=True, help="Path to input JSON (model inputs).")
    p_wit.add_argument("--output-path", required=True, help="Path to write model outputs JSON.")
    p_wit.add_argument("--witness-path", required=True, help="Path to write witness.")

    # prove
    p_prove = sub.add_parser("prove", help="Generate a proof from a circuit and witness.")
    p_prove.add_argument("--circuit", required=True, help="Dotted path to Circuit subclass.")
    p_prove.add_argument("--name", required=True, help="Logical circuit/model name.")
    p_prove.add_argument("--circuit-path", required=True, help="Path to the compiled circuit.")
    p_prove.add_argument("--witness-path", required=True, help="Path to an existing witness.")
    p_prove.add_argument("--proof-path", required=True, help="Path to write proof.")

    # verify
    p_verify = sub.add_parser("verify", help="Verify a proof.")
    p_verify.add_argument("--circuit", required=True, help="Dotted path to Circuit subclass.")
    p_verify.add_argument("--name", required=True, help="Logical circuit/model name.")
    p_verify.add_argument("--circuit-path", required=True, help="Path to the compiled circuit.")
    p_verify.add_argument("--input-path", required=True, help="Path to input JSON.")
    p_verify.add_argument("--output-path", required=True, help="Path to expected outputs JSON.")
    p_verify.add_argument("--witness-path", required=True, help="Path to witness.")
    p_verify.add_argument("--proof-path", required=True, help="Path to proof.")

    args = parser.parse_args(argv)

    circuit = _load_circuit(args.circuit)

    try:
        if args.cmd == "compile":
            _ensure_exists(args.model_path, "file")
            _ensure_parent_dir(args.circuit_path)
            _ensure_parent_dir(args.quantized_path)

            # Let the Circuit know where the original model is
            setattr(circuit, "model_path", args.model_path)
            setattr(circuit, "onnx_path", args.model_path)

            circuit.base_testing(
                run_type=RunType.COMPILE_CIRCUIT,
                circuit_name=args.name,
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

            circuit.base_testing(
                run_type=RunType.GEN_WITNESS,
                circuit_name=args.name,
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

            circuit.base_testing(
                run_type=RunType.PROVE_WITNESS,
                circuit_name=args.name,
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

            circuit.base_testing(
                run_type=RunType.GEN_VERIFY,
                circuit_name=args.name,
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
