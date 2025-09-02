# python/scripts/benchmark_runner.py

import argparse
import io
from contextlib import redirect_stdout
import json
import os
import re
import tempfile
from pathlib import Path
import importlib  # NEW: to load GenericModelONNX dynamically

from python.testing.core.utils.helper_functions import RunType
from python.testing.core.utils.model_registry import list_available_models, get_models_to_test


def benchmark_tests(fn, *args, **kwargs):
    f = io.StringIO()
    kwargs["bench"] = True
    with redirect_stdout(f):
        returncode = fn(*args, **kwargs)
    output = f.getvalue()
    return returncode, output


def parse_benchmark_output(output: str):
    result = {}

    time_match = re.search(r"Rust time taken:\s*([0-9.]+)", output)
    mem_match = re.search(r"Peak Memory used Overall\s*:\s*([0-9.]+)", output) or \
                re.search(r"Rust subprocess memory:\s*([0-9.]+)", output)
    full_time = re.search(r"Full function time\s*:\s*([0-9.]+)", output)
    full_mem = re.search(r"Full function memory\s*:\s*([0-9.]+)", output)

    if not mem_match:
        # Helpful when patterns change
        print(output)

    if time_match:
        result["subprocess_time"] = float(time_match.group(1))
    if mem_match:
        result["subprocess_memory"] = float(mem_match.group(1))
    if full_time:
        result["full_time"] = float(full_time.group(1))
    if full_mem:
        result["full_memory"] = float(full_mem.group(1))

    return result


def get_model_run_kwargs():
    compile_kwargs = {
        "run_type": RunType.COMPILE_CIRCUIT,
        "dev_mode": True,
        "bench": True,
    }
    witness_kwargs = {
        "run_type": RunType.GEN_WITNESS,
        "dev_mode": False,
        "bench": True,
        "write_json": True,
    }
    prove_kwargs = {
        "run_type": RunType.PROVE_WITNESS,
        "dev_mode": False,
        "bench": True,
        "ecc": False,
    }
    verify_kwargs = {
        "run_type": RunType.GEN_VERIFY,
        "dev_mode": False,
        "bench": True,
        "ecc": False,
    }
    return compile_kwargs, witness_kwargs, prove_kwargs, verify_kwargs


# NEW: minimal helper to build the default circuit and target a specific ONNX file
def _build_default_circuit_for_onnx(onnx_path: str, name_hint: str | None = None):
    mod = importlib.import_module("python.testing.core.circuit_models.generic_onnx")
    cls = getattr(mod, "GenericModelONNX")
    name = (name_hint or Path(onnx_path).stem or "cli")

    # Try common constructor shapes
    for attempt in (
        lambda: cls(model_name=name),
        lambda: cls(name=name),
        lambda: cls(name),     # positional
        lambda: cls(),         # last resort
    ):
        try:
            inst = attempt()
            break
        except TypeError:
            continue
    else:
        raise RuntimeError("Could not construct GenericModelONNX")

    # Point the instance at the user ONNX (mirrors the CLI behavior)
    setattr(inst, "model_file_name", onnx_path)
    setattr(inst, "onnx_path", onnx_path)
    setattr(inst, "model_path", onnx_path)
    return inst


def benchmark_model(model_name, model_cls_or_onnx, model_run_kwargs, args=(), kwargs=None, runs=1):
    if kwargs is None:
        kwargs = {}
    times = []
    memories = []

    # Detect whether we were given a class or an ONNX path
    is_callable = callable(model_cls_or_onnx)
    is_pathlike = isinstance(model_cls_or_onnx, str)

    for _ in range(runs):
        if is_callable:
            model = model_cls_or_onnx(*args, **kwargs)
        elif is_pathlike:
            model = _build_default_circuit_for_onnx(model_cls_or_onnx, name_hint=model_name)
        else:
            raise TypeError(f"Unrecognized model spec for '{model_name}': {type(model_cls_or_onnx)}")

        returncode, output = benchmark_tests(model.base_testing, **model_run_kwargs)
        result = parse_benchmark_output(output)
        times.append(result.get("subprocess_time", "ERR"))
        memories.append(result.get("subprocess_memory", "ERR"))

    avg_time = sum(t for t in times if t != "ERR") / len([t for t in times if t != "ERR"]) if "ERR" not in times else -1
    avg_memory = sum(m for m in memories if m != "ERR") / len([m for m in memories if m != "ERR"]) if "ERR" not in memories else -1

    return {
        "model": model_name,
        "testing_type": model_run_kwargs["run_type"].name,
        "runs": runs,
        "avg_time": avg_time,
        "avg_memory": avg_memory,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", help="Model(s) to benchmark. Example: --model path/to/net.onnx other.onnx")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to average the results.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--list-models", action="store_true")

    args = parser.parse_args()

    if args.list_models:
        for m in list_available_models():
            print(m)
        return

    print(args.model)
    selected_models = get_models_to_test(args.model)
    print(selected_models)

    results = []
    compile_kwargs, witness_kwargs, prove_kwargs, verify_kwargs = get_model_run_kwargs()

    # NOTE: selected_models can be:
    #   * list of (name, class) or (name, onnx_path)
    #   * list of onnx_path strings
    #   * dict name -> onnx_path
    #
    # Minimal normalization below:
    if isinstance(selected_models, dict):
        iterable = [(name, spec) for name, spec in selected_models.items()]
    else:
        iterable = selected_models

    for item in iterable:
        # Accept tuple or bare string ONNX path
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            name, cls_or_path = item[0], item[1]
        elif isinstance(item, str):
            cls_or_path = item
            name = Path(item).stem
        else:
            raise ValueError(f"Unrecognized registry entry: {item!r}")

        print(f"Benchmarking {name}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            circuit_file = os.path.join(tmpdir, "circuit.txt")
            # CHANGED: quantized model should be .onnx, not .pt
            quantized_file = os.path.join(tmpdir, "quantized.onnx")
            input_file = os.path.join(tmpdir, "input.json")
            output_file = os.path.join(tmpdir, "output.json")
            # CHANGED: use .bin for witness/proof (consistent with CLI)
            witness_file = os.path.join(tmpdir, "witness.bin")
            proof_file = os.path.join(tmpdir, "proof.bin")

            compile_kwargs["circuit_path"] = circuit_file
            compile_kwargs["quantized_path"] = quantized_file
            compile_kwargs["input_file"] = input_file
            compile_kwargs["output_file"] = output_file
            compile_kwargs["witness_file"] = witness_file
            compile_kwargs["proof_file"] = proof_file

            witness_kwargs["circuit_path"] = circuit_file
            witness_kwargs["quantized_path"] = quantized_file
            witness_kwargs["input_file"] = input_file
            witness_kwargs["output_file"] = output_file
            witness_kwargs["witness_file"] = witness_file
            witness_kwargs["proof_file"] = proof_file

            prove_kwargs["circuit_path"] = circuit_file
            # (prove doesn’t need quantized_path but keeping symmetry won’t hurt)
            prove_kwargs["quantized_path"] = quantized_file
            prove_kwargs["input_file"] = input_file
            prove_kwargs["output_file"] = output_file
            prove_kwargs["witness_file"] = witness_file
            prove_kwargs["proof_file"] = proof_file

            verify_kwargs["circuit_path"] = circuit_file
            verify_kwargs["quantized_path"] = quantized_file
            verify_kwargs["input_file"] = input_file
            verify_kwargs["output_file"] = output_file
            verify_kwargs["witness_file"] = witness_file
            verify_kwargs["proof_file"] = proof_file

            # Run phases (compile once; others repeated per args.runs)
            result = benchmark_model(name, cls_or_path, compile_kwargs, runs=1)
            print(result)
            results.append(result)

            result = benchmark_model(name, cls_or_path, witness_kwargs, runs=args.runs)
            print(result)
            results.append(result)

            result = benchmark_model(name, cls_or_path, prove_kwargs, runs=args.runs)
            print(result)
            results.append(result)

            result = benchmark_model(name, cls_or_path, verify_kwargs, runs=args.runs)
            print(result)
            results.append(result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
