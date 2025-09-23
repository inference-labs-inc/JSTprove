import argparse
import io
from contextlib import redirect_stdout
import json
import os
import re
import tempfile
from python.core.utils.helper_functions import RunType
from python.core.utils.model_registry import list_available_models, get_models_to_test
# slow, not sure why?



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
    mem_match = re.search(r"Rust subprocess memory:\s*([0-9.]+)", output)
    full_time = re.search(r"Full function time\s*:\s*([0-9.]+)", output)
    full_mem = re.search(r"Full function memory\s*:\s*([0-9.]+)", output)
    if not mem_match:
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
                # "circuit_path": str(model.circuit_path),
                # "quantized_path": str(model.quantized_model_file_path),
                "bench": True
            }
    witness_kwargs = {
                "run_type": RunType.GEN_WITNESS,
                "dev_mode": False,
                "bench": True,
                "write_json": True
            }
    prove_kwargs = {
                "run_type": RunType.PROVE_WITNESS,
                "dev_mode": False,
                "bench": True,
                "ecc": False
            }
    verify_kwargs = {
                "run_type": RunType.GEN_VERIFY,
                "dev_mode": False,
                "bench": True,
                "ecc": False

            }
    return compile_kwargs, witness_kwargs, prove_kwargs, verify_kwargs 

def benchmark_model(model_name, model_cls, model_run_kwargs,  args=(), kwargs=None, runs = 1):
    if kwargs is None:
        kwargs = {}
    times = []
    memories = []
    for _ in range(runs):
        model = model_cls(*args, **kwargs)
        
        
        returncode, output = benchmark_tests(model.base_testing, **model_run_kwargs)
        # if returncode != 0:
        #     raise RuntimeError(f"Benchmarking failed for {model_name} with return code {returncode}")
        result = parse_benchmark_output(output)
        times.append(result.get("subprocess_time", "ERR"))
        memories.append(result.get("subprocess_memory", "ERR"))

    avg_time = sum(times) / len(times) if "ERR" not in times else -1
    avg_memory = sum(memories) / len(memories) if "ERR" not in memories else -1

    return {
        "model": model_name,
        "testing_type": model_run_kwargs["run_type"].name,
        "runs": runs,
        "avg_time": avg_time,
        "avg_memory": avg_memory
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", help="Model(s) to benchmark. Example: --model demo relu_dual")
    parser.add_argument("--runs", type=int, default=1, help = "Number of runs to average the results.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--list-models", action="store_true")

    args = parser.parse_args()

    if args.list_models:
        for m in list_available_models():
            print(m)
        return

    # class DummyConfig:
    #     def getoption(self, name):
    #         return getattr(args, name.replace("-", "_"))
    print(args.model)
    selected_models = get_models_to_test(args.model)
    print(selected_models)
    results = []
    compile_kwargs, witness_kwargs, prove_kwargs, verify_kwargs = get_model_run_kwargs()


    for name, cls in selected_models:
        a = ()
        kw = {}
        print(f"Benchmarking {name}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            circuit_file = os.path.join(tmpdir, "circuit.txt")
            quantized_file = os.path.join(tmpdir, "quantized.pt")
            input_file = os.path.join(tmpdir, "input.json")
            output_file = os.path.join(tmpdir, "output.json")
            witness_file = os.path.join(tmpdir, "witness.txt")
            proof_file = os.path.join(tmpdir, "proof.pf")

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

            result = benchmark_model(name, cls, compile_kwargs, a, kw, runs=1)
            print(result)
            results.append(result)

            result = benchmark_model(name, cls, witness_kwargs, a, kw, runs=args.runs)
            print(result)
            results.append(result)

            result = benchmark_model(name, cls, prove_kwargs, a, kw, runs=args.runs)
            print(result)
            results.append(result)

            result = benchmark_model(name, cls, verify_kwargs, a, kw, runs=args.runs)
            print(result)
            results.append(result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
