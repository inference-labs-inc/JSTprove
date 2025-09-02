# python/scripts/benchmark_runner.py
"""
Benchmark JSTProve phases (compile → witness → prove → verify) and write results to JSONL.

JSONL = "JSON Lines": one JSON object per line. Easy to append/parse and ideal for later analysis.

Example:
  python -m python.scripts.benchmark_runner --list-models
  python -m python.scripts.benchmark_runner --model lenet --iterations 5 --output results.jsonl --summarize
"""

import argparse
import io
import json
import os
import re
import sys
import tempfile
import socket
import platform
from datetime import datetime, timezone
from contextlib import redirect_stdout

from python.testing.core.utils.helper_functions import RunType
from python.testing.core.utils.model_registry import list_available_models, get_models_to_test


# ---------------------------
# Parse runner stdout
# ---------------------------

_TIME_PATTERNS = [
    re.compile(r"Rust time taken:\s*([0-9.]+)"),            # common in compile/prove/verify
    re.compile(r"Time elapsed:\s*([0-9.]+)\s*seconds"),     # common in witness/prove/verify
]
_MEM_PATTERNS = [
    re.compile(r"Peak Memory used Overall\s*:\s*([0-9.]+)"),   # common runner print
    re.compile(r"Rust subprocess memory\s*:\s*([0-9.]+)"),     # legacy
]

def parse_benchmark_output(output: str) -> dict:
    """Extract time (seconds) and peak memory (MB) from runner output."""
    time_s = None
    mem_mb = None

    for pat in _TIME_PATTERNS:
        m = pat.search(output)
        if m:
            try:
                time_s = float(m.group(1))
                break
            except ValueError:
                pass

    for pat in _MEM_PATTERNS:
        m = pat.search(output)
        if m:
            try:
                mem_mb = float(m.group(1))
                break
            except ValueError:
                pass

    return {"time_s": time_s, "mem_mb": mem_mb}


# ---------------------------
# Run a single phase with capture
# ---------------------------

def _run_with_capture(fn, **kwargs) -> tuple[int, dict, str]:
    """
    Calls model.base_testing(**kwargs) while capturing stdout.
    Returns (return_code, parsed_metrics_dict, raw_stdout).
    """
    buf = io.StringIO()
    kwargs["bench"] = True  # ensure the runner prints timing/memory
    rc = 0
    with redirect_stdout(buf):
        try:
            ret = fn(**kwargs)
            rc = int(ret) if isinstance(ret, int) else 0
        except SystemExit as e:
            rc = int(e.code) if isinstance(e.code, int) else 1
        except Exception:
            rc = 1
            raise
    out = buf.getvalue()
    metrics = parse_benchmark_output(out)
    return rc, metrics, out


# ---------------------------
# Canonical kwargs per phase
# ---------------------------

def make_phase_kwargs(tmpdir: str, *, input_path: str | None) -> dict:
    """
    Construct a dict with per-phase kwargs (each is a dict) pointing
    to temp artifacts in tmpdir; reuses input_path if provided.
    """
    circuit_path   = os.path.join(tmpdir, "circuit.txt")
    quantized_path = os.path.join(tmpdir, "quantized.onnx")
    output_path    = os.path.join(tmpdir, "output.json")
    witness_path   = os.path.join(tmpdir, "witness.bin")
    proof_path     = os.path.join(tmpdir, "proof.bin")

    # If user provided a fixed input, use it; otherwise let the model
    # implementation handle input generation or defaults (if it can).
    input_file = input_path if input_path else os.path.join(tmpdir, "input.json")

    compile_kwargs = dict(
        run_type=RunType.COMPILE_CIRCUIT,
        dev_mode=True,                # refresh dev build path when compiling circuits
        circuit_path=circuit_path,
        quantized_path=quantized_path,
    )
    witness_kwargs = dict(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        circuit_path=circuit_path,
        quantized_path=quantized_path,
        input_file=input_file,
        output_file=output_path,
        witness_file=witness_path,
        write_json=True,             # ask pipeline to persist I/O
    )
    prove_kwargs = dict(
        run_type=RunType.PROVE_WITNESS,
        dev_mode=False,
        circuit_path=circuit_path,
        witness_file=witness_path,
        proof_file=proof_path,
        ecc=False,                   # run proof directly via Expander
    )
    verify_kwargs = dict(
        run_type=RunType.GEN_VERIFY,
        dev_mode=False,
        circuit_path=circuit_path,
        quantized_path=quantized_path,  # hydrate shapes if needed
        input_file=input_file,
        output_file=output_path,
        witness_file=witness_path,
        proof_file=proof_path,
        ecc=False,
    )
    return dict(
        compile=compile_kwargs,
        witness=witness_kwargs,
        prove=prove_kwargs,
        verify=verify_kwargs,
    )


# ---------------------------
# Model selection normalization
# ---------------------------

def normalize_selected_models(selected):
    """
    Accepts whatever get_models_to_test(...) returns and normalizes to an
    iterable of (name, cls, init_args, init_kwargs).

    Supported shapes:
      - dict-like: {name: cls, ...}
      - list of classes: [ClsA, ClsB]
      - list of tuples:
          (name, cls)
          (name, cls, dict_kwargs)
          (name, cls, tuple/list args)
          (cls,)  -> name inferred from class
    """
    # Dict-like
    if hasattr(selected, "items"):
        for name, cls in selected.items():
            yield str(name), cls, (), {}
        return

    # Sequence
    for item in selected:
        # Tuple-like
        if isinstance(item, tuple):
            if len(item) == 2:
                name, cls = item
                yield str(name), cls, (), {}
            elif len(item) >= 3:
                name, cls = item[0], item[1]
                third = item[2]
                if isinstance(third, dict):
                    yield str(name), cls, (), third
                elif isinstance(third, (list, tuple)):
                    yield str(name), cls, tuple(third), {}
                else:
                    yield str(name), cls, (), {}
            elif len(item) == 1:
                cls = item[0]
                name = getattr(cls, "NAME", getattr(cls, "__name__", "model"))
                yield str(name), cls, (), {}
            continue

        # Class only
        if hasattr(item, "__name__"):
            cls = item
            name = getattr(cls, "NAME", getattr(cls, "__name__", "model"))
            yield str(name), cls, (), {}
            continue

        # Fallback: skip unknown shapes
        # (Optionally: print warning)
        # print(f"[warn] unknown model entry shape: {item!r}", file=sys.stderr)


# ---------------------------
# Aggregation helpers
# ---------------------------

def _safe_mean(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return (sum(vals) / len(vals)) if vals else None

def _safe_stdev(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    n = len(vals)
    if n < 2:
        return None
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    return var ** 0.5

def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _env_metadata():
    return {
        "ts": _now_iso(),
        "host": socket.gethostname(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
    }


# ---------------------------
# Main benchmark routine
# ---------------------------

def run_benchmarks(selected_models, *, iterations: int, input_path: str | None,
                   keep_tmp: bool, jsonl_path: str, summarize: bool):
    """
    For each model: run compile→witness→prove→verify for N iterations.
    Write per-run JSON lines to jsonl_path. If summarize=True, append
    summary records at the end of the same JSONL file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)) or ".", exist_ok=True)

    with open(jsonl_path, "w", encoding="utf-8") as out_f:
        meta = _env_metadata()

        for model_name, model_cls, init_args, init_kwargs in normalize_selected_models(selected_models):
            print(f"\n=== Benchmarking {model_name} for {iterations} iteration(s) ===")

            per_phase_times = { "compile": [], "witness": [], "prove": [], "verify": [] }
            per_phase_mems  = { "compile": [], "witness": [], "prove": [], "verify": [] }

            for it in range(1, iterations + 1):
                # Fresh temp dir per iteration to avoid artifact reuse
                ctx = tempfile.TemporaryDirectory()
                tmpdir = ctx.name
                temp_mgr = ctx if not keep_tmp else None  # keep handle alive unless keep_tmp

                phase_kwargs = make_phase_kwargs(tmpdir, input_path=input_path)
                model = model_cls(*init_args, **init_kwargs)  # fresh instance each iteration

                for phase in ("compile", "witness", "prove", "verify"):
                    rc, metrics, raw = _run_with_capture(model.base_testing, **phase_kwargs[phase])

                    record = {
                        "record_type": "run",
                        "model": model_name,
                        "iteration": it,
                        "phase": phase,
                        "return_code": rc,
                        "time_s": metrics.get("time_s"),
                        "mem_mb": metrics.get("mem_mb"),
                        "tmpdir": tmpdir if keep_tmp else None,
                        **meta,
                    }
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

                    # Collect for summary
                    t = metrics.get("time_s")
                    m = metrics.get("mem_mb")
                    per_phase_times[phase].append(t if isinstance(t, (int, float)) else None)
                    per_phase_mems[phase].append(m if isinstance(m, (int, float)) else None)

                    if rc != 0:
                        print(f"[{model_name}][{phase}][iter {it}] return_code={rc} — recorded; continuing.")

                # Drop the tempdir unless --keep-tmp set
                if temp_mgr is None:
                    ctx.cleanup()

            if summarize:
                # Append summary records (one per phase) for this model
                for phase in ("compile", "witness", "prove", "verify"):
                    times = [x for x in per_phase_times[phase] if isinstance(x, (int, float))]
                    mems  = [x for x in per_phase_mems[phase]  if isinstance(x, (int, float))]
                    summary = {
                        "record_type": "summary",
                        "model": model_name,
                        "phase": phase,
                        "iterations": iterations,
                        "n_ok_time": len(times),
                        "mean_time_s": _safe_mean(times),
                        "stdev_time_s": _safe_stdev(times),
                        "n_ok_mem": len(mems),
                        "mean_mem_mb": _safe_mean(mems),
                        "stdev_mem_mb": _safe_stdev(mems),
                        **meta,
                    }
                    out_f.write(json.dumps(summary) + "\n")
                    out_f.flush()

    print(f"\n✔ Wrote results to {jsonl_path} (per-run records{' + summaries' if summarize else ''}).")


# ---------------------------
# CLI
# ---------------------------

def main():
    p = argparse.ArgumentParser(description="Benchmark JSTProve phases (compile→witness→prove→verify).")
    p.add_argument("--model", nargs="+", help="Model(s) to benchmark (see --list-models).")
    p.add_argument("--list-models", action="store_true", help="List available models and exit.")
    p.add_argument("--iterations", type=int, default=5, help="Number of full E2E loops (default: 5).")
    p.add_argument("--input", type=str, default=None,
                   help="Optional path to input JSON. If omitted, model may generate its own (depends on model).")
    p.add_argument("--output", type=str, default="benchmark_results.jsonl",
                   help="Output JSONL path (per-run records; summaries appended if --summarize).")
    p.add_argument("--keep-tmp", action="store_true", help="Keep per-iteration temp dirs (debugging).")
    p.add_argument("--summarize", action="store_true", help="Append per-model summary rows to the JSONL.")

    args = p.parse_args()

    if args.list_models:
        for m in list_available_models():
            print(m)
        return 0

    selected = get_models_to_test(args.model)
    if not selected:
        print("No models selected. Use --list-models to see options.", file=sys.stderr)
        return 2

    input_path = os.path.abspath(args.input) if args.input else None
    jsonl_path = os.path.abspath(args.output)

    run_benchmarks(
        selected_models=selected,
        iterations=args.iterations,
        input_path=input_path,
        keep_tmp=args.keep_tmp,
        jsonl_path=jsonl_path,
        summarize=args.summarize,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())