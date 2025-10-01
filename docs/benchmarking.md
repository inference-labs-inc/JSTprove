# Benchmarking (experimental)

> **Status:** This benchmarking flow is **experimental/untested** and may change or break.
> Expect rough edges, missing counters, and occasional CLI/flag churn. Please open issues with logs if you hit problems.

This page explains how to benchmark JSTprove across three tracks:

- **LeNet (fixed model)** — runs the same LeNet model/inputs used in the Quickstart.
- **Depth sweep** — synthetic CNNs that vary the number of conv blocks at fixed input size.
- **Breadth sweep** — synthetic CNNs at fixed depth with varying input height/width.

Each run logs **one JSON object per phase per iteration** to a JSONL file so you can slice/plot later.

## What gets recorded

For each phase (`compile`, `witness`, `prove`, `verify`) and iteration:

- **`time_s`** — wall-clock seconds.
- **`mem_mb`** — peak RSS (MB) of child Rust processes (via `psutil`).
- **`mem_mb_rust` / `mem_mb_psutil`** — raw sources when available.
- **`ecc`** (compile only) — parsed ECC counters like `numAdd`, `numMul`, `numVars`, `numConstraints`, `totalCost`.
- **Artifact sizes** (bytes) when applicable:
  - `circuit_size_bytes`, `quantized_size_bytes`, `witness_size_bytes`, `output_size_bytes`, `proof_size_bytes`.
- **Context** — `model` (path), `param_count` (from ONNX initializers), command list, `return_code`, timestamp.

Example row (abridged):

```json
{
  "timestamp": "2025-03-12T18:42:33Z",
  "model": "python/models/models_onnx/lenet.onnx",
  "iteration": 1,
  "phase": "compile",
  "return_code": 0,
  "time_s": 12.345,
  "mem_mb": 512.7,
  "mem_mb_rust": 508.1,
  "mem_mb_psutil": 512.7,
  "param_count": 431080,
  "ecc": { "numAdd": 123456, "numMul": 23456, "numVars": 345678, "numConstraints": 300000, "totalCost": 999999 },
  "circuit_size_bytes": 1234567,
  "quantized_size_bytes": 345678
}
````

> **Notes**
>
> * `mem_mb` prefers the psutil peak across child processes; Rust-reported memory (if any) is kept for reference.
> * ECC parsing is best-effort; keys may be absent if the format shifts.

## Quick recipes

### 1) LeNet (fixed model)

Runs the same demo as in Quickstart, repeatedly, and logs results.

```bash
# Default paths ship with the repo
#   ONNX:  python/models/models_onnx/lenet.onnx
#   Input: python/models/inputs/lenet_input.json
jst bench lenet --iterations 3 --results benchmarking/lenet.jsonl
```

* `--iterations` controls end-to-end loops (default 3 if omitted).
* `--results` chooses the JSONL file (created if missing).

After all iterations, a **summary card** is printed automatically.

### 2) Depth sweep (vary conv depth, fixed input H=W)

```bash
# simple defaults
jst bench depth

# customized
jst bench --sweep depth \
  --depth-min 1 --depth-max 16 \
  --input-hw 56 \
  --iterations 3 \
  --results benchmarking/depth_sweep.jsonl
```

**Topology:** first K blocks are `conv → relu → maxpool(2,2)`, remaining are `conv → relu`, then a small FC tail.
**Pooling:** cap via `--pool-cap` (e.g., 2) or use `--stop-at-hw <min_side>` to stop once H/W drops below a threshold.

### 3) Breadth sweep (vary input H=W, fixed conv depth)

```bash
# simple defaults
jst bench breadth

# customized
jst bench --sweep breadth \
  --arch-depth 5 \
  --input-hw-list 28,56,84,112 \
  --iterations 3 \
  --results benchmarking/breadth_sweep.jsonl \
  --pool-cap 2 --conv-out-ch 16 --fc-hidden 256
```

## Interpreting results

* **Compilation dominates both time and memory.** Expect the compile step to be by far the slowest and most memory-hungry phase.
* **ECC `totalCost` (compile):** Handy single-number proxy for circuit complexity; within a family, runtime/memory often scale ~linearly with it.
* **Best vs mean:** For noisy hosts, run ≥3 iterations and look at both **best** and **mean ± stdev**.

The CLI prints an ASCII table with per-phase `μ ± σ`, best time, and peak memory automatically after the group of iterations.

## Repro & caveats

* Keep **circuit.txt** and **quantized ONNX** from the **same** compile when running witness/prove/verify.
* Large inputs / deep nets can be memory-hungry; close heavy apps on laptops (especially macOS).
* Paths are explicit; the CLI **does not** infer directories.
* Because this is **experimental**, counters/flags and even row fields may evolve.

## Known limitations

* Single-input models only (CLI limitation).
* Memory is sampled from child processes; extreme short-lived peaks can be missed.
* ECC log parsing is brittle to format changes.
