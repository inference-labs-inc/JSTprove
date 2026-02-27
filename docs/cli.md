# CLI Reference

JSTprove provides two CLI binaries: **`jstprove`** (Expander backend) and **`jstprove-remainder`** (Remainder backend).

---

## `jstprove` (Expander backend)

Built from `rust/jstprove_circuits/bin/generic_demo.rs`. Uses positional command dispatch (not subcommands).

### Synopsis

```bash
jstprove <command> [options]
```

### Global options

| Flag | Short | Description |
|------|-------|-------------|
| `--circuit` | `-c` | Path to compiled circuit (msgpack) |
| `--input` | `-i` | Path to input data file |
| `--output` | `-o` | Path to output data file |
| `--witness` | `-w` | Path to witness file |
| `--proof` | `-p` | Path to proof file |
| `--meta` | `-m` | Path to ONNX circuit params / metadata (msgpack) |
| `--arch` | `-a` | Path to ONNX architecture definition (msgpack) |
| `--wandb` | `-b` | Path to ONNX W&B data (msgpack) |
| `--manifest` | `-f` | Path to batch manifest (msgpack) |
| `--name` | `-n` | Circuit name for file naming |
| `--no-compress` | | Disable zstd compression for output files |
| `--backend` | | Proving backend: `expander` (default) or `remainder` |
| `--model` | | Path to quantized ONNX model (Remainder backend) |
| `--onnx` | | Path to ONNX model (generates metadata automatically; requires the Cargo feature `remainder`, e.g. `cargo build --features remainder`) |

### Commands

**Compilation**

| Command | Description |
|---------|-------------|
| `run_compile_circuit` | Compile circuit and serialize to legacy format. Requires `-c`, `--meta`, `--arch`. |
| `msgpack_compile` | Compile circuit and serialize as `CompiledCircuit` msgpack bundle. Requires `-c`, `--meta`, `--arch` (or `--onnx`). |

**Witness generation**

| Command | Description |
|---------|-------------|
| `run_gen_witness` | Generate witness from compiled circuit. Requires `-c`, `-i`, `-o`, `-w`, `--meta`. |
| `run_debug_witness` | Generate witness with debug evaluation. Requires `-c`, `-i`, `-o`, `-w`, `--meta`, `--arch`. |
| `msgpack_witness_stdin` | Read witness request from stdin (msgpack), write witness to stdout. |

**Proving**

| Command | Description |
|---------|-------------|
| `run_prove_witness` | Generate proof from witness. Requires `-c`, `-w`, `-p`. |
| `msgpack_prove` | Prove from msgpack circuit bundle. Requires `-c`, `-w`, `-p`. |
| `msgpack_prove_stdin` | Read prove request from stdin, write proof to stdout. |

**Verification**

| Command | Description |
|---------|-------------|
| `run_gen_verify` | Verify proof against circuit and I/O. Requires `-c`, `-i`, `-o`, `-w`, `-p`, `--meta`. |
| `msgpack_verify` | Verify from msgpack files. Requires `-c`, `-w`, `-p`. |
| `msgpack_verify_stdin` | Read verify request from stdin, write response to stdout. |

**Batch operations**

| Command | Description |
|---------|-------------|
| `run_batch_witness` | Batch witness generation from manifest. Requires `-c`, `-f`, `--meta`. |
| `run_batch_prove` | Batch proving from manifest. Requires `-c`, `-f`. |
| `run_batch_verify` | Batch verification from manifest. Requires `-c`, `-f`, `--meta`. |

**Pipe operations** (stdin/stdout streaming)

| Command | Description |
|---------|-------------|
| `run_pipe_witness` | Stream witness jobs from stdin. Requires `-c`, `--meta`. |
| `run_pipe_prove` | Stream prove jobs from stdin. Requires `-c`. |
| `run_pipe_verify` | Stream verify jobs from stdin. Requires `-c`, `--meta`. |

### Example

```bash
jstprove msgpack_compile \
  --onnx rust/jstprove_remainder/models/lenet.onnx \
  -c artifacts/lenet/circuit.msgpack

jstprove run_gen_witness \
  -c artifacts/lenet/circuit.msgpack \
  -i artifacts/lenet/input.json \
  -o artifacts/lenet/output.json \
  -w artifacts/lenet/witness.msgpack

jstprove run_prove_witness \
  -c artifacts/lenet/circuit.msgpack \
  -w artifacts/lenet/witness.msgpack \
  -p artifacts/lenet/proof.msgpack

jstprove run_gen_verify \
  -c artifacts/lenet/circuit.msgpack \
  -i artifacts/lenet/input.json \
  -o artifacts/lenet/output.json \
  -w artifacts/lenet/witness.msgpack \
  -p artifacts/lenet/proof.msgpack
```

---

## `jstprove-remainder` (Remainder backend)

Built from `rust/jstprove_remainder/src/main.rs`. Uses clap-derive subcommands.

### Synopsis

```bash
jstprove-remainder <subcommand> [options]
```

### Subcommands

**compile** -- Compile an ONNX model into a `QuantizedModel` (msgpack).

| Flag | Short | Description |
|------|-------|-------------|
| `--model` | `-m` | Path to input ONNX model |
| `--output` | `-o` | Path to output compiled model |
| `--no-compress` | | Disable zstd compression |

**witness** -- Generate witness data from a compiled model and input.

| Flag | Short | Description |
|------|-------|-------------|
| `--model` | | Path to compiled model |
| `--input` | `-i` | Path to input data |
| `--output` | `-o` | Path to output witness |
| `--no-compress` | | Disable zstd compression |

**prove** -- Generate proof from model and witness.

| Flag | Short | Description |
|------|-------|-------------|
| `--model` | | Path to compiled model |
| `--witness` | `-w` | Path to witness file |
| `--output` | `-o` | Path to output proof |
| `--no-compress` | | Disable zstd compression |

**verify** -- Verify a proof.

| Flag | Short | Description |
|------|-------|-------------|
| `--model` | | Path to compiled model |
| `--proof` | | Path to proof file |
| `--input` | `-i` | Path to input data |

**batch-witness** -- Batch witness generation from a manifest.

| Flag | Short | Description |
|------|-------|-------------|
| `--model` | | Path to compiled model |
| `--manifest` | `-m` | Path to batch manifest |
| `--no-compress` | | Disable zstd compression |

**batch-prove** -- Batch proving from a manifest.

| Flag | Short | Description |
|------|-------|-------------|
| `--model` | | Path to compiled model |
| `--manifest` | `-m` | Path to batch manifest |
| `--no-compress` | | Disable zstd compression |

**batch-verify** -- Batch verification from a manifest.

| Flag | Short | Description |
|------|-------|-------------|
| `--model` | | Path to compiled model |
| `--manifest` | `-m` | Path to batch manifest |

**pipe-witness**, **pipe-prove**, **pipe-verify** -- Streaming operations via stdin/stdout.

| Flag | Short | Description |
|------|-------|-------------|
| `--model` | | Path to compiled model |
| `--no-compress` | | Disable zstd compression (witness/prove only) |

### Example

```bash
jstprove-remainder compile \
  -m rust/jstprove_remainder/models/lenet.onnx \
  -o artifacts/lenet/model.msgpack

jstprove-remainder witness \
  --model artifacts/lenet/model.msgpack \
  -i artifacts/lenet/input.json \
  -o artifacts/lenet/witness.msgpack

jstprove-remainder prove \
  --model artifacts/lenet/model.msgpack \
  -w artifacts/lenet/witness.msgpack \
  -o artifacts/lenet/proof.msgpack

jstprove-remainder verify \
  --model artifacts/lenet/model.msgpack \
  --proof artifacts/lenet/proof.msgpack \
  -i artifacts/lenet/input.json
```

---

## Notes

- The circuit type used by `jstprove` is `Circuit` (defined in `jstprove_circuits::onnx`).
- All paths are **mandatory**; no automatic discovery or inference.
- Output artifacts use the jstprove envelope format (msgpack with optional zstd compression). Pass `--no-compress` to disable compression.
