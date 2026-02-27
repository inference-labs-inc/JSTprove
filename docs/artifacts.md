# Artifacts

This page describes the files JSTprove reads and writes during the pipeline.

---

## Serialization format

All artifacts are serialized as **msgpack** wrapped in a jstprove envelope (`JST\x01` magic, 20-byte header). The envelope contains:

| Field | Size | Description |
|-------|------|-------------|
| Magic | 4 bytes | `JST\x01` |
| Flags | 4 bytes (LE u32) | Bit 0: zstd compressed |
| Payload length | 8 bytes (LE u64) | Length of the payload after the header |
| CRC32c | 4 bytes (LE u32) | CRC32c checksum of the payload |

The payload is msgpack data, optionally zstd-compressed (controlled by the `--no-compress` flag). Legacy formats (bare zstd or raw msgpack without envelope) are also accepted on read.

---

## Expander backend artifacts

**`CompiledCircuit`** (msgpack) -- produced by `msgpack_compile` or `run_compile_circuit`.

| Field | Type | Description |
|-------|------|-------------|
| `circuit` | bytes | Serialized Expander layered circuit |
| `witness_solver` | bytes | Serialized Expander witness solver |
| `metadata` | optional `CircuitParams` | ONNX model parameters (scale, inputs, outputs, etc.) |
| `version` | optional `ArtifactVersion` | Artifact version tag |

**`WitnessBundle`** (msgpack) -- produced by witness commands.

| Field | Type | Description |
|-------|------|-------------|
| `witness` | bytes | Serialized Expander witness |
| `output_data` | optional `Vec<i64>` | Model output values |
| `version` | optional `ArtifactVersion` | Artifact version tag |

**`ProofBundle`** (msgpack) -- produced by prove commands.

| Field | Type | Description |
|-------|------|-------------|
| `proof` | bytes | Serialized proof |
| `version` | optional `ArtifactVersion` | Artifact version tag |

---

## Remainder backend artifacts

**`QuantizedModel`** (msgpack) -- produced by `jstprove-remainder compile`. Contains the quantized ONNX graph and scale configuration.

Witness and proof artifacts for the remainder backend also use msgpack serialization with the same envelope format.

---

## Tips

- Keep artifacts from the **same compile** together (compiled circuit + witness + proof) to avoid version mismatches.
- If you change the ONNX model, **re-run compile** before witness/prove/verify.
- Pass `--no-compress` to disable zstd compression if you need to inspect raw msgpack payloads.
