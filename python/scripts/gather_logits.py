# python/scripts/gather_logits.py
# ruff: noqa: S603

"""
python -m python.scripts.gather_logits \
  --model python/models/models_onnx/lenet.onnx \
  --input python/models/inputs/lenet_input.json \
  --out benchmarking/lenet_logits.jsonl \
  --force-pre-softmax

python -m python.scripts.gather_logits \
  --model python/models/models_onnx/lenet.onnx \
  --inputs-glob "python/models/inputs/mnist/*.json" \
  --out benchmarking/mnist_logits.jsonl \
  --force-pre-softmax
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

# Optional deps: onnx, onnxruntime
try:
    import onnx
    from onnx import (
        helper,
        numpy_helper,
        TensorProto,
        ModelProto,
        GraphProto,
        NodeProto,
        ValueInfoProto,
    )
    import onnxruntime as ort
except Exception as e:
    print(
        "This script requires 'onnx' and 'onnxruntime'. Install them in your env.",
        file=sys.stderr,
    )
    raise

# ---------- IO helpers ----------


def _read_input_json(p: Path) -> np.ndarray:
    """Read {"input":[...], "shape":[N,C,H,W]?} into an ndarray shaped as NCHW."""
    with p.open("r", encoding="utf-8") as f:
        blob = json.load(f)
    arr = np.array(blob["input"], dtype=np.float32)
    shp = blob.get("shape")
    if shp:
        arr = arr.reshape(shp)
    return arr


def _iter_inputs(
    single: Optional[Path], glob_pat: Optional[str]
) -> Iterable[Tuple[str, Path]]:
    if single:
        yield (single.name, single)
        return
    if glob_pat:
        for s in sorted(glob.glob(glob_pat)):
            p = Path(s)
            if p.is_file():
                yield (p.name, p)
        return
    raise SystemExit("Must pass --input or --inputs-glob")


# ---------- ONNX graph surgery (expose pre-softmax) ----------


def _find_softmax_output_pair(g: GraphProto) -> Optional[Tuple[NodeProto, str]]:
    """If the sole graph output is produced by a Softmax node, return (softmax_node, softmax_input_name)."""
    if len(g.output) != 1:
        return None
    out_name = g.output[0].name
    # Build producer map: output name -> node
    prod = {o: n for n in g.node for o in n.output}
    node = prod.get(out_name)
    if node is None or node.op_type != "Softmax":
        return None
    if len(node.input) != 1:
        return None
    return (node, node.input[0])


def _expose_tensor_as_output(
    model: ModelProto, tensor_name: str, output_name: Optional[str] = None
) -> ModelProto:
    """Return a new model with 'tensor_name' also exposed as a graph output (non-destructive)."""
    model = ModelProto()
    model.CopyFrom(model)
    g = model.graph
    # Avoid duplicate outputs
    if any(vi.name == tensor_name for vi in g.output):
        return model
    # Try to discover type/shape from existing value_info/initializers
    vi = next((vi for vi in g.value_info if vi.name == tensor_name), None)
    if vi is None:
        # Fallback: make a minimally-typed float tensor info
        vi = helper.ValueInfoProto()
        vi.name = tensor_name
        vi.type.tensor_type.elem_type = TensorProto.FLOAT
    # Append as additional output
    g.output.extend([vi])
    return model


def _ensure_pre_softmax_as_output(model_path: Path) -> Tuple[Path, Optional[str]]:
    """
    If the model's sole output is the result of a Softmax, write a temp model that
    also exposes the pre-softmax tensor as an output. Returns (path_to_use, pre_softmax_output_name or None).
    """
    m = onnx.load(str(model_path))
    pair = _find_softmax_output_pair(m.graph)
    if not pair:
        return model_path, None
    softmax_node, pre_name = pair
    tmp = model_path.with_suffix(".prelogits.onnx")
    newm = _expose_tensor_as_output(m, pre_name)
    onnx.save(newm, str(tmp))
    return tmp, pre_name


# ---------- ORT inference ----------


def _session(model_path: Path) -> ort.InferenceSession:
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(
        str(model_path), sess_options=sess_opts, providers=providers
    )


def _run_logits(
    sess: ort.InferenceSession, x: np.ndarray, prefer_pre: Optional[str]
) -> np.ndarray:
    """
    Run the session and pick logits:
      - If 'prefer_pre' is provided and is among outputs, return that tensor.
      - Else return the first output.
      - Squeezes batch dimension when N==1.
    """
    in_name = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    want = prefer_pre if (prefer_pre and prefer_pre in outs) else outs[0]
    y = sess.run([want], {in_name: x})[0]
    y = np.asarray(y)
    if y.ndim >= 2 and y.shape[0] == 1:
        y = y.reshape(y.shape[1:])  # drop batch
    return y.astype(np.float32, copy=False)


# ---------- Quantized path discovery ----------


def _quantized_path_from_circuit(circuit_path: Path) -> Path:
    return circuit_path.with_name(f"{circuit_path.stem}_quantized_model.onnx")


def _get_or_make_quantized(fp_model: Path, explicit_q: Optional[Path]) -> Path:
    if explicit_q:
        return explicit_q
    # Call jst compile → writes circuit + quantized next to the circuit
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        circuit = tmp / "circuit.txt"
        cmd = ["jst", "--no-banner", "compile", "-m", str(fp_model), "-c", str(circuit)]
        rc = subprocess.run(cmd, text=True).returncode
        if rc != 0:
            raise SystemExit(f"[compile] failed rc={rc}")
        q = _quantized_path_from_circuit(circuit)
        if not q.exists():
            # Some versions might name it slightly differently; fall back to scan
            cand = list(tmp.glob("*quantized*.onnx"))
            if not cand:
                raise SystemExit("Quantized ONNX not found after compile.")
            q = cand[0]
        # Copy to a stable sibling next to FP model
        out = fp_model.with_name(fp_model.stem + "_quantized.onnx")
        shutil.copy2(q, out)
        return out


# ---------- Metrics ----------


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ---------- Main ----------


@dataclass
class Row:
    sample: str
    model_fp: str
    model_q: str
    argmax_fp: Optional[int]
    argmax_q: Optional[int]
    logits_fp: list[float]
    logits_q: list[float]
    l2: Optional[float]
    cosine: Optional[float]

    def to_json(self) -> str:
        return json.dumps(self.__dict__)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gather logits from FP vs quantized ONNX models into a JSONL."
    )
    ap.add_argument("--model", required=True, help="FP ONNX path (e.g., lenet.onnx)")
    ap.add_argument(
        "--quantized",
        help="Quantized ONNX path (if omitted, we'll run 'jst compile' to produce one).",
    )
    ap.add_argument("--input", help="Single input JSON to run.")
    ap.add_argument(
        "--inputs-glob",
        help="Glob for many inputs (e.g., 'python/models/inputs/mnist/*.json').",
    )
    ap.add_argument("--out", default="logits_pairs.jsonl", help="Output JSONL path.")
    ap.add_argument(
        "--force-pre-softmax",
        action="store_true",
        help="Try to expose pre-softmax tensor and use it when present.",
    )
    args = ap.parse_args()

    fp_model = Path(args.model).resolve()
    if not fp_model.exists():
        raise SystemExit(f"Missing model: {fp_model}")

    q_model = _get_or_make_quantized(
        fp_model, Path(args.quantized).resolve() if args.quantized else None
    )

    # Possibly expose pre-softmax on both graphs
    prefer_fp = None
    prefer_q = None
    use_fp_path = fp_model
    use_q_path = q_model
    if args.force_pre_softmax:
        use_fp_path, prefer_fp = _ensure_pre_softmax_as_output(fp_model)
        use_q_path, prefer_q = _ensure_pre_softmax_as_output(q_model)

    fp_sess = _session(use_fp_path)
    q_sess = _session(use_q_path)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with out_path.open("w", encoding="utf-8") as f:
        for sid, ipath in _iter_inputs(
            Path(args.input) if args.input else None, args.inputs_glob
        ):
            x = _read_input_json(Path(ipath))
            # ORT prefers float; quantized model ops will internally cast as needed
            x_fp32 = x.astype(np.float32, copy=False)
            logits_fp = _run_logits(fp_sess, x_fp32, prefer_fp)
            logits_q = _run_logits(q_sess, x_fp32, prefer_q)

            # Prepare row
            amx_fp = int(np.argmax(logits_fp)) if logits_fp.size else None
            amx_q = int(np.argmax(logits_q)) if logits_q.size else None

            row = Row(
                sample=sid,
                model_fp=str(fp_model),
                model_q=str(q_model),
                argmax_fp=amx_fp,
                argmax_q=amx_q,
                logits_fp=[float(v) for v in np.ravel(logits_fp)],
                logits_q=[float(v) for v in np.ravel(logits_q)],
                l2=_l2(logits_fp, logits_q),
                cosine=_safe_cosine(logits_fp, logits_q),
            )
            f.write(row.to_json() + "\n")
            rows += 1

    print(f"✔ Wrote {rows} rows to {out_path}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
