# python/scripts/gather_logits.py
# ruff: noqa: S603

"""
Examples
--------
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
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

# ----- Optional deps: onnx, onnxruntime (required for FP path & ORT attempt on quant) -----
try:
    import onnx
    from onnx import helper, TensorProto, ModelProto, NodeProto, GraphProto
    import onnxruntime as ort
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail as ORTFail
except Exception:
    print(
        "This script requires 'onnx' and 'onnxruntime'. Install them in your env.",
        file=sys.stderr,
    )
    raise


# -----------------------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------------------
# ONNX graph surgery (expose pre-softmax logits when final op is Softmax)
# -----------------------------------------------------------------------------------------


def _producer_map(g: GraphProto) -> Dict[str, NodeProto]:
    return {out: node for node in g.node for out in node.output}


def _find_softmax_output_pair(g: GraphProto) -> Optional[Tuple[NodeProto, str]]:
    """If the sole graph output is produced by Softmax(X), return (softmax_node, X_name)."""
    if len(g.output) != 1:
        return None
    out_name = g.output[0].name
    prod = _producer_map(g)
    node = prod.get(out_name)
    if node is None or node.op_type != "Softmax":
        return None
    if len(node.input) != 1:
        return None
    return (node, node.input[0])


def _clone_model(m: ModelProto) -> ModelProto:
    newm = ModelProto()
    newm.CopyFrom(m)
    return newm


def _expose_tensor_as_output(model: ModelProto, tensor_name: str) -> ModelProto:
    """Return a new model with 'tensor_name' also exposed as a graph output (non-destructive)."""
    m2 = _clone_model(model)
    g = m2.graph
    if any(vi.name == tensor_name for vi in g.output):
        return m2
    # Try to find existing ValueInfo; else create a minimal float tensor info
    vi = next((vi for vi in g.value_info if vi.name == tensor_name), None)
    if vi is None:
        vi = helper.ValueInfoProto()
        vi.name = tensor_name
        vi.type.tensor_type.elem_type = TensorProto.FLOAT
    g.output.extend([vi])
    return m2


def _ensure_pre_softmax_as_output(model_path: Path) -> Tuple[Path, Optional[str]]:
    """
    If the model's only output is Softmax(X), write a temp model that
    also exposes X as an output. Returns (path_to_use, pre_softmax_output_name or None).
    """
    m = onnx.load(str(model_path))
    pair = _find_softmax_output_pair(m.graph)
    if not pair:
        return model_path, None
    _, pre_name = pair
    tmp = model_path.with_suffix(".prelogits.onnx")
    onnx.save(_expose_tensor_as_output(m, pre_name), str(tmp))
    return tmp, pre_name


# -----------------------------------------------------------------------------------------
# ORT helpers
# -----------------------------------------------------------------------------------------


def _ort_session(model_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(
        str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def _run_logits_ort(
    sess: ort.InferenceSession, x: np.ndarray, prefer_output: Optional[str]
) -> np.ndarray:
    in_name = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    want = prefer_output if (prefer_output and prefer_output in outs) else outs[0]
    y = sess.run([want], {in_name: x})[0]
    y = np.asarray(y)
    if y.ndim >= 2 and y.shape[0] == 1:
        y = y.reshape(y.shape[1:])  # drop batch for N==1
    return y.astype(np.float32, copy=False)


# -----------------------------------------------------------------------------------------
# Compile → circuit & quantized; witness fallback for quantized path
# -----------------------------------------------------------------------------------------


def _quantized_path_from_circuit(circuit_path: Path) -> Path:
    return circuit_path.with_name(f"{circuit_path.stem}_quantized_model.onnx")


def _compile_fp_once(fp_model: Path, workdir: Path) -> Tuple[Path, Path]:
    """
    Compile the FP ONNX once, returning (circuit_path, quantized_onnx_path).
    Both are placed in 'workdir'.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    circuit = workdir / "circuit.txt"
    rc = subprocess.run(
        ["jst", "--no-banner", "compile", "-m", str(fp_model), "-c", str(circuit)],
        text=True,
    ).returncode
    if rc != 0:
        raise SystemExit(f"[compile] failed rc={rc}")
    q = _quantized_path_from_circuit(circuit)
    if not q.exists():
        # Fallback scan
        cands = list(workdir.glob("*quantized*.onnx"))
        if not cands:
            raise SystemExit("Quantized ONNX not found after compile.")
        q = cands[0]
    return circuit, q


def _witness_logits_once(
    circuit_path: Path, input_json: Path, out_dir: Path
) -> np.ndarray:
    """
    Run 'jst witness' to compute outputs for the given input JSON with the compiled circuit.
    Returns the output tensor as float32 (cast from whatever domain).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "output.json"
    wit_bin = out_dir / "witness.bin"
    rc = subprocess.run(
        [
            "jst",
            "--no-banner",
            "witness",
            "-c",
            str(circuit_path),
            "-i",
            str(input_json),
            "-o",
            str(out_json),
            "-w",
            str(wit_bin),
        ],
        text=True,
    ).returncode
    if rc != 0:
        raise SystemExit(f"[witness] failed rc={rc}")
    with out_json.open("r", encoding="utf-8") as f:
        blob = json.load(f)
    return np.array(blob["output"], dtype=np.float32).ravel()


# -----------------------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------------------


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# -----------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------


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
        help="Quantized ONNX path; if omitted, we'll compile once to produce one.",
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
        help="Try to expose pre-softmax tensor and use it when present (ORT paths only).",
    )
    args = ap.parse_args()

    fp_model = Path(args.model).resolve()
    if not fp_model.exists():
        raise SystemExit(f"Missing model: {fp_model}")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Work area that persists for the whole run so we can reuse compiled artifacts
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)

        # Compile once to get circuit + (default) quantized ONNX (for fallback and/or ORT)
        circuit_path, q_from_compile = _compile_fp_once(fp_model, tmp)

        # Choose quantized path (explicit arg wins)
        q_model = Path(args.quantized).resolve() if args.quantized else q_from_compile

        # Optionally expose pre-softmax for ORT models (no effect on witness fallback)
        use_fp_path = fp_model
        prefer_fp_out = None
        use_q_path = q_model
        prefer_q_out = None
        if args.force_pre_softmax:
            use_fp_path, prefer_fp_out = _ensure_pre_softmax_as_output(fp_model)
            # Only try to rewrite quantized graph if we're going to use ORT on it
            use_q_path, prefer_q_out = _ensure_pre_softmax_as_output(q_model)

        # ORT session for FP model
        fp_sess = _ort_session(use_fp_path)

        # Try ORT for quantized; if it fails with unregistered op/domain issues, switch to runner
        use_runner_fallback = False
        q_sess: Optional[ort.InferenceSession] = None
        try:
            q_sess = _ort_session(use_q_path)
        except ORTFail as e:
            msg = str(e)
            if (
                ("not a registered function/op" in msg)
                or ("is not a registered function" in msg)
                or ("domain" in msg.lower())
            ):
                use_runner_fallback = True
            else:
                raise
        except Exception:
            # Any other unexpected ORT initialization failure -> fallback
            use_runner_fallback = True

        warned_pre_softmax_once = False

        rows = 0
        with out_path.open("w", encoding="utf-8") as f:
            for sid, ipath in _iter_inputs(
                Path(args.input) if args.input else None, args.inputs_glob
            ):
                x = _read_input_json(Path(ipath))
                x_fp32 = x.astype(np.float32, copy=False)

                # FP logits via ORT
                logits_fp = _run_logits_ort(fp_sess, x_fp32, prefer_fp_out)

                # Quantized logits: ORT if available else runner fallback (witness)
                if not use_runner_fallback and q_sess is not None:
                    logits_q = _run_logits_ort(q_sess, x_fp32, prefer_q_out)
                else:
                    if args.force_pre_softmax and not warned_pre_softmax_once:
                        print(
                            "[warn] --force-pre-softmax is ignored for runner fallback; using model outputs.",
                            file=sys.stderr,
                        )  # noqa: T201
                        warned_pre_softmax_once = True
                    # Use circuit + witness to get outputs
                    run_dir = tmp / f"wit_{rows:06d}"
                    logits_q = _witness_logits_once(circuit_path, Path(ipath), run_dir)

                # Row + metrics
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
