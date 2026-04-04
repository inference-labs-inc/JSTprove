#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$SCRIPT_DIR/models"
WORK_DIR=$(mktemp -d)
BINARY="$REPO_ROOT/target/release/jstprove"

trap 'rm -rf "$WORK_DIR"' EXIT

branch=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)
commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD)

echo "================================================================"
echo "  Soundness constraint benchmark"
echo "  Branch: $branch ($commit)"
echo "================================================================"
echo ""

if [ ! -d "$MODEL_DIR" ]; then
    echo "Generating ONNX models..."
    python3 "$SCRIPT_DIR/generate_models.py"
    echo ""
fi

echo "Building jstprove (release)..."
cargo build --release --bin jstprove --manifest-path "$REPO_ROOT/Cargo.toml" 2>&1 | tail -3
echo ""

OPS="exp sigmoid gelu softmax layer_norm resize gridsample sqrt tanh erf pow matmul averagepool pad reducesum"

printf "%-14s %10s %10s %14s %12s %12s %12s\n" \
    "Layer" "MulGates" "AddGates" "CircuitFile" "Compile" "Prove" "Verify"
printf "%-14s %10s %10s %14s %12s %12s %12s\n" \
    "-----------" "--------" "--------" "----------" "----------" "----------" "----------"

for op in $OPS; do
    onnx_path="$MODEL_DIR/${op}.onnx"
    compiled_path="$WORK_DIR/${op}.circuit"
    witness_path="$WORK_DIR/${op}.witness"
    proof_path="$WORK_DIR/${op}.proof"

    if [ ! -f "$onnx_path" ]; then
        printf "%-14s SKIP\n" "$op"
        continue
    fi

    mul_gates="-"
    add_gates="-"
    circuit_size="-"
    compile_t="-"
    prove_t="-"
    verify_t="-"

    compile_out=$("$BINARY" run_compile_circuit \
        --onnx "$onnx_path" -c "$compiled_path" 2>&1) || true

    mul_gates=$(echo "$compile_out" | grep -oE '[0-9]+ mul' | grep -oE '[0-9]+' || echo "-")
    add_gates=$(echo "$compile_out" | grep -oE '[0-9]+ add' | grep -oE '[0-9]+' || echo "-")
    compile_t=$(echo "$compile_out" | grep -oE 'in [0-9.]+s' | grep -oE '[0-9.]+' || echo "-")
    if [ -d "$compiled_path" ]; then
        circuit_size=$(du -sk "$compiled_path" | awk '{printf "%dKB", $1}')
    fi

    output_path="$WORK_DIR/${op}_output.msgpack"
    if [ -d "$compiled_path" ]; then
        witness_out=$("$BINARY" run_gen_witness \
            -c "$compiled_path" \
            -i "$MODEL_DIR/${op}_input.msgpack" \
            -o "$output_path" \
            -w "$witness_path" 2>&1) || true
    fi

    if [ -f "$witness_path" ] || [ -d "$witness_path" ]; then
        prove_start=$(python3 -c 'import time; print(time.monotonic())')
        prove_out=$("$BINARY" run_prove_witness \
            -c "$compiled_path" -w "$witness_path" \
            -p "$proof_path" 2>&1) || true
        prove_end=$(python3 -c 'import time; print(time.monotonic())')
        prove_t=$(python3 -c "print(f'{$prove_end - $prove_start:.2f}')")
    fi

    if [ -f "$proof_path" ] || [ -d "$proof_path" ]; then
        verify_start=$(python3 -c 'import time; print(time.monotonic())')
        verify_out=$("$BINARY" run_gen_verify \
            -c "$compiled_path" \
            -w "$witness_path" \
            -p "$proof_path" 2>&1) || true
        verify_end=$(python3 -c 'import time; print(time.monotonic())')
        verify_t=$(python3 -c "print(f'{$verify_end - $verify_start:.2f}')")
    fi

    printf "%-14s %10s %10s %14s %12s %12s %12s\n" \
        "$op" "$mul_gates" "$add_gates" "$circuit_size" "${compile_t}s" "${prove_t}s" "${verify_t}s"

    rm -rf "$compiled_path" "$witness_path" "$proof_path"
done

echo ""
echo "Branch: $branch ($commit)"
echo "Tensor size: DIM=32 (elementwise), 8x8 (spatial)"
echo "Lookup table: MAX_FUNCTION_LOOKUP_BITS=18 (262K entries)"
