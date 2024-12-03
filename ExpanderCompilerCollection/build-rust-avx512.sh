#!/bin/sh
cd "$(dirname "$0")"
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx512f" cargo build --release
mkdir -p ~/.cache/ExpanderCompilerCollection
cp target/release/libec_go_lib.so ~/.cache/ExpanderCompilerCollection