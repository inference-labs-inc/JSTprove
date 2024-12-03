#!/bin/sh
cd "$(dirname "$0")"
cargo build --release
mkdir -p ~/.cache/ExpanderCompilerCollection
cp target/release/libec_go_lib.so ~/.cache/ExpanderCompilerCollection