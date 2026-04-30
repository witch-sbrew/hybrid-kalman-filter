#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/kf-gpu"

cmake -S "$ROOT_DIR/src/kf-gpu" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR"

"$BUILD_DIR/kalman_cuda_batch" \
  "$ROOT_DIR/initial_states.npy" \
  "$ROOT_DIR/measurements.npy" \
  "$ROOT_DIR/reference_outputs.npy" \
  "$ROOT_DIR/outputs"

cd "$ROOT_DIR"
python3 python/verify.py
