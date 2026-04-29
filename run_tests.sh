#!/bin/bash
# run_tests.sh
# Usage: bash run_tests.sh
# Runs all implementations, converts outputs, verifies correctness. TODO: untested

set -e  # stop on first error
echo "=================================================="
echo "  Kalman Filter Test Pipeline"
echo "=================================================="

# Create output directory
mkdir -p outputs

# ── Step 1: Python reference ──────────────────────────────
echo ""
echo "[1/5] Running Python reference..."
python reference_filter.py
echo "      Done."

# ── Step 2: C++ serial ────────────────────────────────────
echo ""
echo "[2/5] Running C++ serial..."
if [ -f "./kalman_serial" ]; then
    ./kalman_serial
    echo "      Done."
else
    echo "      [SKIP] kalman_serial not found — build it first:"
    echo "      g++ -O3 -I./eigen kalman_serial.cpp -o kalman_serial"
fi

# ── Step 3: C++ OpenMP ────────────────────────────────────
echo ""
echo "[3/5] Running C++ OpenMP..."
if [ -f "./kalman_openmp" ]; then
    ./kalman_openmp
    echo "      Done."
else
    echo "      [SKIP] kalman_openmp not found — build it first:"
    echo "      g++ -O3 -fopenmp -I./eigen kalman_openmp.cpp -o kalman_openmp"
fi

# ── Step 4: CUDA ──────────────────────────────────────────
echo ""
echo "[4/5] Running CUDA..."
if [ -f "./kalman_cuda" ]; then
    ./kalman_cuda
    echo "      Done."
else
    echo "      [SKIP] kalman_cuda not found — build it first:"
    echo "      nvcc -O3 -lcublas kalman_cublas.cu -o kalman_cuda"
fi

# ── Step 5: Convert + verify ──────────────────────────────
echo ""
echo "[5/5] Converting outputs and verifying correctness..."
python convert_output.py
python verify.py

echo ""
echo "Pipeline complete."