# convert_output.py
import numpy as np
import os
import sys


def convert_bin_to_npy(bin_path, npy_path):
    """
    Reads a .bin file written by save_outputs() in C/CUDA.
    Format: [int64 N][int64 state_dim][float64 x N x state_dim]
    """
    if not os.path.exists(bin_path):
        print(f"  [SKIP] {bin_path} not found")
        return False

    with open(bin_path, "rb") as f:
        # Read header
        N = np.frombuffer(f.read(8), dtype=np.int64)[0]
        state_dim = np.frombuffer(f.read(8), dtype=np.int64)[0]
        # Read data
        data = np.frombuffer(f.read(), dtype=np.float64)

    expected = N * state_dim
    if len(data) != expected:
        print(f"  [ERROR] {bin_path}: expected {expected} doubles, got {len(data)}")
        print(f"          N={N}, state_dim={state_dim}")
        print(f"          File may be corrupted or save_outputs() called wrong.")
        return False

    arr = data.reshape(N, state_dim)
    np.save(npy_path, arr)
    print(f"  [OK] {bin_path} → {npy_path}  shape={arr.shape}")
    return True


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    # TODO: rename based on actual naming conventions
    conversions = [
        ("outputs/cpp_serial_raw.bin", "outputs/cpp_serial_outputs.npy"),
        ("outputs/cpp_openmp_raw.bin", "outputs/cpp_openmp_outputs.npy"),
        ("outputs/cuda_single_raw.bin", "outputs/cuda_single_outputs.npy"),
        ("outputs/cuda_batch_raw.bin", "outputs/cuda_batch_outputs.npy"),
    ]

    print("\nConverting raw C/CUDA outputs to .npy...")
    for bin_path, npy_path in conversions:
        convert_bin_to_npy(bin_path, npy_path)
    print("Done.\n")
