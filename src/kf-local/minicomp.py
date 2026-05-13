import numpy as np
import sys
import os

def compare(name, ref, candidate, tol):
    if candidate is None:
        print(f"\n  [SKIP] {name} — file missing")
        return True  # not a failure, just not ready yet

    if ref.shape != candidate.shape:
        print(f"\n  [FAIL] {name}")
        print(f"         Shape mismatch: ref={ref.shape} candidate={candidate.shape}")
        print(f"         Most likely cause: N or STATE_DIM don't match reference.")
        print(f"         Fix: rerun C/CUDA with same N and STATE_DIM as Python.")
        return False

    diff = np.abs(ref - candidate)
    max_diff = diff.max()
    mean_diff = diff.mean()
    worst_idx = np.unravel_index(diff.argmax(), diff.shape)
    passed = max_diff < tol

    status = "  [PASS]" if passed else "  [FAIL]"
    print(f"\n{status} {name}")
    print(f"         tolerance : {tol:.0e}")
    print(f"         max  diff : {max_diff:.6e}  at index {worst_idx}")
    print(f"         mean diff : {mean_diff:.6e}")

    return passed

ref = np.load("reference_outputs.npy")
cpp = np.load("cpp_outputs.npy")

compare("cpp", ref, cpp, 1e-4)