# verify.py
import numpy as np
import sys
import os

TOLS = {"python_vs_cpp": 1e-9, "python_vs_cuda": 1e-6}


def load(path):
    if not os.path.exists(path):
        return None

    arr = np.load(path)
    print(f"  [LOADED] {path}  shape={arr.shape}")
    return arr


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

    if not passed:
        if max_diff < 1e-6:
            print(f"         LIKELY CAUSE: floating point reordering (not a real bug)")
            print(f"         FIX: loosen tolerance to 1e-6 for this comparison")
        elif max_diff < 1e-2:
            print(f"         LIKELY CAUSE: float32 vs float64 mismatch")
            print(f"         FIX: make sure C/CUDA uses double not float")
            print(f"              cuBLAS: use cublasDgemm not cublasSgemm")
            print(f"              C++:    use double not float in Eigen template")
        else:
            print(f"         LIKELY CAUSE: real implementation bug")
            print(f"         FIX: check matrix layout, pointer arithmetic,")
            print(f"              or whether predict/update steps match Python")

    return passed


def print_summary(results):
    print("\n" + "=" * 60)
    passed = [r for r in results if r[1] is True]
    failed = [r for r in results if r[1] is False]
    skipped = [r for r in results if r[1] is None]

    print(f"  PASSED:  {len(passed)}")
    print(f"  FAILED:  {len(failed)}")
    print(f"  SKIPPED: {len(skipped)} (files not yet generated)")

    if failed:
        print(f"\n  Fix these before running performance experiments:")
        for name, _ in failed:
            print(f"    - {name}")
    elif len(passed) > 0:
        print(f"\n  All available implementations verified correct.")
        print(f"  Safe to run performance experiments.")

    print("=" * 60 + "\n")


def main():
    print("\n" + "=" * 60)
    print("  Kalman Filter Correctness Verification")
    print("=" * 60)

    # Load reference
    print("\nLoading reference...")
    ref = load("outputs/reference_outputs.npy")
    if ref is None:
        print("\n  ERROR: outputs/reference_outputs.npy not found.")
        print("  Run this first:")
        print("    python reference_filter.py")
        sys.exit(1)

    print(f"\n  Reference: N={ref.shape[0]}, state_dim={ref.shape[1]}")
    print(f"  Value range: [{ref.min():.3f}, {ref.max():.3f}]")

    results = []

    checks = [
        (
            "C++ serial   vs Python",
            "outputs/cpp_serial_outputs.npy",
            TOLS["python_vs_cpp"],
        ),
        (
            "C++ OpenMP   vs Python",
            "outputs/cpp_openmp_outputs.npy",
            TOLS["python_vs_cpp"],
        ),
        (
            "CUDA single  vs Python",
            "outputs/cuda_single_outputs.npy",
            TOLS["python_vs_cuda"],
        ),
        (
            "CUDA batched vs Python",
            "outputs/cuda_batch_outputs.npy",
            TOLS["python_vs_cuda"],
        ),
    ]

    print("\nRunning checks...")
    for name, path, tol in checks:
        print(f"\n  {'-'*56}")
        print(f"  Checking: {name}")
        candidate = load(path)
        if candidate is None:
            print(f"  [SKIP] output file not yet generated")
            results.append((name, None))
        else:
            passed = compare(name, ref, candidate, tol)
            results.append((name, passed))

    # Internal CUDA consistency check
    print(f"\n  {'-'*56}")
    print(f"  Checking: CUDA batched vs CUDA single (internal)")
    cuda_single = load("outputs/cuda_single_outputs.npy")
    cuda_batched = load("outputs/cuda_batch_outputs.npy")
    if cuda_single is not None and cuda_batched is not None:
        passed = compare(
            "CUDA batch vs single", cuda_single, cuda_batched, TOLS["cuda_internal"]
        )
        results.append(("CUDA batch vs single", passed))
    else:
        print("  [SKIP] need both CUDA outputs")
        results.append(("CUDA batch vs single", None))

    print_summary(results)


if __name__ == "__main__":
    main()
