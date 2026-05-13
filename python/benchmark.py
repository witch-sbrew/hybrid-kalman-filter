#!/usr/bin/env python3

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_n_values():
    return [2**power for power in range(0, 13)]


def default_t_values(max_t: int, ratio: float):
    values = {1, 2, 4, 8, 16}

    current = 8.0
    while current <= max_t:
        values.add(int(round(current)))
        current *= ratio

    power = 1
    while power <= max_t:
        values.add(power)
        power *= 2

    return sorted(value for value in values if value >= 1 and value <= max_t)


def generate_experiment_data(n: int, t: int, state_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    initial_states = rng.normal(0.0, 100.0, size=(n, state_dim))
    noise = rng.normal(0.0, 5.0, size=(n, t, state_dim))
    measurements = initial_states[:, np.newaxis, :] + noise
    return initial_states, measurements


def estimate_working_set_bytes(n: int, t: int, state_dim: int):
    value_count = n * t * state_dim
    trajectory_bytes = value_count * 8
    measurement_bytes = value_count * 8
    initial_bytes = n * state_dim * 8
    return 2 * trajectory_bytes + 2 * measurement_bytes + 2 * initial_bytes


def build_cuda_binary(root: Path, build_dir: Path):
    source_dir = root / "src" / "kf-gpu"
    subprocess.run(
        ["cmake", "-S", str(source_dir), "-B", str(build_dir)],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir)],
        check=True,
    )
    return build_dir / "kalman_cuda_batch"


def run_case(
    executable: Path,
    root: Path,
    case_dir: Path,
    n: int,
    t: int,
    state_dim: int,
    seed: int,
):
    initial_states, measurements = generate_experiment_data(
        n=n,
        t=t,
        state_dim=state_dim,
        seed=seed,
    )

    initial_path = case_dir / "initial_states.npy"
    measurements_path = case_dir / "measurements.npy"
    np.save(initial_path, initial_states)
    np.save(measurements_path, measurements)

    command = [
        str(executable),
        str(initial_path),
        str(measurements_path),
        str(root / "reference_outputs.npy"),
        str(case_dir),
    ]

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    elapsed = time.perf_counter() - started

    return {
        "elapsed_seconds": elapsed,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_npy": str(case_dir / "cuda_batch_outputs.npy"),
    }


def write_results_csv(path: Path, rows):
    fieldnames = [
        "timestamp_utc",
        "n",
        "t",
        "state_dim",
        "seed",
        "estimated_working_set_bytes",
        "status",
        "elapsed_seconds",
        "threshold_seconds",
        "returncode",
        "output_npy",
        "stdout_log",
        "stderr_log",
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the CUDA Kalman filter across multiple N and T sizes."
    )
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--threshold-seconds", type=float, default=5.0)
    parser.add_argument("--t-ratio", type=float, default=1.4)
    parser.add_argument("--max-t", type=int, default=4096)
    parser.add_argument(
        "--max-working-set-gb",
        type=float,
        default=3.0,
        help="Skip cases whose estimated total host+device working set exceeds this value.",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/benchmark_results.csv",
        help="CSV file for benchmark timings.",
    )
    parser.add_argument(
        "--build-dir",
        default="build/kf-gpu",
        help="Build directory for the CUDA executable.",
    )
    parser.add_argument(
        "--case-dir",
        default="outputs/benchmark_case",
        help="Directory used for per-run temporary inputs and outputs.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base RNG seed; each case adds a deterministic offset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = repo_root()
    build_dir = root / args.build_dir
    case_dir = root / args.case_dir
    output_csv = root / args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    executable = build_cuda_binary(root, build_dir)
    n_values = default_n_values()
    t_values = default_t_values(max_t=args.max_t, ratio=args.t_ratio)
    max_working_set_bytes = int(args.max_working_set_gb * (1024**3))

    results = []
    print(f"Benchmarking {executable}")
    print(f"N values: {n_values}")
    print(f"T values: {t_values}")
    print(f"Stop threshold: {args.threshold_seconds:.2f}s per run")
    print(f"Working-set cap: {args.max_working_set_gb:.2f} GiB")

    for n in n_values:
        for t in t_values:
            estimated_bytes = estimate_working_set_bytes(
                n=n,
                t=t,
                state_dim=args.state_dim,
            )
            timestamp = datetime.now(timezone.utc).isoformat()
            row = {
                "timestamp_utc": timestamp,
                "n": n,
                "t": t,
                "state_dim": args.state_dim,
                "seed": args.seed_base + n * 100000 + t,
                "estimated_working_set_bytes": estimated_bytes,
                "status": "",
                "elapsed_seconds": "",
                "threshold_seconds": args.threshold_seconds,
                "returncode": "",
                "output_npy": "",
                "stdout_log": "",
                "stderr_log": "",
            }

            if estimated_bytes > max_working_set_bytes:
                row["status"] = "skipped_memory_limit"
                results.append(row)
                print(
                    f"skip N={n:5d} T={t:5d} "
                    f"estimated_working_set={estimated_bytes / (1024**3):.2f} GiB"
                )
                break

            print(f"run  N={n:5d} T={t:5d} ...", flush=True)
            run = run_case(
                executable=executable,
                root=root,
                case_dir=case_dir,
                n=n,
                t=t,
                state_dim=args.state_dim,
                seed=row["seed"],
            )

            row["elapsed_seconds"] = f"{run['elapsed_seconds']:.6f}"
            row["returncode"] = run["returncode"]
            row["output_npy"] = run["output_npy"]
            row["stdout_log"] = run["stdout"].strip()
            row["stderr_log"] = run["stderr"].strip()

            if run["returncode"] != 0:
                row["status"] = "failed"
                results.append(row)
                write_results_csv(output_csv, results)
                raise RuntimeError(
                    f"CUDA run failed for N={n}, T={t}:\n{run['stderr'] or run['stdout']}"
                )

            if run["elapsed_seconds"] > args.threshold_seconds:
                row["status"] = "threshold_exceeded"
                results.append(row)
                print(
                    f"stop N={n:5d} T={t:5d} elapsed={run['elapsed_seconds']:.3f}s "
                    f"(>{args.threshold_seconds:.3f}s)"
                )
                break

            row["status"] = "ok"
            results.append(row)
            print(f"done N={n:5d} T={t:5d} elapsed={run['elapsed_seconds']:.3f}s")

    write_results_csv(output_csv, results)
    print(f"\nWrote benchmark results to {output_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.", file=sys.stderr)
        raise SystemExit(130)
