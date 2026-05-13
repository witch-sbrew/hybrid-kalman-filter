#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required to plot benchmark results.") from exc


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot CUDA benchmark results with T on the x-axis and runtime on the y-axis."
    )
    parser.add_argument(
        "--input-csv",
        default="outputs/benchmark_results.csv",
        help="CSV produced by python/benchmark.py",
    )
    parser.add_argument(
        "--output",
        default="outputs/benchmark_plot.png",
        help="Output image path.",
    )
    return parser.parse_args()


def load_series(csv_path: Path):
    series = defaultdict(list)

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["status"] not in {"ok", "threshold_exceeded"}:
                continue

            n = int(row["n"])
            t = int(row["t"])
            elapsed = float(row["elapsed_seconds"])
            if n > 8: series[n].append((t, elapsed, row["status"]))

    for n in series:
        series[n].sort(key=lambda item: item[0])
    
    return series


def line_color(index: int, total: int):
    if total <= 1:
        mix = 0.0
    else:
        mix = index / float(total - 1)

    blue = (0.12, 0.38, 0.95)
    green = (0.10, 0.68, 0.32)
    return tuple(
        blue[channel] + mix * (green[channel] - blue[channel])
        for channel in range(3)
    )


def main():
    args = parse_args()
    root = repo_root()
    input_csv = root / args.input_csv
    output_path = root / args.output

    if not input_csv.exists():
        raise SystemExit(f"Benchmark CSV not found: {input_csv}")

    series = load_series(input_csv)
    if not series:
        raise SystemExit("No plottable benchmark rows were found in the CSV.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ordered_series = sorted(series.items())
    for index, (n, points) in enumerate(ordered_series):
        ts = [point[0] for point in points]
        ys = [point[1] for point in points]
        color = line_color(index, len(ordered_series))
        ax.plot(ts, ys, marker="o", linewidth=1.8, label=f"N={n}", color=color)

        if points[-1][2] == "threshold_exceeded":
            ax.scatter([points[-1][0]], [points[-1][1]], marker="x", s=70, color=color)

    ax.set_title("CUDA Kalman Filter Benchmark")
    ax.set_xlabel("T (timesteps)")
    ax.set_ylabel("Wall-clock runtime (seconds)")
    ax.set_xscale("log", base=2)
    # ax.set_yscale("log", base=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(title="Batch size", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
