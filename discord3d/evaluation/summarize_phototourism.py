#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = [
    "acc",
    "fscore_2cm",
    "precision_2cm",
    "outlier_5cm",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        default="outputs/eval",
        help="Directory containing phototourism_full_nv*_t*_L*.summary.json files.",
    )
    ap.add_argument("--layers", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--views", nargs="+", type=int, default=[3, 5, 8])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument(
        "--output_csv",
        default="outputs/eval/phototourism_layer_matrix.csv",
    )
    return ap.parse_args()


def mean_metric(summary_path: Path, metric: str) -> float:
    data = json.loads(summary_path.read_text())
    rows = [row for row in data["rows"] if row["method"] == "discord"]
    return sum(row[metric] for row in rows) / len(rows)


def resolve_summary_path(input_dir: Path, layer: int, n_views: int, trials: int) -> Path:
    candidates = [
        input_dir / f"phototourism_full_nv{n_views}_t{trials}_L{layer}.summary.json",
    ]
    if layer == 8:
        candidates.append(input_dir / f"phototourism_full_nv{n_views}_t{trials}_current.summary.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find summary for layer={layer}, n_views={n_views}, trials={trials}")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "n_views", *METRICS])
        for layer in args.layers:
            for n_views in args.views:
                summary_path = resolve_summary_path(input_dir, layer, n_views, args.trials)
                row = [layer, n_views]
                row.extend(f"{mean_metric(summary_path, metric):.4f}" for metric in METRICS)
                writer.writerow(row)
    print(f"Saved summary matrix to {output_csv}")


if __name__ == "__main__":
    main()
