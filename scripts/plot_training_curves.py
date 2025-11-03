#!/usr/bin/env python3
"""Utility to plot training losses, learning rates, and validation metrics.

The script consumes the JSONL analytics files generated during training
(`training_history.jsonl` and `validation_history.jsonl`) and exports line
charts that can be embedded into reports or papers.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt

TRAINING_FILE = "training_history.jsonl"
VALIDATION_FILE = "validation_history.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        help="Path to the experiment directory (e.g. experiments/run_name).",
    )
    parser.add_argument(
        "--analytics-dir",
        type=Path,
        help="Directory that contains analytics JSONL files. Overrides --experiment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store generated plots. Defaults to the analytics directory.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Window size for moving-average smoothing. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures interactively instead of saving them.",
    )
    return parser.parse_args()


def moving_average(values: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    window = max(1, int(window))
    prefix = [0.0]
    for v in values:
        prefix.append(prefix[-1] + v)
    result: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        total = prefix[idx + 1] - prefix[start]
        count = idx - start + 1
        result.append(total / max(1, count))
    return result


def load_jsonl(path: Path) -> List[Mapping[str, object]]:
    records: List[Mapping[str, object]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def prepare_series(records: Iterable[Mapping[str, object]], *, include_prefix: str) -> Tuple[List[float], Dict[str, List[float]]]:
    x_values: List[float] = []
    series: Dict[str, List[float]] = defaultdict(list)
    for item in sorted(records, key=lambda r: r.get("iter", 0)):
        iteration = item.get("iter")
        if iteration is None:
            continue
        x_values.append(float(iteration))
        for key, value in item.items():
            if not isinstance(key, str):
                continue
            if key in {"iter", "timestamp", "dataset"}:
                continue
            if not key.startswith(include_prefix):
                continue
            try:
                series[key].append(float(value))
            except (TypeError, ValueError):
                series[key].append(math.nan)
    return x_values, series


def plot_series(
    x_values: Sequence[float],
    series: Mapping[str, Sequence[float]],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
    smooth: int = 1,
    show: bool = False,
) -> None:
    if not series:
        return
    plt.figure(figsize=(10, 6))
    for name, values in series.items():
        if len(x_values) != len(values):
            continue
        smoothed = moving_average(values, smooth)
        plt.plot(x_values, smoothed, label=name)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if show:
        plt.show()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path)
    plt.close()


def prepare_validation_series(records: Iterable[Mapping[str, object]]) -> Dict[str, Dict[str, Tuple[List[float], List[float]]]]:
    grouped: Dict[str, Dict[str, Tuple[List[float], List[float]]]] = defaultdict(lambda: defaultdict(lambda: ([], [])))
    for item in sorted(records, key=lambda r: r.get("iter", 0)):
        iteration = item.get("iter")
        dataset = item.get("dataset", "val")
        if iteration is None:
            continue
        for key, value in item.items():
            if not isinstance(key, str) or key in {"iter", "timestamp", "dataset"}:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            xs, ys = grouped[str(dataset)][key]
            xs.append(float(iteration))
            ys.append(numeric)
    return grouped


def summarize_validation(records: Iterable[Mapping[str, object]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for item in records:
        dataset = str(item.get("dataset", "val"))
        iteration = item.get("iter")
        for key, value in item.items():
            if not isinstance(key, str) or key in {"iter", "timestamp", "dataset"}:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            target = summary[dataset].setdefault(key, {"iter": iteration, "value": numeric})
            better_is_lower = "loss" in key.lower() or "lpips" in key.lower() or "err" in key.lower()
            if better_is_lower:
                if numeric < target["value"]:
                    target.update({"iter": iteration, "value": numeric})
            else:
                if numeric > target["value"]:
                    target.update({"iter": iteration, "value": numeric})
    return summary


def write_summary(path: Path, summary: Mapping[str, Mapping[str, Mapping[str, float]]]) -> None:
    if not summary:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    analytics_dir: Path
    if args.analytics_dir:
        analytics_dir = args.analytics_dir
    elif args.experiment:
        analytics_dir = args.experiment / "analytics"
    else:
        raise SystemExit("Either --analytics-dir or --experiment must be provided.")

    if not analytics_dir.exists():
        raise SystemExit(f"Analytics directory not found: {analytics_dir}")

    output_dir = args.output_dir or analytics_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_records = load_jsonl(analytics_dir / TRAINING_FILE)
    validation_records = load_jsonl(analytics_dir / VALIDATION_FILE)

    if training_records:
        x_losses, loss_series = prepare_series(training_records, include_prefix="l_")
        plot_series(
            x_losses,
            loss_series,
            title="Training Losses",
            ylabel="Loss",
            output_path=output_dir / "training_losses.png",
            smooth=args.smooth,
            show=args.show,
        )

        x_lr, lr_series = prepare_series(training_records, include_prefix="lr_")
        plot_series(
            x_lr,
            lr_series,
            title="Learning Rates",
            ylabel="LR",
            output_path=output_dir / "learning_rates.png",
            smooth=1,
            show=args.show,
        )

    if validation_records:
        grouped = prepare_validation_series(validation_records)
        for dataset, metrics in grouped.items():
            for metric_name, (xs, ys) in metrics.items():
                plot_series(
                    xs,
                    {metric_name: ys},
                    title=f"{dataset} — {metric_name}",
                    ylabel=metric_name,
                    output_path=output_dir / f"validation_{dataset}_{metric_name}.png",
                    smooth=args.smooth,
                    show=args.show,
                )
        summary = summarize_validation(validation_records)
        write_summary(output_dir / "validation_best_metrics.json", summary)

    if not training_records and not validation_records:
        raise SystemExit(
            f"No analytics records found in {analytics_dir}. Ensure training was run with the updated logging pipeline."
        )


if __name__ == "__main__":
    main()
