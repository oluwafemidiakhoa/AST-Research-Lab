"""
Simple result analyzer for AST experiments.
Reads final_results_*.json files and aggregates metrics; optionally plots curves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_results(paths: List[Path]) -> pd.DataFrame:
    records = []
    for p in paths:
        with open(p, "r") as f:
            data = json.load(f)
        hist = data.get("training_history", {})
        record = {
            "path": str(p),
            "best_accuracy": data.get("best_accuracy", 0.0),
            "total_time_hours": data.get("total_time_hours", 0.0),
            "epochs_completed": data.get("epochs_completed", 0),
            "final_activation_rate": hist.get("activation_rate", [None])[-1] if hist else None,
            "final_threshold": hist.get("threshold", [None])[-1] if hist else None,
            "energy_kj_total": data.get("est_energy_kj_total"),
            "flops_g_total": data.get("est_flops_g_total"),
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def plot_history(result_path: Path, save_dir: Path):
    with open(result_path, "r") as f:
        data = json.load(f)
    hist = data["training_history"]
    epochs = hist["epoch"]

    plots = []
    plots.append(("Accuracy", [("train", hist["train_acc"]), ("val", hist["val_acc"])]))
    plots.append(("Activation rate", [("activation", hist["activation_rate"])]))
    plots.append(("Threshold", [("threshold", hist["threshold"])]))
    plots.append(("Energy savings (proxy)", [("energy_savings", hist.get("energy_savings", []))]))
    if "step_time_ms" in hist:
        plots.append(("Step time (ms)", [("step_time_ms", hist["step_time_ms"])]))
    if "est_energy_kj" in hist:
        plots.append(("Est energy (kJ)", [("est_energy_kj", hist["est_energy_kj"])]))
    if "est_flops_g" in hist:
        plots.append(("Est FLOPs (G)", [("est_flops_g", hist["est_flops_g"])]))

    n_plots = len(plots)
    rows = (n_plots + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows))
    axes = axes.flatten()

    for ax, (title, series) in zip(axes, plots):
        for label, values in series:
            ax.plot(epochs, values, label=label)
        ax.set_title(title)
        if len(series) > 1:
            ax.legend()
    for ax in axes[len(plots) :]:
        ax.axis("off")

    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"plot_{result_path.stem}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Analyze AST results JSON files")
    p.add_argument("--results-dir", default="results", help="Directory containing final_results_*.json")
    p.add_argument("--plot", action="store_true", help="Generate plots per result file")
    p.add_argument("--save-dir", default="results/plots", help="Where to save plots")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    paths = sorted(results_dir.glob("final_results_*.json"))
    if not paths:
        print(f"No result files found in {results_dir}")
        return
    df = load_results(paths)
    print(df.sort_values("best_accuracy", ascending=False).to_string(index=False))

    if args.plot:
        for p in paths:
            out = plot_history(p, Path(args.save_dir))
            print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
