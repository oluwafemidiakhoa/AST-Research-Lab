"""
Sweep runner: execute multiple AST configs and log summaries to JSONL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from train_ast import load_config, save_results, train
from run_experiment import merge_overrides

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def append_jsonl(path: Path, record: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_single(cfg_path: Path, overrides: Dict[str, Any], sweep_log: Path):
    cfg = load_config(str(cfg_path))
    cfg = merge_overrides(cfg, overrides)
    result = train(cfg)
    out_dir = cfg.get("logging", {}).get("output_dir", "results")
    out_path = save_results(result, out_dir)
    summary = {
        "config": str(cfg_path),
        "output": str(out_path),
        "best_accuracy": result["best_accuracy"],
        "epochs_completed": result["epochs_completed"],
        "target_activation": cfg["ast"]["target_activation"],
        "est_energy_kj_total": result.get("est_energy_kj_total"),
        "est_flops_g_total": result.get("est_flops_g_total"),
    }
    append_jsonl(sweep_log, summary)
    print(json.dumps(summary, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Run multiple AST configs sequentially")
    p.add_argument("--configs", nargs="+", required=True, help="List of YAML/JSON configs")
    p.add_argument("--output-log", default="results/sweep_log.jsonl", help="JSONL summary output path")
    p.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    p.add_argument("--target-activation", type=float, default=None, help="Override ast.target_activation")
    p.add_argument("--output-dir", default=None, help="Override logging.output_dir")
    return p.parse_args()


def main():
    args = parse_args()
    if load_dotenv:
        load_dotenv()
    sweep_log = Path(args.output_log)
    overrides = {
        "training.epochs": args.epochs,
        "ast.target_activation": args.target_activation,
        "logging.output_dir": args.output_dir,
    }
    for cfg_path in args.configs:
        run_single(Path(cfg_path), overrides, sweep_log)


if __name__ == "__main__":
    main()
