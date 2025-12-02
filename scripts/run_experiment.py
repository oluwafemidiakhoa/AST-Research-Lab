"""
Wrapper script to run a single AST experiment from a YAML/JSON config.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from train_ast import load_config, train, save_results

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def merge_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow override helper for simple CLI tweaks."""
    out = cfg.copy()
    for k, v in overrides.items():
        if v is None:
            continue
        # Support dotted keys: training.lr=0.05
        if "." in k:
            parts = k.split(".")
            cursor = out
            for p in parts[:-1]:
                cursor = cursor.setdefault(p, {})
            cursor[parts[-1]] = v
        else:
            out[k] = v
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Run AST experiment from config")
    p.add_argument("--config", required=True, help="YAML/JSON config path")
    p.add_argument("--output-dir", default=None, help="Override logging.output_dir")
    p.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    p.add_argument("--target-activation", type=float, default=None, help="Override ast.target_activation")
    p.add_argument("--jsonl-path", default=None, help="Override logging.jsonl_path")
    return p.parse_args()


def main():
    args = parse_args()
    if load_dotenv:
        load_dotenv()
    cfg = load_config(args.config)
    overrides = {
        "logging.output_dir": args.output_dir,
        "training.epochs": args.epochs,
        "ast.target_activation": args.target_activation,
        "logging.jsonl_path": args.jsonl_path,
    }
    cfg = merge_overrides(cfg, overrides)

    result = train(cfg)
    out_dir = cfg.get("logging", {}).get("output_dir", "results")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = save_results(result, out_dir)

    summary = {
        "config": args.config,
        "output": str(out_path),
        "best_accuracy": result["best_accuracy"],
        "epochs_completed": result["epochs_completed"],
        "target_activation": cfg["ast"]["target_activation"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
