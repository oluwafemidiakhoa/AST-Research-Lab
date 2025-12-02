AST-LAB: Adaptive Sparse Training with PI control
=================================================

What this repo is
-----------------
Script-driven lab for Adaptive Sparse Training (AST) on ImageNet-scale datasets. We compute per-sample significance (loss + entropy), select active samples per batch, and regulate a threshold with a PI controller to hit a target activation rate (~20–30%).

Quick start
-----------
1) Prepare data: point `dataset.train_dir` and `dataset.val_dir` in a config to your ImageNet-like folders.
2) Run an experiment: `python scripts/run_experiment.py --config configs/imagenet_resnet50_ast.yaml`
3) Analyze: `python scripts/analyze_results.py --results-dir results --plot`
4) Sweep: `python scripts/run_sweep.py --configs configs/imagenet_resnet50_ast.yaml configs/imagenet_vitb16_ast.yaml`

Key files
---------
- `scripts/train_ast.py`: training loop with AST selection + PI control (AMP, grad accumulation, early stop, wall-clock cap).
- `scripts/run_experiment.py`: config loader/runner with simple overrides + JSONL logging.
- `scripts/run_sweep.py`: sequential sweep runner that writes summary JSONL.
- `scripts/analyze_results.py`: aggregates final_results_*.json and optional plots (activation, threshold, energy/step time).
- `scripts/download_imagenette.py`: fetch Imagenette from HF and export to ImageFolder for smoke tests.
- `configs/*.yaml`: experiment presets for ImageNet-style runs.
- `kaggle/*.ipynb`: Kaggle-friendly notebook stubs to reproduce runs.
- `requirements.txt`: deps (torch/torchvision, timm, datasets, matplotlib, pandas).
- `.env.example`: template for `KAGGLE_USERNAME`, `KAGGLE_KEY`, `HF_TOKEN` (do not commit .env).

Config notes
------------
- `ast.target_activation`: desired fraction of active samples per batch.
- `ast.kp`, `ast.ki`: PI gains; start small (kp~0.1, ki~0.01).
- `ast.kp_warmup_epochs` / `ast.ki_warmup_epochs`: ramp gains for stability early on.
- `ast.entropy_weight`: weight on entropy term in the significance score.
- `training.mixed_precision`: keep True for GPUs; disable on CPU-only debugging.
- `training.grad_accum_steps`: bump for small GPUs to simulate larger batch.
- `training.max_minutes`: wall-clock cap (handy on Kaggle).
- `dataset.type`: `folder` or `hf` (Hugging Face dataset via `datasets`).
- `metrics.base_flop_per_sample`: rough forward+backward FLOPs per sample to estimate compute savings; `metrics.gpu_tdp_watts` to get energy proxy.

Outputs
-------
Each run writes `results/final_results_*.json`:
```
{
  "best_accuracy": 0.0,
  "total_time_hours": 0.0,
  "epochs_completed": 0,
  "training_history": {
    "epoch": [],
    "train_acc": [],
    "val_acc": [],
    "energy_savings": [],
    "activation_rate": [],
    "time_min": [],
    "threshold": [],
    "step_time_ms": [],
    "est_energy_kj": [],
    "est_flops_g": []
  }
}
```

Secret handling
---------------
- Do not hardcode credentials. Copy `.env.example` to `.env` locally and set `KAGGLE_USERNAME`, `KAGGLE_KEY`, `HF_TOKEN`, or use your platform’s secret manager (Kaggle “Add Secret”, HF CLI login).
- Hugging Face: `HF_TOKEN` is picked up by `datasets` automatically. Alternatively run `huggingface-cli login` once in the environment. Scripts auto-load `.env` if present (via python-dotenv).
- Kaggle: place `kaggle.json` in `~/.kaggle` or set `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars. In Kaggle notebooks, add secrets via the UI and read into env vars (see notebook stub).

Roadmap ideas
-------------
- HF hub sharing for checkpoints and result artifacts.
- GPU perf counter integration for better energy estimation.
- Automated sweeps with Ax/Optuna for PI tuning.
# AST-Research-Lab
