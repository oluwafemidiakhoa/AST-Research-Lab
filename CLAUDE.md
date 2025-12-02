# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is AST-LAB: a script-driven laboratory for Adaptive Sparse Training (AST) on ImageNet-scale datasets. The core innovation is using per-sample significance scores (loss + entropy) to selectively activate training samples, with a PI controller dynamically regulating the threshold to hit target activation rates (typically 20-30%). This reduces computational cost while maintaining model accuracy.

## Development Commands

### Running Experiments

**Single experiment:**
```bash
python scripts/run_experiment.py --config configs/imagenet_resnet50_ast.yaml
```

**With overrides:**
```bash
python scripts/run_experiment.py --config configs/imagenet_resnet50_ast.yaml --epochs 50 --target-activation 0.3
```

**Multiple configs (sweep):**
```bash
python scripts/run_sweep.py --configs configs/imagenet_resnet50_ast.yaml configs/imagenet_vitb16_ast.yaml
```

**Direct training (advanced):**
```bash
python scripts/train_ast.py --config configs/imagenet_resnet50_ast.yaml
```

### Analysis

**Analyze results:**
```bash
python scripts/analyze_results.py --results-dir results
```

**With plots:**
```bash
python scripts/analyze_results.py --results-dir results --plot --save-dir results/plots
```

### Data Preparation

**Download Imagenette for smoke tests:**
```bash
python scripts/download_imagenette.py
```

## Architecture & Key Components

### Training Pipeline (scripts/train_ast.py)

The core training loop implements the AST algorithm:

1. **Per-sample significance scoring** (`ast_forward` function, line 205-248):
   - Computes `significance = loss + entropy_weight * entropy(logits)` for each sample
   - Applies threshold to determine active set
   - Enforces minimum active samples via top-k selection if needed
   - Only backpropagates through active samples

2. **PI Controller** (`PIController` class, line 49-63):
   - Dynamically adjusts threshold to maintain target activation rate
   - Uses proportional-integral control: `delta = kp * error + ki * integral`
   - Optional warmup ramps for stability early in training

3. **Metrics tracking**:
   - Energy proxy: `step_time * gpu_tdp_watts`
   - FLOP estimation: `base_flop_per_sample * active_samples`
   - Activation rate, threshold evolution, step time per epoch

### Config System

All experiments are configured via YAML files in `configs/`. Key sections:

- **model**: Name (resnet50, vit_b_16, or any timm model), pretrained flag
- **dataset**: Type (folder/hf), paths, num_classes, image_size
- **ast**: PI gains (kp, ki), target_activation, entropy_weight, warmup schedules
- **training**: Learning rate, batch_size, epochs, grad_accum_steps, early stopping, wall-clock cap
- **metrics**: base_flop_per_sample, gpu_tdp_watts for energy/compute estimates
- **logging**: output_dir, jsonl_path for per-epoch logs

### Data Loading (scripts/train_ast.py, line 135-171)

Supports two dataset types:
- `folder`: Standard ImageFolder layout (train/val directories with class subdirectories)
- `hf`: Hugging Face datasets via `datasets` library

The `HFDataset` wrapper (line 118-132) adapts HF datasets to PyTorch by applying torchvision transforms.

### Entry Points

- **run_experiment.py**: Single config runner with simple CLI overrides (dotted keys like `training.lr=0.05`)
- **run_sweep.py**: Sequential sweep across multiple configs, writes summary JSONL
- **analyze_results.py**: Aggregates `final_results_*.json`, generates plots (accuracy, activation, threshold, energy)

### Output Format

Each run writes `results/final_results_<timestamp>.json`:
```json
{
  "best_accuracy": 0.0,
  "total_time_hours": 0.0,
  "epochs_completed": 0,
  "training_history": {
    "epoch": [],
    "train_acc": [],
    "val_acc": [],
    "activation_rate": [],
    "threshold": [],
    "energy_savings": [],
    "step_time_ms": [],
    "est_energy_kj": [],
    "est_flops_g": []
  }
}
```

Optional per-epoch JSONL logs written to `logging.jsonl_path` for real-time monitoring.

## Important Implementation Details

### PI Controller Warmup
The PI gains can be ramped during early epochs for stability:
- `ast.kp_warmup_epochs`: linearly ramp kp from 0 to target over N epochs
- `ast.ki_warmup_epochs`: linearly ramp ki from 0 to target over N epochs

This prevents aggressive threshold oscillations before the model has learned meaningful loss/entropy distributions.

### Gradient Accumulation
When `training.grad_accum_steps > 1`, the training loop accumulates gradients across multiple forward passes before stepping the optimizer. The `any_active_in_accum` flag ensures optimizer.step() only runs if at least one batch in the accumulation window had active samples.

### Mixed Precision
AMP is enabled by default (`training.mixed_precision: true`). The `GradScaler` handles loss scaling for active samples. Disable on CPU-only debugging.

### Wall-Clock & Early Stopping
- `training.max_minutes`: hard cap on total run time (useful for Kaggle notebooks with time limits)
- `training.early_stop_patience`: stop if validation accuracy doesn't improve for N epochs

### Model Support
- Built-in: resnet50, resnet18, vit_b_16 via torchvision.models
- Extended: any model from `timm` library via `timm.create_model()`
- Custom head replacement for non-ImageNet num_classes (see `get_model` function, line 78-95)

### Energy & FLOP Estimation
Rough proxies for compute savings:
- `metrics.base_flop_per_sample`: forward+backward FLOPs for one sample (e.g., 7.8e9 for ResNet-50)
- `metrics.gpu_tdp_watts`: GPU thermal design power for energy calculation
- Actual FLOPs = `base_flop_per_sample * sum(active_samples_per_step)`
- Energy (kJ) ≈ `sum(step_time * gpu_tdp_watts) / 1000`

## Environment & Secrets

All scripts use `python-dotenv` to load `.env` if present. Required secrets:
- `KAGGLE_USERNAME`, `KAGGLE_KEY`: for Kaggle API (if downloading datasets via kagglehub)
- `HF_TOKEN`: for Hugging Face datasets (alternative: `huggingface-cli login`)

Never commit `.env`. The template is `.env.example`.

## Typical Workflow

1. **Prepare dataset**: For ImageNet-style datasets, point `dataset.train_dir` and `dataset.val_dir` to your local ImageFolder. For HuggingFace datasets, set `dataset.type: hf` and `dataset.name`.

2. **Configure experiment**: Copy/edit a config in `configs/`. Start with low `ast.kp` (~0.1) and `ast.ki` (~0.01) for stability.

3. **Run experiment**: Use `run_experiment.py` for single runs, `run_sweep.py` for parameter sweeps.

4. **Analyze results**: `analyze_results.py` aggregates metrics and plots training curves.

5. **Iterate**: Adjust PI gains, target_activation, or entropy_weight based on activation rate stability and final accuracy.

## Testing & Validation

For quick smoke tests, use the CIFAR-10 configs or download Imagenette:
```bash
python scripts/download_imagenette.py
python scripts/run_experiment.py --config configs/imagenette_hf_ast.yaml --epochs 10
```

The `configs/cifar10_resnet18_fast.yaml` preset is designed for rapid iteration on smaller datasets.
