"""
Adaptive Sparse Training loop with a PI controller regulating the activation rate.

Features:
- Per-sample significance = loss + entropy_weight * entropy(logits)
- PI-regulated threshold to hit target activation rate
- Folder or Hugging Face datasets, optional timm models
- Energy/FLOP proxies and JSONL logging for sweeps
- Mixed precision, grad accumulation, early stop, wall-clock cap
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import yaml

try:
    import timm
except ImportError:
    timm = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# -----------------------------
# Utility structures
# -----------------------------

@dataclass
class PIController:
    target: float
    kp: float
    ki: float
    threshold: float
    min_threshold: float = 0.0
    integral: float = 0.0

    def update(self, measured: float) -> float:
        error = self.target - measured
        self.integral += error
        delta = self.kp * error + self.ki * self.integral
        self.threshold = max(self.min_threshold, self.threshold + delta)
        return self.threshold


def per_sample_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs + 1e-8)).sum(dim=1)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def get_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if name == "resnet50":
        weight_enum = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weight_enum)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet18":
        weight_enum = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weight_enum)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "vit_b_16":
        weight_enum = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weight_enum)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif timm is not None:
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model or timm not installed: {name}")
    return model


def build_transforms(image_size: int):
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tfms, val_tfms


class HFDataset(Dataset):
    """Wrap HF datasets to apply torchvision transforms and return tensors."""

    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"].convert("RGB")
        label = int(sample["label"])
        return self.transform(img), label


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    data_cfg = cfg["dataset"]
    image_size = data_cfg.get("image_size", 224)
    train_tfms, val_tfms = build_transforms(image_size)

    data_type = data_cfg.get("type", "folder")

    if data_type == "folder":
        train_dir = data_cfg.get("train_dir")
        val_dir = data_cfg.get("val_dir")
        train_set = datasets.ImageFolder(train_dir, transform=train_tfms)
        val_set = datasets.ImageFolder(val_dir, transform=val_tfms)
    elif data_type == "hf":
        if load_dataset is None:
            raise ImportError("Install datasets to use HF datasets (pip install datasets).")
        hf_name = data_cfg["name"]
        split_train = data_cfg.get("split_train", "train")
        split_val = data_cfg.get("split_val", "validation")
        cache_dir = data_cfg.get("cache_dir")
        train_hf = load_dataset(hf_name, split=split_train, cache_dir=cache_dir)
        val_hf = load_dataset(hf_name, split=split_val, cache_dir=cache_dir)
        train_set = HFDataset(train_hf, transform=train_tfms)
        val_set = HFDataset(val_hf, transform=val_tfms)
    else:
        raise ValueError(f"Unsupported dataset.type: {data_type}")

    def make_loader(dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=shuffle,
            num_workers=cfg["training"].get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )

    return make_loader(train_set, shuffle=True), make_loader(val_set, shuffle=False)


# -----------------------------
# Core training
# -----------------------------

@dataclass
class TrainState:
    epoch: int = 0
    best_acc: float = 0.0
    best_epoch: int = 0
    history: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "epoch": [],
            "train_acc": [],
            "val_acc": [],
            "energy_savings": [],
            "activation_rate": [],
            "time_min": [],
            "threshold": [],
            "step_time_ms": [],
            "est_energy_kj": [],
            "est_flops_g": [],
        }
    )


def append_jsonl(path: Path, record: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def ast_forward(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    controller: PIController,
    device: torch.device,
    entropy_weight: float,
    min_active: int,
    scaler: torch.cuda.amp.GradScaler,
    grad_accum_steps: int,
) -> Tuple[torch.Tensor, float, float, int, bool]:
    inputs, targets = batch
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    with torch.cuda.amp.autocast():
        logits = model(inputs)
        per_sample_loss = F.cross_entropy(logits, targets, reduction="none")
        ent = per_sample_entropy(logits)
        significance = per_sample_loss + entropy_weight * ent

    # Determine active set
    threshold = controller.threshold
    active_mask = significance >= threshold
    active_count = active_mask.sum().item()
    if active_count < min_active:
        k = min(min_active, significance.numel())
        topk_thresh = torch.topk(significance, k=k).values.min()
        active_mask = significance >= topk_thresh
        active_count = active_mask.sum().item()

    activation_rate = active_count / significance.numel()

    # PI update after measuring activation
    controller.update(activation_rate)

    if active_count == 0:
        return torch.tensor(0.0, device=device, requires_grad=False), activation_rate, threshold, active_count, False

    active_loss = per_sample_loss[active_mask].mean()

    loss_scaled = active_loss / grad_accum_steps
    scaler.scale(loss_scaled).backward()

    return active_loss.detach(), activation_rate, threshold, active_count, True


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    model.train()
    return correct / max(1, total)


def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(cfg)

    model = get_model(
        cfg["model"]["name"],
        num_classes=cfg["dataset"]["num_classes"],
        pretrained=cfg["model"].get("pretrained", False),
    ).to(device)

    criterion = nn.CrossEntropyLoss(reduction="none")

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["training"]["lr"],
        momentum=cfg["training"].get("momentum", 0.9),
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
        nesterov=True,
    )

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"].get("min_lr", 0.0),
    )

    controller = PIController(
        target=cfg["ast"]["target_activation"],
        kp=cfg["ast"].get("kp", 0.1),
        ki=cfg["ast"].get("ki", 0.01),
        threshold=cfg["ast"].get("initial_threshold", 1.0),
        min_threshold=cfg["ast"].get("min_threshold", 0.0),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"].get("mixed_precision", True))
    grad_accum_steps = cfg["training"].get("grad_accum_steps", 1)
    max_minutes = cfg["training"].get("max_minutes")
    early_stop_patience = cfg["training"].get("early_stop_patience", 10)

    metrics_cfg = cfg.get("metrics", {})
    base_flop_per_sample = metrics_cfg.get("base_flop_per_sample")  # FLOPs per full sample (forward+backward)
    gpu_tdp_watts = metrics_cfg.get("gpu_tdp_watts")
    jsonl_path = cfg.get("logging", {}).get("jsonl_path")
    jsonl_path = Path(jsonl_path) if jsonl_path else None

    state = TrainState()
    start_time = time.time()
    no_improve_epochs = 0
    total_energy_kj = 0.0
    total_flops = 0.0

    for epoch in range(cfg["training"]["epochs"]):
        state.epoch = epoch
        correct = 0
        total = 0
        activation_meter = []
        threshold_meter = []
        step_time_ms = []
        optimizer.zero_grad(set_to_none=True)
        any_active_in_accum = False

        # Optional PI warmup for stability
        kp_base = cfg["ast"].get("kp", 0.1)
        ki_base = cfg["ast"].get("ki", 0.01)
        kp_warmup = cfg["ast"].get("kp_warmup_epochs", 0)
        ki_warmup = cfg["ast"].get("ki_warmup_epochs", 0)
        if kp_warmup > 0:
            controller.kp = kp_base * min(1.0, (epoch + 1) / kp_warmup)
        if ki_warmup > 0:
            controller.ki = ki_base * min(1.0, (epoch + 1) / ki_warmup)

        for step, batch in enumerate(train_loader):
            step_t0 = time.perf_counter()

            loss_val, activation_rate, prev_threshold, active_count, used_active = ast_forward(
                model,
                batch,
                controller,
                device,
                entropy_weight=cfg["ast"].get("entropy_weight", 1.0),
                min_active=cfg["ast"].get("min_active", 4),
                scaler=scaler,
                grad_accum_steps=grad_accum_steps,
            )

            activation_meter.append(activation_rate)
            threshold_meter.append(prev_threshold)

            any_active_in_accum = any_active_in_accum or used_active
            if (step + 1) % grad_accum_steps == 0:
                if any_active_in_accum:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                any_active_in_accum = False

            if used_active:
                with torch.no_grad():
                    inputs, targets = batch
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    preds = model(inputs).argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)

            step_time = time.perf_counter() - step_t0
            step_time_ms.append(step_time * 1000.0)

            if base_flop_per_sample is not None:
                effective_samples = active_count
                total_flops += base_flop_per_sample * effective_samples
            if gpu_tdp_watts is not None:
                total_energy_kj += (step_time * gpu_tdp_watts) / 1000.0

            if max_minutes and (time.time() - start_time) / 60 > max_minutes:
                break

        if any_active_in_accum:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_acc = correct / max(1, total)
        val_acc = evaluate(model, val_loader, device)
        lr_scheduler.step()

        mean_activation = float(sum(activation_meter) / max(1, len(activation_meter)))
        mean_threshold = float(sum(threshold_meter) / max(1, len(threshold_meter)))
        mean_step_time_ms = float(sum(step_time_ms) / max(1, len(step_time_ms)))

        elapsed_min = (time.time() - start_time) / 60
        energy_savings = 1.0 - mean_activation
        est_energy_kj = total_energy_kj
        est_flops_g = total_flops / 1e9 if total_flops else 0.0

        state.history["epoch"].append(epoch)
        state.history["train_acc"].append(train_acc)
        state.history["val_acc"].append(val_acc)
        state.history["energy_savings"].append(energy_savings)
        state.history["activation_rate"].append(mean_activation)
        state.history["time_min"].append(elapsed_min)
        state.history["threshold"].append(mean_threshold)
        state.history["step_time_ms"].append(mean_step_time_ms)
        state.history["est_energy_kj"].append(est_energy_kj)
        state.history["est_flops_g"].append(est_flops_g)

        if jsonl_path:
            append_jsonl(
                jsonl_path,
                {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "activation_rate": mean_activation,
                    "threshold": mean_threshold,
                    "energy_savings": energy_savings,
                    "step_time_ms": mean_step_time_ms,
                    "est_energy_kj": est_energy_kj,
                    "est_flops_g": est_flops_g,
                },
            )

        if val_acc > state.best_acc:
            state.best_acc = val_acc
            state.best_epoch = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if max_minutes and elapsed_min > max_minutes:
            break
        if no_improve_epochs >= early_stop_patience:
            break

    result = {
        "best_accuracy": state.best_acc,
        "total_time_hours": (time.time() - start_time) / 3600,
        "epochs_completed": state.epoch + 1,
        "training_history": state.history,
        "est_energy_kj_total": total_energy_kj,
        "est_flops_g_total": total_flops / 1e9 if total_flops else 0.0,
    }
    return result


def save_results(result: Dict[str, Any], output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = Path(output_dir) / f"final_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive Sparse Training with PI control")
    p.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config")
    return p.parse_args()


def main():
    args = parse_args()
    if load_dotenv:
        load_dotenv()
    cfg = load_config(args.config)
    result = train(cfg)
    out_path = save_results(result, cfg.get("logging", {}).get("output_dir", "results"))
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
