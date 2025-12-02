"""
Benchmark AST vs Standard Training
Generates publication-ready comparison tables and plots
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

try:
    import timm
except ImportError:
    timm = None

try:
    from adaptive_sparse_training import AdaptiveSparseTrainer, ASTConfig
    HAS_AST_PACKAGE = True
except ImportError:
    HAS_AST_PACKAGE = False
    print("⚠️  adaptive-sparse-training not installed. Install: pip install adaptive-sparse-training")

import matplotlib.pyplot as plt
import pandas as pd


class StandardTrainer:
    """Standard training baseline for comparison"""

    def __init__(self, model, train_loader, val_loader, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50
        )

    def train_epoch(self):
        self.model.train()
        correct = 0
        total = 0
        total_loss = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return correct / total, total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return correct / total

    def train(self, epochs=50, warmup_epochs=0):
        history = {
            "epoch": [],
            "train_acc": [],
            "val_acc": [],
            "time_min": [],
        }

        start_time = time.time()
        best_acc = 0

        for epoch in range(epochs):
            train_acc, train_loss = self.train_epoch()
            val_acc = self.evaluate()
            self.scheduler.step()

            elapsed = (time.time() - start_time) / 60

            history["epoch"].append(epoch)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["time_min"].append(elapsed)

            if val_acc > best_acc:
                best_acc = val_acc

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")

        return {
            "best_accuracy": best_acc,
            "total_time_hours": (time.time() - start_time) / 3600,
            "epochs_completed": epochs,
            "training_history": history,
        }


def get_cifar10_loaders(batch_size=128):
    """Get CIFAR-10 train/val loaders"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


def run_benchmark(
    model_name: str = "resnet18",
    dataset: str = "cifar10",
    epochs: int = 50,
    ast_activation_rates: List[float] = [0.25, 0.35, 0.50],
    run_baseline: bool = True,
    output_dir: str = "benchmark_results",
):
    """
    Run comprehensive benchmark: AST vs baseline

    Args:
        model_name: Model architecture
        dataset: Dataset name
        epochs: Training epochs
        ast_activation_rates: List of AST activation rates to test
        run_baseline: Whether to run baseline (100% training)
        output_dir: Output directory for results
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    # Prepare data
    if dataset == "cifar10":
        train_loader, val_loader = get_cifar10_loaders(batch_size=128)
        num_classes = 10
    else:
        raise ValueError(f"Dataset {dataset} not supported yet")

    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Run baseline
    if run_baseline:
        print("\n" + "="*60)
        print("📊 BASELINE: Standard Training (100% samples)")
        print("="*60)

        if model_name == "resnet18":
            model = torchvision.models.resnet18(num_classes=num_classes)
        elif timm:
            model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        else:
            raise ValueError(f"Model {model_name} not available")

        trainer = StandardTrainer(model, train_loader, val_loader, device=str(device))
        baseline_result = trainer.train(epochs=epochs)

        results["baseline"] = baseline_result
        print(f"✅ Baseline: {baseline_result['best_accuracy']:.2%} accuracy")
        print(f"   Time: {baseline_result['total_time_hours']:.2f}h")

    # Run AST with different activation rates
    for activation_rate in ast_activation_rates:
        print("\n" + "="*60)
        print(f"⚡ AST: Activation Rate = {activation_rate:.0%} ({(1-activation_rate)*100:.0f}% savings)")
        print("="*60)

        if not HAS_AST_PACKAGE:
            print("❌ Skipping AST (package not installed)")
            continue

        # Fresh model
        if model_name == "resnet18":
            model = torchvision.models.resnet18(num_classes=num_classes)
        elif timm:
            model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

        config = ASTConfig(
            target_activation_rate=activation_rate,
            entropy_weight=1.0,
            kp=0.1,
            ki=0.01,
            use_mixed_precision=True,
        )

        trainer = AdaptiveSparseTrainer(
            model, train_loader, val_loader, config, device=device
        )
        ast_result = trainer.train(epochs=epochs, warmup_epochs=5)

        results[f"ast_{activation_rate}"] = ast_result
        print(f"✅ AST ({activation_rate:.0%}): {ast_result['best_accuracy']:.2%} accuracy")
        print(f"   Time: {ast_result['total_time_hours']:.2f}h")
        print(f"   Energy Savings: {ast_result.get('energy_savings', (1-activation_rate))*100:.1f}%")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"benchmark_{model_name}_{dataset}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Generate comparison table
    generate_comparison_table(results, output_path, model_name, dataset)

    # Generate plots
    generate_comparison_plots(results, output_path, model_name, dataset)

    return results


def generate_comparison_table(results: Dict, output_dir: Path, model: str, dataset: str):
    """Generate markdown comparison table"""

    table_data = []

    baseline = results.get("baseline")
    if baseline:
        table_data.append({
            "Method": "Baseline (100%)",
            "Accuracy": f"{baseline['best_accuracy']:.2%}",
            "Time (h)": f"{baseline['total_time_hours']:.2f}",
            "Energy Savings": "0%",
            "Speedup": "1.0×",
        })

    for key, result in results.items():
        if key.startswith("ast_"):
            activation_rate = float(key.split("_")[1])
            savings = (1 - activation_rate) * 100

            if baseline:
                speedup = baseline['total_time_hours'] / result['total_time_hours']
                acc_diff = result['best_accuracy'] - baseline['best_accuracy']
                acc_str = f"{result['best_accuracy']:.2%} ({acc_diff:+.2%})"
            else:
                speedup = 1 / activation_rate
                acc_str = f"{result['best_accuracy']:.2%}"

            table_data.append({
                "Method": f"AST ({activation_rate:.0%})",
                "Accuracy": acc_str,
                "Time (h)": f"{result['total_time_hours']:.2f}",
                "Energy Savings": f"{savings:.0f}%",
                "Speedup": f"{speedup:.2f}×",
            })

    df = pd.DataFrame(table_data)

    # Save as markdown
    md_path = output_dir / f"comparison_{model}_{dataset}.md"
    with open(md_path, "w") as f:
        f.write(f"# AST Benchmark Results: {model} on {dataset}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n**Key Findings:**\n")
        if baseline and len(table_data) > 1:
            best_ast = table_data[-1]  # Last AST run
            f.write(f"- AST achieves {best_ast['Accuracy'].split('(')[0]} accuracy\n")
            f.write(f"- {best_ast['Energy Savings']} energy savings compared to baseline\n")
            f.write(f"- {best_ast['Speedup']} faster training\n")

    print(f"\n📊 Comparison table saved to: {md_path}")
    print("\n" + df.to_string(index=False))


def generate_comparison_plots(results: Dict, output_dir: Path, model: str, dataset: str):
    """Generate comparison plots"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Accuracy comparison
    ax = axes[0, 0]
    methods = []
    accuracies = []

    baseline = results.get("baseline")
    if baseline:
        methods.append("Baseline\n(100%)")
        accuracies.append(baseline['best_accuracy'] * 100)

    for key, result in results.items():
        if key.startswith("ast_"):
            rate = float(key.split("_")[1])
            methods.append(f"AST\n({rate:.0%})")
            accuracies.append(result['best_accuracy'] * 100)

    colors = ['#e74c3c' if 'Baseline' in m else '#3498db' for m in methods]
    ax.bar(methods, accuracies, color=colors, alpha=0.7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Final Accuracy Comparison")
    ax.set_ylim([min(accuracies) - 5, max(accuracies) + 2])
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Training time comparison
    ax = axes[0, 1]
    times = []
    if baseline:
        times.append(baseline['total_time_hours'])
    for key, result in results.items():
        if key.startswith("ast_"):
            times.append(result['total_time_hours'])

    ax.bar(methods, times, color=colors, alpha=0.7)
    ax.set_ylabel("Training Time (hours)")
    ax.set_title("Training Time Comparison")
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Energy savings
    ax = axes[1, 0]
    savings = []
    if baseline:
        savings.append(0)
    for key, result in results.items():
        if key.startswith("ast_"):
            rate = float(key.split("_")[1])
            savings.append((1 - rate) * 100)

    ax.bar(methods, savings, color=colors, alpha=0.7)
    ax.set_ylabel("Energy Savings (%)")
    ax.set_title("Energy Savings vs Baseline")
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Pareto curve (Accuracy vs Energy Savings)
    ax = axes[1, 1]
    if baseline:
        ax.scatter([0], [baseline['best_accuracy'] * 100],
                  s=200, c='red', marker='o', label='Baseline', zorder=3)

    ast_savings = []
    ast_accs = []
    for key, result in results.items():
        if key.startswith("ast_"):
            rate = float(key.split("_")[1])
            ast_savings.append((1 - rate) * 100)
            ast_accs.append(result['best_accuracy'] * 100)

    if ast_savings:
        ax.scatter(ast_savings, ast_accs, s=200, c='blue', marker='s', label='AST', zorder=3)
        ax.plot(ast_savings, ast_accs, 'b--', alpha=0.5)

    ax.set_xlabel("Energy Savings (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Pareto Frontier: Accuracy vs Energy")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f"comparison_{model}_{dataset}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 Plots saved to: {plot_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark AST vs Standard Training")
    parser.add_argument("--model", default="resnet18", help="Model architecture")
    parser.add_argument("--dataset", default="cifar10", help="Dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--activation-rates", nargs="+", type=float,
                       default=[0.25, 0.35, 0.50], help="AST activation rates to test")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline run")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")

    args = parser.parse_args()

    print("🚀 AST Benchmark Suite")
    print(f"   Model: {args.model}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Epochs: {args.epochs}")
    print(f"   AST Rates: {args.activation_rates}")
    print()

    results = run_benchmark(
        model_name=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        ast_activation_rates=args.activation_rates,
        run_baseline=not args.no_baseline,
        output_dir=args.output_dir,
    )

    print("\n" + "="*60)
    print("✅ BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
