"""
AST Training with Real-Time Energy Dashboard
Perfect for Kaggle notebooks and HuggingFace Spaces
"""

import time
from tqdm import tqdm
import torch
from adaptive_sparse_training import AdaptiveSparseTrainer, ASTConfig


class EnergyDashboard:
    """Real-time energy tracking with rich console output"""

    def __init__(self, gpu_name="T4", region="us-central1"):
        self.gpu_costs = {
            "T4": 0.35,      # $/hour
            "P100": 1.46,
            "V100": 2.48,
            "A100": 3.67,
        }
        self.gpu_tdp = {
            "T4": 70,        # Watts
            "P100": 250,
            "V100": 300,
            "A100": 400,
        }
        self.co2_intensity = 0.429  # kg CO2 per kWh (US avg)
        self.gpu_name = gpu_name
        self.cost_per_hour = self.gpu_costs.get(gpu_name, 0.35)
        self.tdp_watts = self.gpu_tdp.get(gpu_name, 70)

    def format_stats(self, epoch, total_epochs, stats):
        """Format training statistics with energy savings"""
        energy_kwh = stats['energy_kj'] / 3600  # kJ to kWh
        co2_g = energy_kwh * self.co2_intensity * 1000  # kg to grams
        cost = (stats['time_hours']) * self.cost_per_hour

        # Full training estimates (100% activation)
        full_energy_kwh = energy_kwh / (stats['activation_rate'] + 0.01)
        full_co2_g = full_energy_kwh * self.co2_intensity * 1000
        full_cost = full_energy_kwh * (self.cost_per_hour / (self.tdp_watts / 1000))

        savings_pct = stats['energy_savings'] * 100
        co2_saved = full_co2_g - co2_g
        cost_saved = full_cost - cost

        return f"""
╔══════════════════════════════════════════════════════════════╗
║ Epoch {epoch}/{total_epochs} | ⚡ Energy: {energy_kwh:.2f} kWh | 💰 ${cost:.2f}
║ 🎯 Accuracy: {stats['val_acc']:.1%} | 📊 Activation: {stats['activation_rate']:.1%}
╠══════════════════════════════════════════════════════════════╣
║ SAVINGS vs Full Training:
║   ⚡ Energy:  {savings_pct:.1f}% saved ({(full_energy_kwh-energy_kwh):.2f} kWh)
║   🌍 CO2:     {co2_saved:.0f}g avoided
║   💵 Cost:    ${cost_saved:.2f} saved
╚══════════════════════════════════════════════════════════════╝
"""


def train_with_dashboard(model, train_loader, val_loader, epochs=50,
                         target_activation=0.35, gpu_name="T4"):
    """
    Train with AST and display live energy dashboard

    Perfect for Kaggle notebooks - shows cost/energy savings in real-time
    """

    config = ASTConfig(
        target_activation_rate=target_activation,
        entropy_weight=1.0,
        use_mixed_precision=True,
        kp=0.1,
        ki=0.01,
    )

    trainer = AdaptiveSparseTrainer(model, train_loader, val_loader, config)
    dashboard = EnergyDashboard(gpu_name=gpu_name)

    print(f"\n🚀 Starting AST Training on {gpu_name}")
    print(f"🎯 Target Activation: {target_activation:.0%} (seeking {(1-target_activation):.0%} energy savings)")
    print("="*64)

    best_acc = 0
    total_energy_kj = 0
    total_time_hours = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train one epoch
        epoch_stats = trainer.train_epoch(epoch)
        val_acc = trainer.evaluate()

        epoch_time = time.time() - epoch_start
        epoch_hours = epoch_time / 3600
        total_time_hours += epoch_hours

        # Accumulate energy
        epoch_energy_kj = epoch_stats.get('energy_kj', epoch_time * 70)  # Fallback to 70W
        total_energy_kj += epoch_energy_kj

        stats = {
            'val_acc': val_acc,
            'activation_rate': epoch_stats.get('activation_rate', target_activation),
            'energy_savings': 1 - epoch_stats.get('activation_rate', target_activation),
            'energy_kj': total_energy_kj,
            'time_hours': total_time_hours,
        }

        # Display dashboard
        print(dashboard.format_stats(epoch, epochs, stats))

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"✨ New best accuracy: {best_acc:.1%}")

    # Final summary
    print("\n" + "="*64)
    print("🏁 TRAINING COMPLETE")
    print("="*64)
    print(f"Best Accuracy: {best_acc:.1%}")
    print(f"Total Energy: {total_energy_kj/3600:.2f} kWh")
    print(f"Total Time: {total_time_hours:.2f} hours")
    print(f"Energy Savings: ~{(1-target_activation)*100:.0f}%")
    print("="*64)

    return {
        'best_accuracy': best_acc,
        'total_energy_kwh': total_energy_kj / 3600,
        'total_time_hours': total_time_hours,
        'energy_savings_pct': (1-target_activation) * 100,
    }


if __name__ == "__main__":
    # Example usage for Kaggle
    import torchvision
    from torch.utils.data import DataLoader

    # Detect Kaggle environment
    import os
    is_kaggle = os.path.exists('/kaggle/working')
    gpu_name = "T4" if is_kaggle else "V100"

    # Example: CIFAR-10
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Create model
    import timm
    model = timm.create_model('resnet18', pretrained=True, num_classes=10)

    # Train with dashboard
    results = train_with_dashboard(
        model, train_loader, val_loader,
        epochs=50,
        target_activation=0.35,  # 65% energy savings
        gpu_name=gpu_name
    )

    print(f"\n💾 Saved ${results.get('cost_saved', 0):.2f} in GPU costs!")
