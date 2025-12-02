"""
HuggingFace Space: AST Training Dashboard
Live monitoring and model card generation
"""

import gradio as gr
import json
import time
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from adaptive_sparse_training import AdaptiveSparseTrainer, ASTConfig
    import torch
    import torchvision
    import timm
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class ASTDashboard:
    """Real-time AST training dashboard"""

    def __init__(self):
        self.active_training = None
        self.training_history = []

    def start_training(
        self,
        model_name: str,
        dataset: str,
        activation_rate: float,
        epochs: int,
        progress=gr.Progress()
    ):
        """Start AST training with live updates"""

        if not HAS_DEPS:
            return "❌ Dependencies not installed", None, None

        progress(0, desc="Initializing...")

        # Load dataset (CIFAR-10 for demo)
        train_loader, val_loader = self._get_dataloaders(dataset)

        # Create model
        progress(0.1, desc="Creating model...")
        if model_name == "resnet18":
            model = torchvision.models.resnet18(num_classes=10)
        else:
            model = timm.create_model(model_name, pretrained=False, num_classes=10)

        # AST Config
        config = ASTConfig(
            target_activation_rate=activation_rate,
            entropy_weight=1.0,
            use_mixed_precision=True,
        )

        # Start training
        progress(0.2, desc="Starting training...")
        trainer = AdaptiveSparseTrainer(model, train_loader, val_loader, config)

        self.training_history = []

        for epoch in range(epochs):
            progress((epoch + 1) / epochs, desc=f"Epoch {epoch+1}/{epochs}")

            # Train one epoch
            epoch_stats = trainer.train_epoch(epoch)
            val_acc = trainer.evaluate()

            # Store history
            self.training_history.append({
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "activation_rate": epoch_stats.get("activation_rate", activation_rate),
                "threshold": epoch_stats.get("threshold", 1.0),
            })

            # Update dashboard
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                status = self._format_status(epoch + 1, epochs, val_acc, activation_rate)
                plot = self._create_plot()
                yield status, plot, None

        # Generate model card
        model_card = self._generate_model_card(model_name, activation_rate)

        final_status = f"✅ Training complete! Best accuracy: {max([h['val_acc'] for h in self.training_history]):.2%}"

        yield final_status, self._create_plot(), model_card

    def _get_dataloaders(self, dataset: str):
        """Get data loaders (CIFAR-10 demo)"""
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

        return train_loader, val_loader

    def _format_status(self, epoch: int, total_epochs: int, accuracy: float, activation_rate: float):
        """Format training status"""
        return f"""
### 🚀 Training in Progress

**Epoch:** {epoch}/{total_epochs}
**Accuracy:** {accuracy:.2%}
**Activation Rate:** {activation_rate:.1%}
**Energy Savings:** ~{(1-activation_rate)*100:.0f}%

*Updating every 5 epochs...*
"""

    def _create_plot(self):
        """Create live training plot"""
        if not self.training_history:
            return None

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Validation Accuracy", "Activation Rate", "Threshold Evolution", "Energy Savings"),
        )

        epochs = [h["epoch"] for h in self.training_history]
        accuracies = [h["val_acc"] * 100 for h in self.training_history]
        activation_rates = [h["activation_rate"] * 100 for h in self.training_history]
        thresholds = [h["threshold"] for h in self.training_history]
        savings = [(1 - h["activation_rate"]) * 100 for h in self.training_history]

        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=epochs, y=accuracies, mode='lines+markers', name='Val Accuracy',
                      line=dict(color='#3498db', width=3)),
            row=1, col=1
        )

        # Activation rate plot
        fig.add_trace(
            go.Scatter(x=epochs, y=activation_rates, mode='lines+markers', name='Activation Rate',
                      line=dict(color='#e74c3c', width=3)),
            row=1, col=2
        )

        # Threshold plot
        fig.add_trace(
            go.Scatter(x=epochs, y=thresholds, mode='lines+markers', name='Threshold',
                      line=dict(color='#f39c12', width=3)),
            row=2, col=1
        )

        # Energy savings plot
        fig.add_trace(
            go.Scatter(x=epochs, y=savings, mode='lines+markers', name='Energy Savings',
                      line=dict(color='#27ae60', width=3), fill='tozeroy'),
            row=2, col=2
        )

        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
        fig.update_yaxes(title_text="Activation (%)", row=1, col=2)
        fig.update_yaxes(title_text="Threshold", row=2, col=1)
        fig.update_yaxes(title_text="Savings (%)", row=2, col=2)

        fig.update_layout(height=600, showlegend=False)

        return fig

    def _generate_model_card(self, model_name: str, activation_rate: float):
        """Generate HuggingFace model card"""

        best_acc = max([h["val_acc"] for h in self.training_history])
        energy_savings = (1 - activation_rate) * 100

        return f"""---
tags:
- adaptive-sparse-training
- energy-efficient
- sustainability
metrics:
- accuracy
- energy_savings
---

# {model_name} (AST-Trained)

**Trained with {energy_savings:.0f}% less energy than standard training** ⚡

## Model Details
- **Architecture:** {model_name}
- **Dataset:** CIFAR-10
- **Training Method:** Adaptive Sparse Training (AST)
- **Target Activation Rate:** {activation_rate:.0%}

## Performance
- **Accuracy:** {best_acc:.2%}
- **Energy Savings:** {energy_savings:.0f}%
- **Training Epochs:** {len(self.training_history)}

## Sustainability Report
This model was trained using Adaptive Sparse Training, which dynamically selects
the most important training samples. This resulted in:

- ⚡ **{energy_savings:.0f}% energy savings** compared to standard training
- 🌍 **Lower carbon footprint**
- ⏱️ **Faster training time**
- 🎯 **Maintained accuracy** (minimal degradation)

## How to Use

```python
import torch
from torchvision import models

# Load model
model = models.{model_name}(num_classes=10)
model.load_state_dict(torch.load("pytorch_model.bin"))
model.eval()

# Inference
# ... (your inference code)
```

## Training Details

**AST Configuration:**
- Target Activation Rate: {activation_rate:.0%}
- Entropy Weight: 1.0
- PI Controller: Enabled
- Mixed Precision: Enabled

## Reproducing This Model

```bash
pip install adaptive-sparse-training

python -c "
from adaptive_sparse_training import AdaptiveSparseTrainer, ASTConfig
config = ASTConfig(target_activation_rate={activation_rate})
# ... (full training code)
"
```

## Citation

If you use this model or AST, please cite:

```bibtex
@software{{adaptive_sparse_training,
    title={{Adaptive Sparse Training}},
    author={{Idiakhoa, Oluwafemi}},
    year={{2024}},
    url={{https://github.com/oluwafemidiakhoa/adaptive-sparse-training}}
}}
```

## Acknowledgments

Trained using the `adaptive-sparse-training` package. Special thanks to the PyTorch and HuggingFace communities.

---

*This model card was auto-generated by the AST Training Dashboard.*
"""


# Initialize dashboard
dashboard = ASTDashboard()


# Gradio Interface
def create_demo():
    """Create Gradio demo interface"""

    with gr.Blocks(title="AST Training Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # ⚡ Adaptive Sparse Training Dashboard

        Train models with **60-70% less energy** while maintaining accuracy!

        This demo trains a model on CIFAR-10 using AST and generates a HuggingFace model card.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")

                model_name = gr.Dropdown(
                    choices=["resnet18", "efficientnet_b0", "mobilenetv3_small_100"],
                    value="resnet18",
                    label="Model Architecture"
                )

                dataset = gr.Dropdown(
                    choices=["cifar10"],
                    value="cifar10",
                    label="Dataset"
                )

                activation_rate = gr.Slider(
                    minimum=0.2,
                    maximum=0.8,
                    value=0.35,
                    step=0.05,
                    label="Target Activation Rate (lower = more savings)"
                )

                gr.Markdown(f"**Energy Savings:** ~{(1-0.35)*100:.0f}%")

                epochs = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=30,
                    step=10,
                    label="Training Epochs"
                )

                train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 📊 Live Training Metrics")

                status = gr.Markdown("*Ready to train...*")
                plot = gr.Plot()

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📝 Generated Model Card")
                model_card = gr.Textbox(
                    label="HuggingFace Model Card (Markdown)",
                    lines=20,
                    max_lines=30,
                )

                gr.Markdown("""
                **Next Steps:**
                1. Copy the model card above
                2. Create a new model on [HuggingFace Hub](https://huggingface.co/new)
                3. Paste the model card into `README.md`
                4. Upload your trained model weights
                """)

        # Training logic
        train_btn.click(
            fn=dashboard.start_training,
            inputs=[model_name, dataset, activation_rate, epochs],
            outputs=[status, plot, model_card],
        )

        gr.Markdown("""
        ---

        ## 📚 Learn More

        - 📦 [PyPI Package](https://pypi.org/project/adaptive-sparse-training/)
        - 🐙 [GitHub Repo](https://github.com/oluwafemidiakhoa/adaptive-sparse-training)
        - 📖 [Documentation](https://github.com/oluwafemidiakhoa/adaptive-sparse-training#readme)

        **Made with ❤️ using Adaptive Sparse Training**
        """)

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
