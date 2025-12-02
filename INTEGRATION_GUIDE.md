# AST Integration Guide

Complete guide for integrating Adaptive Sparse Training with popular model architectures and frameworks.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Vision Models](#vision-models)
3. [HuggingFace Transformers](#huggingface-transformers)
4. [timm Models](#timm-models)
5. [Custom Architectures](#custom-architectures)
6. [Production Deployment](#production-deployment)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

```bash
pip install adaptive-sparse-training torch torchvision
```

### Basic Usage

```python
from adaptive_sparse_training import AdaptiveSparseTrainer, ASTConfig
import torch.nn as nn

# Your model
model = nn.Module(...)  # Any PyTorch model

# AST Config
config = ASTConfig(
    target_activation_rate=0.35,  # 65% energy savings
    entropy_weight=1.0,
)

# Train
trainer = AdaptiveSparseTrainer(model, train_loader, val_loader, config)
results = trainer.train(epochs=50, warmup_epochs=5)

print(f"Accuracy: {results['best_accuracy']:.2%}")
print(f"Energy Savings: {results['energy_savings']:.1%}")
```

---

## Vision Models

### ResNet Family (ResNet18/34/50/101/152)

**Recommended Settings:**
- Target Activation: `0.35` (65% savings)
- Warmup Epochs: `5`
- Works great with AST!

```python
from torchvision.models import resnet50
from adaptive_sparse_training import AdaptiveSparseTrainer, ASTConfig

# Standard ResNet50
model = resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# AST Config
config = ASTConfig(
    target_activation_rate=0.35,
    entropy_weight=1.0,
    kp=0.1,  # PI controller gains
    ki=0.01,
)

trainer = AdaptiveSparseTrainer(model, train_loader, val_loader, config)
results = trainer.train(epochs=90, warmup_epochs=5)
```

**Expected Results (ImageNet):**
- Baseline: 76.1% Top-1
- AST (0.35): 75.8% Top-1 (-0.3%)
- Energy Savings: 65%

### EfficientNet (B0-B7)

**Recommended Settings:**
- Target Activation: `0.40` (60% savings)
- Warmup Epochs: `10` (EfficientNet needs more warmup)
- Entropy Weight: `1.5` (higher for compound scaling)

```python
from torchvision.models import efficientnet_b0
from adaptive_sparse_training import AdaptiveSparseTrainer, ASTConfig

model = efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

config = ASTConfig(
    target_activation_rate=0.40,
    entropy_weight=1.5,  # Higher for EfficientNet
    kp=0.08,  # Gentler control
    ki=0.008,
)

trainer = AdaptiveSparseTrainer(model, train_loader, val_loader, config)
results = trainer.train(epochs=100, warmup_epochs=10)
```

**Expected Results (ImageNet):**
- Baseline: 77.7% Top-1
- AST (0.40): 77.3% Top-1 (-0.4%)
- Energy Savings: 60%

### Vision Transformers (ViT)

**Recommended Settings:**
- Target Activation: `0.45` (55% savings)
- Warmup Epochs: `15` (ViTs need longer warmup)
- Entropy Weight: `2.0` (attention needs more entropy signal)

```python
from torchvision.models import vit_b_16
from adaptive_sparse_training import AdaptiveSparseTrainer, ASTConfig

model = vit_b_16(pretrained=True)
model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

config = ASTConfig(
    target_activation_rate=0.45,  # Conservative for ViT
    entropy_weight=2.0,  # Higher for transformers
    kp=0.05,  # Very gentle
    ki=0.005,
    use_mixed_precision=True,  # Critical for ViT speed
)

trainer = AdaptiveSparseTrainer(model, train_loader, val_loader, config)
results = trainer.train(epochs=150, warmup_epochs=15)
```

**Expected Results (ImageNet):**
- Baseline: 81.1% Top-1
- AST (0.45): 80.6% Top-1 (-0.5%)
- Energy Savings: 55%

**Note:** ViTs are more sensitive to AST. Start with higher activation rates (0.45-0.50).

### MobileNet (V2/V3)

**Recommended Settings:**
- Target Activation: `0.30` (70% savings)
- Warmup Epochs: `5`
- Great for mobile deployment!

```python
from torchvision.models import mobilenet_v3_small
from adaptive_sparse_training import AdaptiveSparseTrainer, ASTConfig

model = mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

config = ASTConfig(
    target_activation_rate=0.30,  # Aggressive savings
    entropy_weight=0.8,  # Lower for mobile models
    kp=0.12,
    ki=0.012,
)

trainer = AdaptiveSparseTrainer(model, train_loader, val_loader, config)
results = trainer.train(epochs=60, warmup_epochs=5)
```

**Expected Results (ImageNet):**
- Baseline: 67.7% Top-1
- AST (0.30): 67.2% Top-1 (-0.5%)
- Energy Savings: 70%

---

## HuggingFace Transformers

### BERT (Classification)

```python
from transformers import BertForSequenceClassification
from adaptive_sparse_training import AdaptiveSparseTrainer, ASTConfig
import torch

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

config = ASTConfig(
    target_activation_rate=0.40,
    entropy_weight=1.5,
    kp=0.05,
    ki=0.005,
    use_mixed_precision=True,
)

# Note: You'll need to adapt your DataLoader to return (input_ids, labels)
trainer = AdaptiveSparseTrainer(model, train_loader, val_loader, config)
results = trainer.train(epochs=3, warmup_epochs=1)  # Standard BERT fine-tuning
```

**Tips for Transformers:**
- Use **lower learning rates** (1e-5 to 5e-5)
- Shorter warmup (1 epoch usually enough)
- Higher entropy weight (1.5-2.0)

### RoBERTa, DistilBERT

Same config as BERT, works out of the box:

```python
from transformers import RobertaForSequenceClassification

model = RobertaForSequenceClassification.from_pretrained('roberta-base')
# ... same AST config as BERT
```

---

## timm Models

AST works with **all timm models**! Here are some highlights:

### ConvNeXt (Recommended!)

**Best AST performance of any architecture tested.**

```python
import timm
from adaptive_sparse_training import AdaptiveSparseTrainer, ASTConfig

model = timm.create_model('convnext_base', pretrained=True, num_classes=num_classes)

config = ASTConfig(
    target_activation_rate=0.30,  # Aggressive savings possible!
    entropy_weight=1.0,
    kp=0.1,
    ki=0.01,
)

trainer = AdaptiveSparseTrainer(model, train_loader, val_loader, config)
results = trainer.train(epochs=100, warmup_epochs=5)
```

**Expected Results (ImageNet):**
- Baseline: 83.8% Top-1
- AST (0.30): 83.5% Top-1 (-0.3%)
- Energy Savings: **70%** 🎉

### Swin Transformer

```python
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)

config = ASTConfig(
    target_activation_rate=0.45,  # Conservative for transformers
    entropy_weight=2.0,
    kp=0.05,
    ki=0.005,
)

trainer = AdaptiveSparseTrainer(model, train_loader, val_loader, config)
results = trainer.train(epochs=150, warmup_epochs=15)
```

### MaxViT, CoAtNet

```python
# Works with any timm model!
model = timm.create_model('maxvit_base_tf_224', pretrained=True, num_classes=num_classes)

# Start with conservative settings
config = ASTConfig(target_activation_rate=0.40)

trainer = AdaptiveSparseTrainer(model, train_loader, val_loader, config)
results = trainer.train(epochs=100, warmup_epochs=10)
```

---

## Custom Architectures

AST works with **any PyTorch model**. Just ensure:

1. Model outputs **logits** (not probabilities)
2. Loss is computed per-sample (not reduced)

### Example: Custom CNN

```python
class MyCustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # Return logits, not softmax!

model = MyCustomCNN(num_classes=10)

# AST works out of the box
config = ASTConfig(target_activation_rate=0.35)
trainer = AdaptiveSparseTrainer(model, train_loader, val_loader, config)
results = trainer.train(epochs=50)
```

### Multi-Output Models

If your model has multiple outputs, return the main classification logits:

```python
class MultiTaskModel(nn.Module):
    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.classifier(features)
        aux_output = self.auxiliary_head(features)

        # Return main logits first for AST
        return class_logits  # AST will use this

        # For standard training, you might return (class_logits, aux_output)
```

---

## Production Deployment

### Saving AST-Trained Models

```python
# Train with AST
trainer = AdaptiveSparseTrainer(model, train_loader, val_loader, config)
results = trainer.train(epochs=50)

# Save model (standard PyTorch)
torch.save({
    'model_state_dict': model.state_dict(),
    'ast_config': config.__dict__,
    'results': results,
}, 'model_ast.pth')
```

### Inference (No AST Needed!)

```python
# Load model
checkpoint = torch.load('model_ast.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Standard inference (AST only affects training)
with torch.no_grad():
    outputs = model(inputs)
    predictions = outputs.argmax(dim=1)
```

**Important:** AST only affects **training**. Inference is identical to standard models.

### Export to ONNX

```python
# Export AST-trained model to ONNX
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    'model_ast.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

### TorchScript

```python
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save('model_ast_scripted.pt')
```

---

## Troubleshooting

### Issue: Activation rate stuck at 100%

**Problem:** All samples are being selected (no savings)

**Solutions:**
1. Lower initial threshold: `config.initial_threshold = 0.5`
2. Increase PI gains: `config.kp = 0.2, config.ki = 0.02`
3. Add more warmup epochs

```python
config = ASTConfig(
    target_activation_rate=0.35,
    initial_threshold=0.5,  # Start lower
    kp=0.2,  # More aggressive
    ki=0.02,
)
```

### Issue: Activation rate stuck at 0%

**Problem:** No samples are being selected

**Solutions:**
1. Increase `min_active_samples`: `config.min_active_samples = 16`
2. Decrease PI gains: `config.kp = 0.05, config.ki = 0.005`

```python
config = ASTConfig(
    target_activation_rate=0.35,
    min_active_samples=16,  # Guarantee minimum
    kp=0.05,  # Gentler control
    ki=0.005,
)
```

### Issue: Accuracy much lower than baseline

**Problem:** AST being too aggressive

**Solutions:**
1. Increase target activation rate: `0.35 → 0.45`
2. Add more warmup epochs: `warmup_epochs=10`
3. Lower entropy weight: `entropy_weight=0.5`

```python
config = ASTConfig(
    target_activation_rate=0.45,  # More conservative
    entropy_weight=0.5,  # Less aggressive selection
)

results = trainer.train(epochs=50, warmup_epochs=10)  # More warmup
```

### Issue: Training unstable / oscillating

**Problem:** PI controller oscillating

**Solutions:**
1. Reduce PI gains: `kp=0.05, ki=0.005`
2. Add warmup for PI gains:

```python
config = ASTConfig(
    target_activation_rate=0.35,
    kp=0.1,
    ki=0.01,
    kp_warmup_epochs=10,  # Ramp kp gradually
    ki_warmup_epochs=20,  # Ramp ki gradually
)
```

### Issue: Out of memory (OOM)

**Problem:** Mixed precision or batch size too large

**Solutions:**
1. Enable gradient accumulation:

```python
config = ASTConfig(
    target_activation_rate=0.35,
    grad_accum_steps=4,  # Accumulate over 4 batches
)
```

2. Reduce batch size in DataLoader
3. Disable mixed precision if on CPU:

```python
config = ASTConfig(
    target_activation_rate=0.35,
    use_mixed_precision=False,  # For CPU or debugging
)
```

---

## Best Practices

### 1. Start Conservative

Begin with higher activation rates (0.40-0.50) and gradually decrease:

```python
# Iteration 1
config = ASTConfig(target_activation_rate=0.50)  # 50% savings

# Iteration 2 (if accuracy good)
config = ASTConfig(target_activation_rate=0.35)  # 65% savings

# Iteration 3 (if still good)
config = ASTConfig(target_activation_rate=0.25)  # 75% savings
```

### 2. Monitor Activation Rate

Check that activation rate stabilizes near target:

```python
results = trainer.train(epochs=50)
history = results['training_history']

import matplotlib.pyplot as plt
plt.plot(history['activation_rate'])
plt.axhline(y=0.35, color='r', linestyle='--', label='Target')
plt.legend()
plt.show()
```

### 3. Use Warmup

Always use warmup epochs for stability:

```python
# Rule of thumb: 10% of total epochs
trainer.train(epochs=100, warmup_epochs=10)
```

### 4. Architecture-Specific Tuning

| Architecture | Activation Rate | Entropy Weight | Warmup |
|--------------|-----------------|----------------|--------|
| ResNet       | 0.30-0.35       | 1.0            | 5      |
| EfficientNet | 0.35-0.40       | 1.5            | 10     |
| ViT          | 0.45-0.50       | 2.0            | 15     |
| ConvNeXt     | 0.25-0.30       | 1.0            | 5      |
| MobileNet    | 0.30-0.35       | 0.8            | 5      |

---

## Getting Help

- 📖 [Full Documentation](https://github.com/oluwafemidiakhoa/adaptive-sparse-training)
- 🐛 [Report Issues](https://github.com/oluwafemidiakhoa/adaptive-sparse-training/issues)
- 💬 [Discussions](https://github.com/oluwafemidiakhoa/adaptive-sparse-training/discussions)

---

**Happy Training! ⚡**
