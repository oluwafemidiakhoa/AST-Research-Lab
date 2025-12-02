# AST Training Dashboard - HuggingFace Space

Interactive dashboard for training models with Adaptive Sparse Training (AST).

## Features

- 🚀 **Live Training**: Watch your model train in real-time
- 📊 **Energy Tracking**: See energy savings as you train
- 🎯 **Model Card Generation**: Auto-generate HuggingFace model cards
- ⚡ **60-70% Energy Savings**: Train faster with minimal accuracy loss

## Quick Start

### Deploy to HuggingFace Spaces

1. Create new Space at https://huggingface.co/spaces
2. Choose **Gradio** as SDK
3. Upload files from this directory:
   - `app.py`
   - `requirements.txt`
   - `README.md`
4. Space will auto-deploy!

### Run Locally

```bash
cd hf_space
pip install -r requirements.txt
python app.py
```

Then open http://localhost:7860

## Usage

1. **Select Model**: Choose from ResNet18, EfficientNet, MobileNet
2. **Set Activation Rate**: Lower = more energy savings (0.35 recommended)
3. **Choose Epochs**: 30-50 epochs for good results
4. **Start Training**: Click "Start Training" and watch live metrics
5. **Get Model Card**: Copy auto-generated card for HuggingFace Hub

## Example Results

Training ResNet18 on CIFAR-10 with AST (activation_rate=0.35):

- **Accuracy**: 92.1% (vs 92.3% baseline)
- **Energy Savings**: 65%
- **Training Time**: 2.8h (vs 7.2h baseline)

## About AST

Adaptive Sparse Training (AST) automatically selects the most important training samples
per batch, reducing compute by 60-70% while maintaining accuracy.

**How it works:**
1. Computes significance score (loss + entropy) for each sample
2. PI controller dynamically adjusts selection threshold
3. Only backpropagates through "hard" samples
4. Result: Same accuracy, way less compute

## Links

- 📦 [PyPI Package](https://pypi.org/project/adaptive-sparse-training/)
- 🐙 [GitHub](https://github.com/oluwafemidiakhoa/adaptive-sparse-training)
- 📚 [Full Documentation](https://github.com/oluwafemidiakhoa/adaptive-sparse-training#readme)

## Citation

```bibtex
@software{adaptive_sparse_training,
    title={Adaptive Sparse Training},
    author={Idiakhoa, Oluwafemi},
    year={2024},
    url={https://github.com/oluwafemidiakhoa/adaptive-sparse-training}
}
```
