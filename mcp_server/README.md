# AST Training Hub - MCP Server

**Unified training platform connecting HuggingFace, Kaggle, and local environments**

## Quick Start

### 1. Install Dependencies

```bash
pip install mcp kaggle huggingface_hub python-dotenv adaptive-sparse-training
```

### 2. Configure Credentials

Copy `.env.example` to `.env` and fill in your credentials:

```bash
# Kaggle API
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# HuggingFace
HF_TOKEN=hf_your_token_here
```

### 3. Run the MCP Server

```bash
python mcp_server/ast_training_hub.py
```

## Using with Claude Code

### Configure Claude Code MCP

Add to your Claude Code MCP config (`~/.claude/mcp_config.json`):

```json
{
  "mcpServers": {
    "ast-training-hub": {
      "command": "python",
      "args": ["C:/Users/adminidiakhoa/Demo/AST_local/mcp_server/ast_training_hub.py"],
      "env": {
        "PYTHONPATH": "C:/Users/adminidiakhoa/Demo/AST_local"
      }
    }
  }
}
```

### Example Usage in Claude Code

```
User: "Train ResNet50 on ImageNet-100 using AST on Kaggle,
       then upload to HuggingFace when done"

Claude Code (using MCP):
  1. start_training(
       platform="kaggle",
       dataset="imagenet-100",
       model="resnet50",
       ast_config={"target_activation_rate": 0.35},
       auto_upload_hf=True
     )
     → Returns job_id: "kaggle_1234567"

  2. monitor_training(job_id="kaggle_1234567")
     → Shows live progress every 30s

  3. When complete:
     sync_model(
       from_platform="kaggle",
       to_platform="huggingface",
       hf_repo="your-username/resnet50-ast-imagenet100"
     )

  4. create_model_card(run_id="kaggle_1234567")
     → Generates model card with sustainability metrics
```

## Available MCP Tools

### `start_training`
Launch AST training on any platform

**Parameters:**
- `platform`: "kaggle" | "huggingface" | "local"
- `dataset`: Dataset name or path
- `model`: Model architecture (e.g., "resnet50", "efficientnet_b0")
- `ast_config`: AST configuration
  - `target_activation_rate`: Target % of samples to train on (0.35 = 65% savings)
  - `entropy_weight`: Weight for entropy in significance score
- `auto_upload_hf`: Auto-upload to HuggingFace when done (default: True)

**Returns:**
```json
{
  "job_id": "kaggle_1234567",
  "platform": "kaggle",
  "status": "started",
  "kernel_url": "https://www.kaggle.com/code/...",
  "message": "Training started on Kaggle..."
}
```

### `monitor_training`
Get live training metrics

**Parameters:**
- `job_id`: Job ID from start_training()

**Returns:**
```json
{
  "job_id": "kaggle_1234567",
  "status": "running",
  "epoch": 15,
  "accuracy": 0.874,
  "activation_rate": 0.382,
  "energy_savings": 0.618,
  "eta_minutes": 45
}
```

### `compare_runs`
Benchmark AST vs baseline

**Parameters:**
- `ast_run_id`: AST training job ID
- `baseline_run_id`: Baseline job ID (optional - will estimate if not provided)

**Returns:**
```json
{
  "ast": {...},
  "baseline": {...},
  "comparison": {
    "accuracy_diff": -0.002,
    "energy_savings": 0.61,
    "time_savings_hours": 3.4,
    "cost_savings_usd": 2.80
  }
}
```

### `sync_model`
Sync model between platforms

**Parameters:**
- `from_platform`: Source platform
- `to_platform`: Destination platform
- `model_path`: Path to model checkpoint
- `hf_repo`: HuggingFace repo name (e.g., "username/model-name")
- `include_ast_metadata`: Include AST config (default: True)

**Returns:**
```json
{
  "status": "success",
  "repo_url": "https://huggingface.co/username/model-name",
  "message": "Model uploaded to HuggingFace"
}
```

### `create_model_card`
Generate HuggingFace model card with sustainability metrics

**Parameters:**
- `run_id`: Training job ID
- `template`: "standard" | "competition" | "research"

**Returns:**
```json
{
  "markdown": "...",
  "badges": ["adaptive-sparse-training", "energy-efficient"],
  "metrics": {...}
}
```

## Example Workflows

### Workflow 1: Kaggle Competition

```python
# In Claude Code chat:

"I want to compete in the plant pathology competition on Kaggle.
 Train an EfficientNetB3 model with AST to save GPU hours."

# Claude Code will:
1. start_training(
     platform="kaggle",
     dataset="plant-pathology-2024",
     model="efficientnet_b3",
     ast_config={"target_activation_rate": 0.40}
   )

2. Poll monitor_training() every minute

3. When accuracy plateaus, sync_model() to HuggingFace

4. Generate model card with energy savings
```

### Workflow 2: Multi-Model Comparison

```python
"Train ResNet50, EfficientNetB0, and ConvNeXt-Base on CIFAR-100
 with AST, then compare their efficiency"

# Claude Code launches 3 parallel jobs, then:

compare_runs(
  ast_run_id="local_job1",
  baseline_run_id=None  # Estimates baseline
)

# Generates table:
| Model          | Accuracy | Energy (kWh) | Savings |
|----------------|----------|--------------|---------|
| ResNet50       | 92.1%    | 1.2          | 65%     |
| EfficientNetB0 | 93.4%    | 0.8          | 68%     |
| ConvNeXt-Base  | 94.2%    | 1.5          | 62%     |
```

### Workflow 3: Research Publication

```python
"Run ablation study: AST with activation rates 0.2, 0.3, 0.4, 0.5
 on ImageNet-100, then generate LaTeX table"

# Claude Code:
1. Launches 4 parallel local jobs
2. Monitors all jobs
3. Compares runs
4. Generates LaTeX:

\\begin{table}
\\caption{AST Activation Rate Ablation}
\\begin{tabular}{ccc}
Rate & Accuracy & Energy Savings \\\\
0.20 & 91.2\\% & 80\\% \\\\
0.30 & 91.8\\% & 70\\% \\\\
0.40 & 92.1\\% & 60\\% \\\\
0.50 & 92.3\\% & 50\\% \\\\
\\end{tabular}
\\end{table}
```

## Architecture Diagram

```
┌────────────────────────────────────────────┐
│         Claude Code (User Interface)       │
└────────────────────────────────────────────┘
                    │
                    │ MCP Protocol
                    ▼
┌────────────────────────────────────────────┐
│       AST Training Hub (MCP Server)        │
│                                            │
│  Tools:                                    │
│  • start_training()                        │
│  • monitor_training()                      │
│  • compare_runs()                          │
│  • sync_model()                            │
│  • create_model_card()                     │
└────────────────────────────────────────────┘
         │              │              │
         │              │              │
         ▼              ▼              ▼
   ┌─────────┐   ┌───────────┐   ┌─────────┐
   │ Kaggle  │   │HuggingFace│   │  Local  │
   │   API   │   │    Hub    │   │Training │
   └─────────┘   └───────────┘   └─────────┘
```

## Benefits

### For Researchers
- **One command** to train → benchmark → publish
- **Automatic tracking** of all experiments
- **Reproducibility**: Full config stored with results

### For Kaggle Competitors
- **Save GPU hours**: 60-70% reduction in compute
- **Faster iteration**: Train more models in less time
- **Live monitoring**: No need to refresh Kaggle constantly

### For HuggingFace Community
- **Sustainability badges**: "Trained with 61% less energy"
- **Energy leaderboard**: Rank models by efficiency
- **Easy reproduction**: AST config included in model cards

## Roadmap

- [ ] HuggingFace Spaces integration (train directly in Spaces)
- [ ] Automatic hyperparameter tuning (Optuna integration)
- [ ] Multi-GPU distributed training support
- [ ] Cost estimator for different cloud providers
- [ ] Energy leaderboard API
- [ ] Integration with Weights & Biases for tracking

## Troubleshooting

### "Kaggle API not authenticated"
- Check `.env` has `KAGGLE_USERNAME` and `KAGGLE_KEY`
- Verify Kaggle API is enabled in your account settings

### "HuggingFace token invalid"
- Generate new token at https://huggingface.co/settings/tokens
- Ensure token has `write` permissions
- Update `HF_TOKEN` in `.env`

### "Job not found"
- MCP server might have restarted, losing job state
- For production, add Redis/database for persistent job tracking

## Contributing

Contributions welcome! Areas for improvement:
- Add more platform integrations (Google Colab, AWS SageMaker)
- Improve monitoring with WebSockets for real-time updates
- Add automatic dataset preprocessing
- Build HuggingFace Spaces UI for non-technical users

## License

MIT License - see LICENSE file
