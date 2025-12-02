# 🚀 Deploy AST Dashboard to HuggingFace Spaces

## Quick Deploy (5 minutes)

### Method 1: Direct from GitHub (Recommended)

1. **Create HuggingFace Space**
   - Go to https://huggingface.co/new-space
   - Space name: `ast-training-dashboard`
   - License: `MIT`
   - SDK: **Gradio**
   - Hardware: `CPU basic` (free tier works!)
   - Visibility: `Public`

2. **Connect to GitHub**
   - In Space settings, go to "Files and versions"
   - Click "Import from GitHub"
   - Repository: `oluwafemidiakhoa/Ast-Dashboard`
   - Branch: `main`
   - Click "Import"

3. **Wait for Build**
   - HuggingFace will automatically:
     - Clone your repo
     - Install dependencies from `requirements.txt`
     - Launch `app.py`
   - Build takes ~3-5 minutes

4. **Done!**
   - Your Space will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/ast-training-dashboard`
   - Share the link with the world!

### Method 2: Manual Upload

1. **Create Space** (same as above)

2. **Upload Files**
   - Click "Files" tab
   - Upload these 4 files:
     - `app.py`
     - `requirements.txt`
     - `README.md`
     - `.gitignore`

3. **Commit and Build**
   - Add commit message: "Initial deployment"
   - Space will auto-build

### Method 3: Git Clone (Advanced)

```bash
# Install git-lfs if not already installed
git lfs install

# Clone your HF Space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/ast-training-dashboard
cd ast-training-dashboard

# Copy files from this directory
cp ../hf_space/* .

# Commit and push
git add .
git commit -m "Deploy AST Training Dashboard"
git push
```

## Verifying Deployment

Once deployed, test your Space:

1. **Open Space URL**: `https://huggingface.co/spaces/YOUR_USERNAME/ast-training-dashboard`

2. **Test Training**:
   - Select "resnet18"
   - Keep defaults (activation_rate=0.35, epochs=10)
   - Click "Start Training"
   - Watch live metrics appear

3. **Check Outputs**:
   - ✅ Training progress updates
   - ✅ Live plots appear
   - ✅ Model card generates at end

## Troubleshooting

### "Application startup failed"

**Issue:** Dependencies failed to install

**Fix:** Check `requirements.txt` versions are compatible
```bash
# Locally test installation
pip install -r requirements.txt
python app.py
```

### "Out of memory"

**Issue:** CPU basic tier ran out of RAM

**Solutions:**
1. Reduce `epochs` slider maximum to 50
2. Reduce batch size in `app.py`:
   ```python
   # Line ~120
   train_loader = DataLoader(..., batch_size=64)  # Was 128
   ```
3. Upgrade to CPU upgraded ($0.03/hour)

### "CUDA not available"

**Expected behavior** - CPU tier doesn't have GPU

**If you need GPU:**
- Upgrade to T4 small ($0.60/hour)
- Or keep CPU (works fine for demo with small epochs)

### "Dataset download timeout"

**Issue:** CIFAR-10 download taking too long

**Fix:** Pre-download dataset
```bash
# In app.py, line 108, add:
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

## Customization Ideas

### 1. Add More Datasets

```python
# In app.py, add to _get_dataloaders():
elif dataset == "mnist":
    train_dataset = torchvision.datasets.MNIST(...)
```

### 2. Custom Branding

```python
# In create_demo(), change theme:
with gr.Blocks(theme=gr.themes.Base()) as demo:
```

Available themes:
- `gr.themes.Soft()` (current)
- `gr.themes.Base()`
- `gr.themes.Glass()`
- `gr.themes.Monochrome()`

### 3. Add Model Saving

```python
# After training completes:
torch.save(model.state_dict(), "model.pth")
return status, plot, model_card, gr.File("model.pth")
```

### 4. Email Notifications

```python
# Install: pip install sendgrid
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# After training:
message = Mail(
    from_email='you@example.com',
    to_emails='you@example.com',
    subject=f'AST Training Complete: {model_name}',
    html_content=f'<strong>Accuracy: {best_acc:.2%}</strong>')
sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
sg.send(message)
```

## Promoting Your Space

### 1. Share on Social Media

**Twitter:**
```
🚀 Just launched AST Training Dashboard on @huggingface!

Train ANY model with 60-70% less energy ⚡

Try it: https://huggingface.co/spaces/YOUR_USERNAME/ast-training-dashboard

Features:
✅ Live training visualization
✅ Energy savings tracking
✅ Auto model card generation

#MachineLearning #SustainableAI
```

**LinkedIn:**
```
Excited to share my new HuggingFace Space: AST Training Dashboard!

This interactive tool demonstrates Adaptive Sparse Training, achieving
60-70% energy savings while maintaining model accuracy.

Key features:
• Live training metrics
• Real-time energy tracking
• Automatic HuggingFace model card generation

Try it yourself: [Space URL]

Built with Gradio and adaptive-sparse-training package.

#AI #MachineLearning #Sustainability #GreenAI
```

### 2. Add to HuggingFace Profile

- Pin the Space to your profile
- Add to "Pinned" section
- Add tags: `machine-learning`, `sustainability`, `energy-efficient`

### 3. Submit to HuggingFace Newsletter

Email: spaces@huggingface.co
```
Subject: Feature Request: AST Training Dashboard

Hi HuggingFace Team,

I built an interactive Space that demonstrates energy-efficient
model training using Adaptive Sparse Training (AST).

Space: [Your URL]

It achieves 60-70% energy savings on standard architectures,
which aligns with HuggingFace's sustainability mission.

Would you consider featuring it in your newsletter or blog?

Thanks,
Oluwafemi
```

### 4. Demo Video

Record a 2-minute screencast:
1. Show Space interface
2. Start training ResNet18
3. Point out live energy savings
4. Show generated model card
5. Explain the impact

Upload to YouTube, link in Space README.

## Monitoring

### View Analytics

HuggingFace provides basic analytics:
- Go to Space settings
- Click "Analytics"
- See: visitors, unique users, usage over time

### Track Feedback

Enable Discussions:
- Space settings → "Discussions" → Enable
- Users can ask questions, report bugs
- Engage to build community

## Upgrading

### Enable GPU (Optional)

For faster demo training:
1. Space settings → "Hardware"
2. Select "T4 small" (~$0.60/hour)
3. Update `app.py`:
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   trainer = AdaptiveSparseTrainer(model, train_loader, val_loader, config, device=device)
   ```

**Note:** GPU costs money. CPU is free and sufficient for demo.

### Add Secrets

For MCP integration or HF Hub uploads:
1. Space settings → "Variables and secrets"
2. Add secrets:
   - `HF_TOKEN`: Your HuggingFace token
   - `KAGGLE_USERNAME`: Kaggle username
   - `KAGGLE_KEY`: Kaggle API key

3. Access in code:
   ```python
   import os
   hf_token = os.environ.get("HF_TOKEN")
   ```

## Next Steps

After deployment:
1. ✅ Test all features work
2. ✅ Share on social media
3. ✅ Add to your resume/portfolio
4. ✅ Submit to HuggingFace for featuring
5. ✅ Create tutorial blog post linking to Space
6. ✅ Use in Kaggle competition tutorials

---

## Quick Reference

**Space URL:** `https://huggingface.co/spaces/YOUR_USERNAME/ast-training-dashboard`
**GitHub Repo:** https://github.com/oluwafemidiakhoa/Ast-Dashboard
**Package:** https://pypi.org/project/adaptive-sparse-training/

**Need help?** Open an issue on GitHub!
