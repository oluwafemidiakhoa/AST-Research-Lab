# Quick Fix for HuggingFace Space Error

## The Problem
The Space is showing: `TypeError: BlockContext.__init__() got an unexpected keyword argument 'theme'`

## The Fix (Done ✅)
Changed line 291 in `app.py` from:
```python
with gr.Blocks(title="AST Training Dashboard", theme=gr.themes.Soft()) as demo:
```

To:
```python
with gr.Blocks(title="AST Training Dashboard") as demo:
```

## How to Update Your HuggingFace Space

### Option 1: Direct File Edit on HuggingFace (Fastest - 1 minute)

1. Go to https://huggingface.co/spaces/mgbam/AST_Dashboard
2. Click "Files" tab
3. Click on `app.py`
4. Click "Edit" (pencil icon)
5. Find line 291 (search for `gr.Blocks(title=`)
6. Remove `, theme=gr.themes.Soft()` from that line
7. Click "Commit changes to main"
8. Wait 30 seconds for rebuild
9. Click "App" tab - should work now! ✅

### Option 2: Upload New File

1. Go to https://huggingface.co/spaces/mgbam/AST_Dashboard
2. Click "Files" tab
3. Click "Upload files"
4. Drag and drop the fixed `app.py` from this directory
5. Commit changes
6. Wait for rebuild

### Option 3: Git Clone and Push (Advanced)

```bash
# Clone your Space
git clone https://huggingface.co/spaces/mgbam/AST_Dashboard
cd AST_Dashboard

# Copy fixed file
cp ../hf_space/app.py .

# Commit and push
git add app.py
git commit -m "Fix: Remove theme parameter for Gradio compatibility"
git push
```

## Verify It Works

1. Go to https://huggingface.co/spaces/mgbam/AST_Dashboard
2. Click "App" tab
3. You should see the interface (no error)
4. Try clicking "Start Training" with defaults
5. Should work!

## Why This Happened

The `theme` parameter in `gr.Blocks()` was added in Gradio 4.0+, but HuggingFace Spaces might be running an older version. Removing it makes it compatible with all versions.

The app still looks good without the explicit theme!
