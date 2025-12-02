"""
Download Imagenette (HF) and export to an ImageFolder layout for quick smoke tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def export_split(ds, out_root: Path, split: str):
    for sample in tqdm(ds, desc=f"Export {split}"):
        label = str(sample["label"])
        cls_dir = out_root / split / label
        cls_dir.mkdir(parents=True, exist_ok=True)
        img = sample["image"].convert("RGB")
        idx = sample["idx"] if "idx" in sample else None
        fname = f"{idx}.jpg" if idx is not None else f"{len(list(cls_dir.glob('*.jpg')))}.jpg"
        img.save(cls_dir / fname, format="JPEG")


def main():
    ap = argparse.ArgumentParser(description="Download Imagenette to ImageFolder layout")
    ap.add_argument("--out-dir", type=str, default="data/imagenette", help="Output directory")
    ap.add_argument("--variant", type=str, default="full", choices=["full", "160", "320"], help="Imagenette size")
    args = ap.parse_args()

    name = "frgfm/imagenette"
    ds = load_dataset(name, args.variant)
    out_root = Path(args.out_dir)
    export_split(ds["train"], out_root, "train")
    export_split(ds["validation"], out_root, "val")
    print(f"Saved Imagenette to {out_root}")


if __name__ == "__main__":
    main()
