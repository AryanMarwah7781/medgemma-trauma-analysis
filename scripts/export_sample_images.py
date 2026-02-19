"""
Export Sample CT Images for Demo Testing
=========================================
Pulls real abdominal CT slices from the RSNA 2023 dataset (the same one
used for LoRA fine-tuning) and saves them as PNG files ready for upload
to the Flask web UI.

Usage in Colab:
    python scripts/export_sample_images.py \
        --out_dir /content/sample_ct \
        --n 10 \
        --hf_token $HF_TOKEN

Output:
    /content/sample_ct/
        slice_00_NO_INJURY.png
        slice_01_EXTRAVASATION.png
        slice_02_LIVER_HIGH.png
        ...   (one representative per injury category)
        manifest.txt   (label for each file)
"""

import argparse
import gzip
import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CT_WINDOW_CENTER = 50
CT_WINDOW_WIDTH  = 400


def load_nifti_slice(path: str, num_slices: int = 3) -> Image.Image:
    """Load a NIfTI volume and return a 2.5D RGB PIL image (3 axial slices stacked)."""
    import nibabel as nib, tempfile

    if not path.endswith((".nii", ".nii.gz")):
        with tempfile.TemporaryDirectory() as tmp:
            link = os.path.join(tmp, "ct.nii.gz")
            os.symlink(os.path.abspath(path), link)
            vol = nib.load(link).get_fdata()
    else:
        vol = nib.load(path).get_fdata()

    while vol.ndim > 3:
        vol = vol[..., 0]
    if vol.ndim == 2:
        vol = vol[:, :, np.newaxis]

    depth = vol.shape[2]
    z0 = int(depth * 0.2)
    z1 = max(z0 + 1, int(depth * 0.8))
    idxs = np.linspace(z0, z1 - 1, num_slices, dtype=int)

    lo = CT_WINDOW_CENTER - CT_WINDOW_WIDTH / 2.0
    hi = CT_WINDOW_CENTER + CT_WINDOW_WIDTH / 2.0

    channels = []
    for z in idxs:
        sl = np.clip(vol[:, :, z].astype(np.float32), lo, hi)
        sl = ((sl - lo) / (hi - lo) * 255).astype(np.uint8)
        channels.append(sl)
    while len(channels) < 3:
        channels.append(channels[-1])

    return Image.fromarray(np.stack(channels[:3], axis=-1))


def label_for(item: dict) -> str:
    """Return a short label string describing the injury category."""
    if not item.get("any_injury", False):
        return "NO_INJURY"
    parts = []
    if item.get("extravasation", 0) == 1:
        parts.append("EXTRAV")
    grades = {"liver": item.get("liver", 0), "spleen": item.get("spleen", 0), "kidney": item.get("kidney", 0)}
    for organ, g in grades.items():
        if g == 2:
            parts.append(f"{organ.upper()}_HIGH")
        elif g == 1:
            parts.append(f"{organ.upper()}_LOW")
    if item.get("bowel", 0) == 1:
        parts.append("BOWEL")
    return "_".join(parts) if parts else "INJURY_UNSPEC"


def main():
    parser = argparse.ArgumentParser(description="Export sample CT PNGs from RSNA dataset")
    parser.add_argument("--out_dir", default="/content/sample_ct", help="Output directory")
    parser.add_argument("--n", type=int, default=10, help="Number of images to export")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--num_slices", type=int, default=3, help="Axial slices to stack per image")
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading RSNA dataset...")
    from datasets import load_dataset
    dataset = load_dataset(
        "jherng/rsna-2023-abdominal-trauma-detection",
        split="train",
        token=token,
        trust_remote_code=True,
    )
    print(f"  {len(dataset)} total examples available.")

    saved = []
    seen_labels = set()
    manifest_lines = []

    for i, item in enumerate(dataset):
        if len(saved) >= args.n:
            break

        img_path = item.get("img_path")
        if not img_path or not os.path.exists(img_path):
            continue

        label = label_for(item)

        # Try to get one of each category first, then fill remaining slots
        dedup_key = label.split("_")[0]  # broad category
        if len(saved) < 6 and dedup_key in seen_labels:
            continue

        try:
            img = load_nifti_slice(img_path, num_slices=args.num_slices)
            fname = f"slice_{len(saved):02d}_{label}.png"
            fpath = out_dir / fname
            img.save(str(fpath))
            saved.append(fname)
            seen_labels.add(dedup_key)

            manifest_lines.append(f"{fname}  |  {label}")
            print(f"  [{len(saved):2d}/{args.n}] {fname}  ({img.size[0]}x{img.size[1]})")

        except Exception as e:
            print(f"  Skipping example {i}: {e}")
            continue

    # Write manifest
    manifest_path = out_dir / "manifest.txt"
    with open(manifest_path, "w") as f:
        f.write("CT Sample Images — RSNA 2023 Abdominal Trauma\n")
        f.write("=" * 50 + "\n\n")
        for line in manifest_lines:
            f.write(line + "\n")
    print(f"\n  Manifest: {manifest_path}")

    print(f"\nDone. {len(saved)} images saved to: {out_dir}")
    print("\nTo test in the web UI:")
    print(f"  1. Open the ngrok URL in your browser")
    print(f"  2. Upload any of these files:")
    for fname in saved:
        print(f"       {out_dir / fname}")
    print(f"\n  Or upload multiple at once to test multi-slice analysis!")


if __name__ == "__main__":
    main()
