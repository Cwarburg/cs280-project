"""
Visualize detected cuts: for each sampled cut show the frame before,
the cut frame, and the frame after.

Usage:
    python viz_cuts.py --frames_root frames --cuts_json cuts.json \
                       --n_sample 30 --out cuts_viz/cut_triplets.png
"""
import argparse
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def load_frames(frames_root: Path, key: str) -> list[Path]:
    return sorted((frames_root / key).glob("*.jpg"))


def plot_triplets(samples: list[tuple], out_path: Path):
    """
    samples: list of (key, cut_idx, frames_list)
    Layout: one row per cut, 3 columns (before / cut / after)
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 2.2))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_labels = ["before", "cut", "after"]

    for row, (key, cut, frames) in enumerate(samples):
        idxs = [max(cut - 1, 0), cut, min(cut + 1, len(frames) - 1)]
        for col, fi in enumerate(idxs):
            img = Image.open(frames[fi]).convert("RGB")
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            title = f"{col_labels[col]}  #{fi}"
            if col == 1:
                axes[row, col].set_title(title, fontsize=6, color="red", fontweight="bold")
            else:
                axes[row, col].set_title(title, fontsize=6)

        short_key = "/".join(key.split("/")[-2:])
        axes[row, 0].set_ylabel(short_key, fontsize=5, rotation=0,
                                labelpad=60, va="center")

    fig.suptitle(f"{n} sampled cuts  —  before / cut / after", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_root", default="frames", type=Path)
    p.add_argument("--cuts_json",   default="cuts.json", type=Path)
    p.add_argument("--n_sample",    default=30, type=int)
    p.add_argument("--seed",        default=0, type=int)
    p.add_argument("--out",         default="cuts_viz/cut_triplets.png", type=Path)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.cuts_json) as f:
        cuts_map = json.load(f)

    all_cuts = [(key, c) for key, cuts in cuts_map.items() for c in cuts]
    print(f"Total cuts available: {len(all_cuts)}")

    rng = random.Random(args.seed)
    chosen = rng.sample(all_cuts, min(args.n_sample, len(all_cuts)))
    chosen.sort()

    frame_cache = {}
    samples = []
    for key, cut in chosen:
        if key not in frame_cache:
            frame_cache[key] = load_frames(args.frames_root, key)
        frames = frame_cache[key]
        if cut < len(frames):
            samples.append((key, cut, frames))

    print(f"Plotting {len(samples)} cuts...")
    plot_triplets(samples, args.out)


if __name__ == "__main__":
    main()
