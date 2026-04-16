"""
Visualize detected cuts: for each sampled cut, show frame[i] and frame[i+1].
Also plots the full histogram-difference signal with cut positions marked.

Usage:
    python viz_cuts.py --frames_root frames/test --cuts_json cuts.json \
                       --key 1 --n_sample 20 --out cuts_viz/
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scene_detect import hist_diff


def plot_diff_signal(frames: list[Path], cuts: list[int], out_path: Path, threshold: float):
    print("Computing diff signal (this may take a minute)...")
    diffs = []
    prev = np.array(Image.open(frames[0]).convert("RGB"))
    for path in frames[1:]:
        curr = np.array(Image.open(path).convert("RGB"))
        diffs.append(hist_diff(prev, curr))
        prev = curr

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.plot(diffs, lw=0.6, color="steelblue", label="hist diff")
    ax.axhline(threshold, color="red", lw=1, linestyle="--", label=f"threshold={threshold}")
    for c in cuts:
        ax.axvline(c, color="orange", lw=0.4, alpha=0.6)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Histogram difference")
    ax.set_title(f"Scene-cut signal  —  {len(cuts)} cuts detected")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved diff signal → {out_path}")


def plot_cut_pairs(frames: list[Path], cuts: list[int], n_sample: int, out_path: Path):
    import random
    rng = random.Random(0)
    sample = rng.sample(cuts, min(n_sample, len(cuts)))
    sample.sort()

    # Layout: 2 cuts per row, each cut takes 2 columns (before + after)
    n_cols = 4
    n_rows = (len(sample) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5))
    axes = np.array(axes).reshape(-1, n_cols)

    for idx, cut in enumerate(sample):
        row = idx // 2
        col_offset = (idx % 2) * 2

        img_before = Image.open(frames[cut]).convert("RGB")
        img_after  = Image.open(frames[cut + 1]).convert("RGB") if cut + 1 < len(frames) else img_before
        diff = hist_diff(np.array(img_before), np.array(img_after))

        axes[row, col_offset].imshow(img_before)
        axes[row, col_offset].set_title(f"frame {cut}", fontsize=7)
        axes[row, col_offset].axis("off")

        axes[row, col_offset + 1].imshow(img_after)
        axes[row, col_offset + 1].set_title(f"frame {cut+1}  Δ={diff:.3f}", fontsize=7)
        axes[row, col_offset + 1].axis("off")

    # Hide unused axes if n_sample is odd
    if len(sample) % 2 == 1:
        for c in range(2, n_cols):
            axes[-1, c].axis("off")

    fig.suptitle(f"Sample of {len(sample)} detected cuts (frame before → after)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved cut pairs → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_root", default="frames/test", type=Path)
    p.add_argument("--cuts_json",   default="cuts.json",   type=Path)
    p.add_argument("--key",         default="1",           help="Key in cuts_json")
    p.add_argument("--threshold",   default=0.35,          type=float)
    p.add_argument("--n_sample",    default=20,            type=int, help="Cuts to show in pair plot")
    p.add_argument("--out",         default="cuts_viz",    type=Path)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.cuts_json) as f:
        cuts_map = json.load(f)

    cuts = cuts_map[args.key]
    half_dir = args.frames_root / args.key
    frames = sorted(half_dir.glob("*.jpg"))
    print(f"Frames: {len(frames)},  Cuts: {len(cuts)}")

    plot_diff_signal(frames, cuts, args.out / "diff_signal.png", args.threshold)
    plot_cut_pairs(frames, cuts, args.n_sample, args.out / "cut_pairs.png")


if __name__ == "__main__":
    main()
