"""
Inference + visualization for the best pairwise ordering model (resnet50_r1).

Outputs per sample (saved to viz_inference/):
  - ordering_<seed>.png  : 2-row figure, GT order (top) vs predicted order (bottom)
                           with GradCAM overlays and colour-coded borders
  - score_matrix_<seed>.png : n×n P(i before j) heatmap + predicted/true rank bars
  - gradcam_pairs_<seed>.png: consecutive predicted pairs, each with A/B + saliency

Usage:
    python infer_viz.py [--n_samples 5] [--seq_len 12] [--out viz_inference]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from PIL import Image
from scipy.stats import kendalltau

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from model import PairwiseOrderingModel
from dataset import get_transform
from gradcam import GradCAM


# ── helpers ──────────────────────────────────────────────────────────────────

def denorm(tensor: torch.Tensor) -> np.ndarray:
    """(C,H,W) normalised tensor → (H,W,3) uint8."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = tensor.detach().permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * std + mean, 0, 1)
    return (img * 255).astype(np.uint8)


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
    return np.clip(rgb.astype(float) * (1 - alpha) + heat.astype(float) * alpha, 0, 255).astype(np.uint8)


def border_color(error: int) -> str:
    return "green" if error == 0 else ("orange" if error <= 2 else "red")


def load_sequence(frames_dir: Path, n_frames: int, rng: random.Random):
    """Pick a random contiguous clip of n_frames from frames_dir."""
    all_frames = sorted(frames_dir.glob("*.jpg"))
    if len(all_frames) < n_frames:
        return None
    start = rng.randint(0, len(all_frames) - n_frames)
    return all_frames[start : start + n_frames]


# ── plot 1: ordering figure ───────────────────────────────────────────────────

def plot_ordering(
    frames_tensor: torch.Tensor,
    shuffled_tensor: torch.Tensor,
    shuffled_order: list[int],
    predicted_rank: list[int],
    cams: list[np.ndarray],
    tau: float,
    save_path: Path,
):
    n = frames_tensor.size(0)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.2, 5.2))
    fig.suptitle(
        f"Row 1: ground-truth order  |  Row 2: model-predicted order + GradCAM"
        f"   (Kendall τ = {tau:.3f})",
        fontsize=9,
    )

    for i in range(n):
        # Ground truth row
        axes[0, i].imshow(denorm(frames_tensor[i]))
        axes[0, i].set_title(f"GT {i+1}", fontsize=7)
        axes[0, i].axis("off")

        # Predicted row
        sh_idx = predicted_rank[i]
        true_pos = shuffled_order[sh_idx]
        error = abs(true_pos - i)
        col = border_color(error)

        rgb = denorm(shuffled_tensor[sh_idx])
        axes[1, i].imshow(overlay_cam(rgb, cams[i]))
        axes[1, i].set_title(f"Pred {i+1}\n(true {true_pos+1})", fontsize=7, color=col)
        for side in ["top", "bottom", "left", "right"]:
            axes[1, i].spines[side].set_visible(True)
            axes[1, i].spines[side].set_edgecolor(col)
            axes[1, i].spines[side].set_linewidth(3)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ── plot 2: score matrix ──────────────────────────────────────────────────────

def plot_score_matrix(
    score_mat: np.ndarray,
    predicted_rank: list[int],
    shuffled_order: list[int],
    save_path: Path,
):
    n = score_mat.shape[0]
    true_positions = [shuffled_order[r] for r in predicted_rank]

    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], wspace=0.35)

    # Score matrix
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(score_mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax0.set_title("P(row before col)  —  score matrix", fontsize=9)
    ax0.set_xlabel("Frame j (shuffled)", fontsize=8)
    ax0.set_ylabel("Frame i (shuffled)", fontsize=8)
    ax0.set_xticks(range(n)); ax0.set_xticklabels(range(1, n+1), fontsize=6)
    ax0.set_yticks(range(n)); ax0.set_yticklabels(range(1, n+1), fontsize=6)
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    # Predicted vs true rank bars
    ax1 = fig.add_subplot(gs[1])
    xs = np.arange(n)
    ax1.barh(xs, [p + 1 for p in true_positions], height=0.4, align="center",
             color="steelblue", label="True position")
    ax1.barh(xs + 0.4, range(1, n+1), height=0.4, align="center",
             color="tomato", alpha=0.7, label="Predicted position")
    ax1.set_yticks(xs + 0.2)
    ax1.set_yticklabels([f"Pred {i+1}" for i in range(n)], fontsize=7)
    ax1.set_xlabel("Frame number (1 = earliest)", fontsize=8)
    ax1.set_title("Predicted vs true position", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.invert_yaxis()

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ── plot 3: gradcam pairs ─────────────────────────────────────────────────────

def plot_gradcam_pairs(
    pred_tensor: torch.Tensor,
    cams_a: list[np.ndarray],
    cams_b: list[np.ndarray],
    n_show: int,
    save_path: Path,
):
    """Show first n_show consecutive predicted pairs with A / cam_A / B / cam_B."""
    n_show = min(n_show, pred_tensor.size(0) - 1)
    fig, axes = plt.subplots(n_show, 4, figsize=(10, n_show * 2.4))
    if n_show == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Consecutive predicted pairs: frame | GradCAM  (A → B)", fontsize=9)

    col_titles = ["Frame A", "GradCAM A", "Frame B", "GradCAM B"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=8)

    for row in range(n_show):
        rgb_a = denorm(pred_tensor[row])
        rgb_b = denorm(pred_tensor[row + 1])
        imgs = [rgb_a, overlay_cam(rgb_a, cams_a[row]),
                rgb_b, overlay_cam(rgb_b, cams_b[row])]
        for col, img in enumerate(imgs):
            axes[row, col].imshow(img)
            axes[row, col].set_ylabel(f"Pair {row+1}→{row+2}", fontsize=7)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt = Path("checkpoints/resnet50_r1/best.pt")
    model = PairwiseOrderingModel(
        encoder_name="resnet50",
        pretrained=False,
        embed_dim=256,
        hidden_dim=512,
        dropout=0.3,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"Loaded {ckpt}")

    transform = get_transform(train=False)

    with open("cuts.json") as f:
        cuts_map = json.load(f)
    keys = list(cuts_map.keys())

    taus = []
    for sample_idx in range(args.n_samples):
        seed = args.seed + sample_idx
        rng = random.Random(seed)

        # Pick a random half with enough frames
        for _ in range(50):
            key = rng.choice(keys)
            frames_dir = Path("frames") / key
            frame_paths = load_sequence(frames_dir, args.seq_len, rng)
            if frame_paths is not None:
                break
        else:
            print(f"  Sample {sample_idx}: could not find a long enough sequence, skipping")
            continue

        print(f"\nSample {sample_idx+1}  ({key})")

        # Load and shuffle
        imgs = [transform(Image.open(p).convert("RGB")) for p in frame_paths]
        frames_tensor = torch.stack(imgs).to(device)
        n = frames_tensor.size(0)

        shuffled_order = list(range(n))
        rng.shuffle(shuffled_order)
        shuffled_tensor = frames_tensor[shuffled_order]

        # Rank
        with torch.no_grad():
            score_mat = model.score_matrix(shuffled_tensor).cpu().numpy()
            predicted_rank = model.rank_frames(shuffled_tensor).cpu().tolist()

        # Kendall's tau
        true_positions = [shuffled_order[r] for r in predicted_rank]
        tau, _ = kendalltau(range(n), true_positions)
        taus.append(tau)
        print(f"  Kendall τ = {tau:.4f}")

        # Predicted-order tensor (frames in predicted order)
        pred_tensor = shuffled_tensor[predicted_rank]

        # GradCAM for consecutive pairs in predicted order
        gradcam = GradCAM(model, target_layer="encoder.layer4")
        cams_order = []      # cam_a for ordering figure (one per predicted position)
        cams_pairs_a = []    # cam_a for pairs figure
        cams_pairs_b = []    # cam_b for pairs figure

        for k in range(n):
            ta = pred_tensor[k].unsqueeze(0)
            tb = pred_tensor[min(k + 1, n - 1)].unsqueeze(0)
            cam_a, cam_b = gradcam(ta, tb, target="a")
            cams_order.append(cam_a)
            if k < n - 1:
                cams_pairs_a.append(cam_a)
                _, cam_b2 = gradcam(ta, tb, target="b")
                cams_pairs_b.append(cam_b2)

        gradcam.remove_hooks()

        # Save figures
        tag = f"{sample_idx+1:02d}_seed{seed}"
        plot_ordering(frames_tensor, shuffled_tensor, shuffled_order,
                      predicted_rank, cams_order, tau,
                      out_dir / f"ordering_{tag}.png")
        plot_score_matrix(score_mat, predicted_rank, shuffled_order,
                          out_dir / f"score_matrix_{tag}.png")
        plot_gradcam_pairs(pred_tensor, cams_pairs_a, cams_pairs_b,
                           n_show=min(6, n - 1),
                           save_path=out_dir / f"gradcam_pairs_{tag}.png")

    if taus:
        print(f"\nMean Kendall τ over {len(taus)} samples: {np.mean(taus):.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--seq_len",   type=int, default=12)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--out",       default="viz_inference")
    main(p.parse_args())
