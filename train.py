"""
Training script for the pairwise temporal ordering model.

Usage:
    python train.py \
        --frames_root frames \
        --cuts_json cuts.json \
        --encoder resnet50 \
        --epochs 20 \
        --batch_size 64 \
        --lr 1e-4 \
        --out checkpoints/

Metrics logged each epoch:
  - train/val loss
  - pairwise accuracy
  - Kendall's tau (on val set, sampled sequences of n=30 frames)

Wandb:
  - scalar metrics every epoch
  - sorted-frames panel + GradCAM panel every 5 epochs
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from scipy.stats import kendalltau

import wandb

from dataset import CutWindowDataset, get_transform
from model import PairwiseOrderingModel
from gradcam import GradCAM


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pairwise_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (logits > 0).float()
    return (preds == labels).float().mean().item()


def evaluate_kendall_tau(
    model: PairwiseOrderingModel,
    frames_root: Path,
    cuts_json: Path,
    n_seq: int = 50,
    seq_len: int = 30,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Sample random sequences of seq_len frames and measure Kendall's tau."""
    from PIL import Image

    with open(cuts_json) as f:
        cuts_map = json.load(f)

    transform = get_transform(train=False)
    model.eval()
    taus = []

    keys = list(cuts_map.keys())
    rng = random.Random(0)

    for _ in range(n_seq):
        key = rng.choice(keys)
        half_dir = frames_root / key
        all_frames = sorted(half_dir.glob("*.jpg"))
        if len(all_frames) < seq_len + 1:
            continue

        start = rng.randint(0, len(all_frames) - seq_len)
        indices = list(range(start, start + seq_len))
        true_order = list(range(seq_len))

        imgs = []
        for idx in indices:
            img = Image.open(all_frames[idx]).convert("RGB")
            imgs.append(transform(img))
        frames_tensor = torch.stack(imgs).to(device)

        shuffled_order = list(range(seq_len))
        rng.shuffle(shuffled_order)
        shuffled_tensor = frames_tensor[shuffled_order]

        with torch.no_grad():
            predicted_rank = model.rank_frames(shuffled_tensor).cpu().tolist()

        rank_of_shuffled = [0] * seq_len
        for pos, sh_idx in enumerate(predicted_rank):
            rank_of_shuffled[sh_idx] = pos

        inv_shuffle = [0] * seq_len
        for sh_idx, true_idx in enumerate(shuffled_order):
            inv_shuffle[true_idx] = sh_idx

        predicted_positions = [rank_of_shuffled[inv_shuffle[t]] for t in true_order]
        tau, _ = kendalltau(true_order, predicted_positions)
        if not np.isnan(tau):
            taus.append(tau)

    return float(np.mean(taus)) if taus else 0.0


# ---------------------------------------------------------------------------
# Wandb visualizations
# ---------------------------------------------------------------------------

def _to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize a (C,H,W) tensor to a (H,W,3) uint8 array."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = tensor.detach().permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * std + mean, 0, 1)
    return (img * 255).astype(np.uint8)


def log_ordering_with_gradcam(
    model: PairwiseOrderingModel,
    frames_root: Path,
    cuts_json: Path,
    device: torch.device,
    n_frames: int = 10,
    target_layer: str = "encoder.layer4",
    seed: int = 1,
):
    """
    Sample a contiguous sequence of n_frames, shuffle, let the model sort it,
    and log a single two-row figure to wandb:

      Row 1 (GT):   frames in true temporal order, plain.
      Row 2 (Pred): frames in model-predicted order, each with GradCAM overlaid.
                    GradCAM for frame at predicted position k is computed w.r.t.
                    the pair (frame_k, frame_{k+1}), showing what the model attends
                    to when deciding frame_k precedes frame_{k+1}.
                    Title shows predicted position and true position; border is
                    green (exact), orange (off by 1-2), or red (off by 3+).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from PIL import Image

    with open(cuts_json) as f:
        cuts_map = json.load(f)

    transform = get_transform(train=False)
    rng = random.Random(seed)
    key = rng.choice(list(cuts_map.keys()))
    half_dir = frames_root / key
    all_frames = sorted(half_dir.glob("*.jpg"))

    if len(all_frames) < n_frames:
        return

    start = rng.randint(0, len(all_frames) - n_frames)
    imgs = [transform(Image.open(all_frames[start + i]).convert("RGB")) for i in range(n_frames)]
    frames_tensor = torch.stack(imgs).to(device)

    shuffled_order = list(range(n_frames))
    rng.shuffle(shuffled_order)
    shuffled_tensor = frames_tensor[shuffled_order]

    model.eval()
    with torch.no_grad():
        predicted_rank = model.rank_frames(shuffled_tensor).cpu().tolist()
    # predicted_rank[k] = index in shuffled_tensor placed at predicted position k

    # Build ordered tensor: frames in predicted order
    pred_tensor = shuffled_tensor[predicted_rank]  # (n, C, H, W)

    # Compute GradCAM for each consecutive pair in predicted order
    gradcam = GradCAM(model, target_layer=target_layer)
    cams = []
    for k in range(n_frames):
        ta = pred_tensor[k].unsqueeze(0)
        tb = pred_tensor[min(k + 1, n_frames - 1)].unsqueeze(0)
        cam_a, _ = gradcam(ta, tb, target="a")
        cams.append(cam_a)
    gradcam.remove_hooks()

    def overlay(rgb, cam):
        heat = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
        return (rgb * 0.55 + heat * 0.45).astype(np.uint8)

    fig, axes = plt.subplots(2, n_frames, figsize=(n_frames * 2.2, 5))
    fig.suptitle("Row 1: ground truth order  |  Row 2: predicted order + GradCAM", fontsize=9)

    for i in range(n_frames):
        # Top row: ground truth
        axes[0, i].imshow(_to_rgb(frames_tensor[i]))
        axes[0, i].set_title(f"GT {i+1}", fontsize=7)
        axes[0, i].axis("off")

        # Bottom row: predicted order with GradCAM overlay
        sh_idx = predicted_rank[i]
        true_pos = shuffled_order[sh_idx]
        error = abs(true_pos - i)
        color = "green" if error == 0 else ("orange" if error <= 2 else "red")

        rgb = _to_rgb(shuffled_tensor[sh_idx])
        axes[1, i].imshow(overlay(rgb, cams[i]))
        axes[1, i].set_title(f"Pred {i+1}\n(true {true_pos+1})", fontsize=7, color=color)
        for side in ["top", "bottom", "left", "right"]:
            axes[1, i].spines[side].set_visible(True)
            axes[1, i].spines[side].set_edgecolor(color)
            axes[1, i].spines[side].set_linewidth(3)
        axes[1, i].axis("off")

    plt.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    plt.close(fig)

    wandb.log({"viz/ordering_gradcam": wandb.Image(buf)})


# ---------------------------------------------------------------------------
# Data sanity-check visualisation
# ---------------------------------------------------------------------------

def _log_example_pairs(dataset: "PairDataset", n: int = 8):
    """Log a wandb table of n example pairs (frame A | frame B | label)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rng = random.Random(99)
    indices = rng.sample(range(len(dataset)), min(n, len(dataset)))

    fig, axes = plt.subplots(n, 2, figsize=(6, n * 1.6))
    fig.suptitle("Example training pairs  (left=A, right=B, title=label)", fontsize=8)

    for row, idx in enumerate(indices):
        img_a, img_b, label = dataset[idx]
        lbl = "A before B" if label.item() == 1 else "B before A"
        for col, (img, title) in enumerate([(img_a, f"A  [{lbl}]"), (img_b, "B")]):
            axes[row, col].imshow(_to_rgb(img))
            axes[row, col].set_title(title, fontsize=6)
            axes[row, col].axis("off")

    plt.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    plt.close(fig)
    wandb.log({"data/example_pairs": wandb.Image(buf)}, step=0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config=vars(args),
    )

    # Dataset
    print("Building dataset...")
    full_ds = CutWindowDataset(
        frames_root=args.frames_root,
        cuts_json=args.cuts_json,
        window_size=args.window_size,
        n_windows_per_cut=args.n_windows_per_cut,
        train=True,
        seed=args.seed,
    )

    val_size  = int(len(full_ds) * args.val_fraction)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_ds.dataset.transform = get_transform(train=False)

    labels  = [lbl for _, _, lbl in full_ds.pairs]
    n_pos   = sum(1 for l in labels if l == 1)

    dataset_stats = {
        "dataset/total_pairs":   len(full_ds),
        "dataset/train_pairs":   train_size,
        "dataset/val_pairs":     val_size,
        "dataset/within_cut":    full_ds.n_within,
        "dataset/cross_cut":     full_ds.n_cross,
        "dataset/label_balance": n_pos / max(len(labels), 1),
    }
    wandb.log(dataset_stats, step=0)
    wandb.summary.update(dataset_stats)

    print(f"  Total pairs : {len(full_ds):,}  (train {train_size:,} / val {val_size:,})")
    print(f"  Within-cut  : {full_ds.n_within:,}  |  Cross-cut: {full_ds.n_cross:,}")
    print(f"  Label balance: {n_pos/len(labels):.1%} positive")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Log a grid of example pairs so we can sanity-check the data
    _log_example_pairs(full_ds, n=8)

    # Model
    model = PairwiseOrderingModel(
        encoder_name=args.encoder,
        pretrained=True,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    encoder_params = list(model.encoder.parameters()) + list(model.projector.parameters())
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr * 0.1},
        {"params": head_params,    "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    criterion = nn.BCEWithLogitsLoss()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = []
    best_val_acc = 0.0

    # GradCAM target layer — only supported for ResNet-style encoders
    if "resnet" in args.encoder:
        gradcam_layer = "encoder.layer4"
    elif "vit" in args.encoder:
        gradcam_layer = None  # use attention rollout instead (not yet implemented)
    else:
        gradcam_layer = None

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss, train_acc, n_batches = 0.0, 0.0, 0
        t0 = time.time()
        for img_a, img_b, labels in train_loader:
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(img_a, img_b)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_acc  += pairwise_accuracy(logits.detach(), labels)
            n_batches  += 1

        train_loss /= n_batches
        train_acc  /= n_batches

        # --- Val ---
        model.eval()
        val_loss, val_acc, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for img_a, img_b, labels in val_loader:
                img_a = img_a.to(device)
                img_b = img_b.to(device)
                labels = labels.to(device)
                logits = model(img_a, img_b)
                val_loss += criterion(logits, labels).item()
                val_acc  += pairwise_accuracy(logits, labels)
                n_val    += 1
        val_loss /= n_val
        val_acc  /= n_val

        # Kendall's tau + wandb visualizations every 5 epochs
        tau = 0.0
        if epoch % 5 == 0 or epoch == args.epochs:
            tau = evaluate_kendall_tau(
                model, args.frames_root, args.cuts_json,
                n_seq=args.tau_n_seq, seq_len=args.tau_seq_len, device=device,
            )
            if gradcam_layer:
                log_ordering_with_gradcam(
                    model, args.frames_root, args.cuts_json, device,
                    n_frames=10, target_layer=gradcam_layer, seed=epoch,
                )

        elapsed = time.time() - t0
        scheduler.step()

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "kendall_tau": tau,
        }
        log.append(entry)

        wandb.log({
            "train/loss":  train_loss,
            "train/acc":   train_acc,
            "val/loss":    val_loss,
            "val/acc":     val_acc,
            "kendall_tau": tau if tau else None,
            "lr/encoder":  optimizer.param_groups[0]["lr"],
            "lr/head":     optimizer.param_groups[1]["lr"],
            "epoch":       epoch,
        }, step=epoch)

        tau_str = f"  τ={tau:.4f}" if tau else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss {train_loss:.4f}/{val_loss:.4f} | "
            f"acc {train_acc:.4f}/{val_acc:.4f}{tau_str} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best.pt")

        with open(out_dir / "log.json", "w") as f:
            json.dump(log, f, indent=2)

    torch.save(model.state_dict(), out_dir / "last.pt")
    print(f"\nDone. Best val acc: {best_val_acc:.4f}")
    wandb.summary["best_val_acc"] = best_val_acc
    run.finish()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_root", type=Path, default=Path("frames"))
    p.add_argument("--cuts_json",   type=Path, default=Path("cuts.json"))
    p.add_argument("--encoder",     default="resnet50")
    p.add_argument("--embed_dim",   type=int, default=256)
    p.add_argument("--hidden_dim",  type=int, default=512)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--epochs",      type=int, default=20)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--window_size",       type=int, default=10)
    p.add_argument("--n_windows_per_cut", type=int, default=20)
    p.add_argument("--val_fraction",    type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--tau_n_seq",   type=int, default=50)
    p.add_argument("--tau_seq_len", type=int, default=30)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out",         default="checkpoints")
    p.add_argument("--wandb_project", default="cs280-temporal-ordering")
    p.add_argument("--wandb_run",     default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
