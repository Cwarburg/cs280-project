"""
CutWindowDataset: samples 10-frame windows that always straddle a camera cut.

For each window all C(10,2)=45 pairs are generated with temporal labels.
Every pair either:
  - crosses the cut  → high visual contrast, unambiguous temporal signal
  - stays within one scene → subtle but correctly labelled

This avoids the problem of nearly-identical within-shot pairs sampled far
from any cut, where the model has no meaningful signal to learn from.
"""
import json
import random
from itertools import combinations
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_MASK_REGIONS = [
    (90,  0, 510, 90),    # scoreboard top-left
    (1130, 0, 1280, 90),  # watermark top-right
]


class _MaskRegions:
    def __call__(self, img: Image.Image) -> Image.Image:
        from PIL import ImageDraw
        img = img.copy()
        draw = ImageDraw.Draw(img)
        for region in _MASK_REGIONS:
            draw.rectangle(region, fill=0)
        return img


def get_transform(train: bool, size: int = 224) -> transforms.Compose:
    if train:
        return transforms.Compose([
            _MaskRegions(),
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    return transforms.Compose([
        _MaskRegions(),
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class CutWindowDataset(Dataset):
    """
    Args:
        frames_root:       root directory containing per-half frame folders
        cuts_json:         JSON mapping relative_path → list of cut frame indices
        window_size:       total frames per window (default 10)
        n_windows_per_cut: how many different windows to sample per cut
        min_scene_frames:  minimum frames required on each side of the cut
        train:             whether to apply training augmentations
        seed:              random seed
    """

    def __init__(
        self,
        frames_root: Path,
        cuts_json: Path,
        window_size: int = 10,
        n_windows_per_cut: int = 20,
        min_scene_frames: int = 2,
        train: bool = True,
        seed: int = 42,
    ):
        self.frames_root = Path(frames_root)
        self.transform   = get_transform(train)
        rng = random.Random(seed)

        with open(cuts_json) as f:
            cuts_map: dict[str, list[int]] = json.load(f)

        # Each entry: list of frame paths for one window (in temporal order)
        # We store path lists; pairs are generated on the fly in __getitem__
        # to keep memory low — but we pre-build the pair index here.
        #
        # pairs: list of (path_a, path_b, label)
        self.pairs: list[tuple[Path, Path, int]] = []

        n_cross  = 0
        n_within = 0

        for rel_key, cut_indices in cuts_map.items():
            half_dir   = self.frames_root / rel_key
            all_frames = sorted(half_dir.glob("*.jpg"))
            n = len(all_frames)

            if n < window_size + 2:
                continue

            for cut_frame_no in cut_indices:
                # Find position of this cut in the sorted frame list
                stems = [int(f.stem) for f in all_frames]
                try:
                    cut_pos = stems.index(cut_frame_no)
                except ValueError:
                    cut_pos = min(range(n), key=lambda i: abs(stems[i] - cut_frame_no))

                # Need at least min_scene_frames on each side
                if cut_pos < min_scene_frames or cut_pos >= n - min_scene_frames:
                    continue

                for _ in range(n_windows_per_cut):
                    # Randomly pick how many frames come from scene 1 vs scene 2.
                    # Scene 1 gets between min_scene_frames and
                    # window_size - min_scene_frames frames.
                    n_before = rng.randint(
                        min_scene_frames,
                        min(window_size - min_scene_frames, cut_pos),
                    )
                    n_after  = window_size - n_before
                    n_after  = min(n_after, n - cut_pos - 1)
                    if n_after < min_scene_frames:
                        continue

                    # Sample contiguous frames from each scene ending/starting at cut
                    before_start = cut_pos - n_before
                    after_end    = cut_pos + n_after  # exclusive

                    window_paths = all_frames[before_start : cut_pos] + \
                                   all_frames[cut_pos + 1 : after_end + 1]

                    if len(window_paths) < 2:
                        continue

                    # Generate all pairs within this window
                    # Temporal position = index in window_paths
                    for i, j in combinations(range(len(window_paths)), 2):
                        # i < j always (combinations are ordered)
                        label = 1  # i comes before j
                        if rng.random() < 0.5:
                            i, j = j, i
                            label = 0
                        self.pairs.append((window_paths[i], window_paths[j], label))

                        # Track cross vs within for stats
                        if i < n_before and j >= n_before:
                            n_cross += 1
                        elif j < n_before and i >= n_before:
                            n_cross += 1
                        else:
                            n_within += 1

        rng.shuffle(self.pairs)
        self.n_cross  = n_cross
        self.n_within = n_within

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        path_a, path_b, label = self.pairs[idx]
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        return self.transform(img_a), self.transform(img_b), torch.tensor(label, dtype=torch.float32)
