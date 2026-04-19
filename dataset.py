"""
CutWindowDataset: builds pairs from frames within a fixed radius of each cut.

For each cut, the cut_radius frames before and cut_radius frames after are used
(2 * cut_radius frames per cut). All C(2R, 2) pairs are generated with temporal
labels, giving a clean signal concentrated around scene boundaries.
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
        frames_root: root directory containing per-half frame folders
        cuts_json:   JSON mapping relative_path → list of cut frame indices
        cut_radius:  number of frames to take from each side of the cut (default 5)
        train:       whether to apply training augmentations
        seed:        random seed for pair shuffling
    """

    def __init__(
        self,
        frames_root: Path,
        cuts_json: Path,
        cut_radius: int = 5,
        train: bool = True,
        seed: int = 42,
        cross_cut_weight: float = 0.5,
    ):
        self.frames_root = Path(frames_root)
        self.transform   = get_transform(train)
        rng = random.Random(seed)

        with open(cuts_json) as f:
            cuts_map: dict[str, list[int]] = json.load(f)

        self.pairs: list[tuple[Path, Path, Path, Path, int, float]] = []

        n_cross  = 0
        n_within = 0

        for rel_key, cut_indices in cuts_map.items():
            half_dir   = self.frames_root / rel_key
            all_frames = sorted(half_dir.glob("*.jpg"))
            n = len(all_frames)

            if n < cut_radius * 2 + 1:
                continue

            stems = [int(f.stem) for f in all_frames]

            for cut_frame_no in cut_indices:
                try:
                    cut_pos = stems.index(cut_frame_no)
                except ValueError:
                    cut_pos = min(range(n), key=lambda i: abs(stems[i] - cut_frame_no))

                if cut_pos < cut_radius or cut_pos + cut_radius >= n:
                    continue

                before = all_frames[cut_pos - cut_radius : cut_pos]
                after  = all_frames[cut_pos + 1 : cut_pos + 1 + cut_radius]
                window = before + after

                for i, j in combinations(range(len(window)), 2):
                    i_side = 0 if i < cut_radius else 1
                    j_side = 0 if j < cut_radius else 1
                    is_cross = i_side != j_side

                    if is_cross:
                        n_cross += 1
                        weight = cross_cut_weight
                    else:
                        n_within += 1
                        weight = 1.0

                    # clamp neighbors within their own side of the cut
                    if i < cut_radius:
                        i_nbr = i + 1 if i + 1 < cut_radius else i - 1
                    else:
                        i_nbr = i + 1 if i + 1 < len(window) else i - 1
                    if j < cut_radius:
                        j_nbr = j + 1 if j + 1 < cut_radius else j - 1
                    else:
                        j_nbr = j + 1 if j + 1 < len(window) else j - 1

                    label = 1
                    if rng.random() < 0.5:
                        i, j, i_nbr, j_nbr = j, i, j_nbr, i_nbr
                        label = 0
                    self.pairs.append((window[i], window[i_nbr], window[j], window[j_nbr], label, weight))

        rng.shuffle(self.pairs)
        self.n_cross  = n_cross
        self.n_within = n_within

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        path_a, path_a_nbr, path_b, path_b_nbr, label, weight = self.pairs[idx]
        img_a     = self.transform(Image.open(path_a).convert("RGB"))
        img_a_nbr = self.transform(Image.open(path_a_nbr).convert("RGB"))
        img_b     = self.transform(Image.open(path_b).convert("RGB"))
        img_b_nbr = self.transform(Image.open(path_b_nbr).convert("RGB"))
        return (img_a, img_a_nbr, img_b, img_b_nbr,
                torch.tensor(label, dtype=torch.float32),
                torch.tensor(weight, dtype=torch.float32))