"""
Pairwise temporal ordering model.

Architecture:
  - Shared CNN encoder φ (ResNet-50 or ViT-S/16 via timm)
  - Classifier head: MLP over [e_i || e_j]  →  scalar logit
  - Loss: binary cross-entropy

At inference: build n×n score matrix S[i,j] = P(i before j),
then rank frames by row sums.

ResNet: 6-channel input [frame || frame-neighbor] encodes motion.
ViT (DINO): standard 3-channel input; DINO weights kept untouched.
"""
import numpy as np
import torch
import torch.nn as nn
import timm


class PairwiseOrderingModel(nn.Module):
    """
    Args:
        encoder_name: timm model name, e.g. 'resnet50', 'vit_small_patch16_224.dino'
        pretrained: load pretrained weights
        embed_dim: output dim of projection head (None = use raw encoder output)
        hidden_dim: hidden dim of classifier MLP
        dropout: dropout in classifier head
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        pretrained: bool = True,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.is_vit = "vit" in encoder_name

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,
        )
        feat_dim = self.encoder.num_features

        if "resnet" in encoder_name:
            # 6-channel input: [frame, frame - neighbor] captures motion
            old_conv = self.encoder.conv1
            self.encoder.conv1 = nn.Conv2d(
                6, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                self.encoder.conv1.weight[:, :3] = old_conv.weight
                self.encoder.conv1.weight[:, 3:] = old_conv.weight * 0.1

        elif self.is_vit:
            # Keep DINO weights entirely untouched — standard 3-channel input.
            # Disable fused attention so attention weights are accessible for viz.
            for block in self.encoder.blocks:
                if hasattr(block, 'attn') and hasattr(block.attn, 'fused_attn'):
                    block.attn.fused_attn = False

        # Optional projection to lower-dim embedding
        if embed_dim is not None and embed_dim != feat_dim:
            self.projector = nn.Sequential(
                nn.Linear(feat_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )
            diff_dim = embed_dim * 2
        else:
            self.projector = nn.Identity()
            diff_dim = feat_dim * 2

        # Classifier: [e_i - e_j || e_i * e_j] → scalar
        self.classifier = nn.Sequential(
            nn.Linear(diff_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, x: torch.Tensor, x_nbr: torch.Tensor) -> torch.Tensor:
        if self.is_vit:
            return self.projector(self.encoder(x))
        diff = x - x_nbr
        return self.projector(self.encoder(torch.cat([x, diff], dim=1)))

    def get_attention_maps(self, x: torch.Tensor) -> "np.ndarray | None":
        """
        Extract CLS-token attention maps from the last transformer block.

        Args:
            x: (1, 3, H, W)
        Returns:
            (num_heads, h_patch, w_patch) float32 array normalised to [0, 1],
            or None for non-ViT encoders.
        """
        if not self.is_vit:
            return None

        captured = []

        def _hook(module, inp, out):
            captured.append(inp[0].detach())   # inp[0]: (B, heads, N, N)

        hook = self.encoder.blocks[-1].attn.attn_drop.register_forward_hook(_hook)
        with torch.no_grad():
            self.encoder(x)
        hook.remove()

        if not captured:
            return None

        attn = captured[0][0]        # (heads, N, N)
        cls  = attn[:, 0, 1:]        # (heads, num_patches) — CLS attends to patches
        h = w = int(cls.shape[-1] ** 0.5)
        maps = cls.reshape(-1, h, w).cpu().numpy()   # (heads, h, w)

        for i in range(maps.shape[0]):
            lo, hi = maps[i].min(), maps[i].max()
            if hi > lo:
                maps[i] = (maps[i] - lo) / (hi - lo)

        return maps

    def forward(self, img_a: torch.Tensor, img_a_nbr: torch.Tensor,
                img_b: torch.Tensor, img_b_nbr: torch.Tensor) -> torch.Tensor:
        """Returns logit (scalar per pair). Positive → a comes before b."""
        e_a = self.encode(img_a, img_a_nbr)
        e_b = self.encode(img_b, img_b_nbr)
        return self.classifier(torch.cat([e_a - e_b, e_a * e_b], dim=-1)).squeeze(-1)
        
    def score_matrix(self, frames: torch.Tensor, frame_nbrs: torch.Tensor) -> torch.Tensor:
        """
        Compute n×n score matrix for a sequence of n frames.
        S[i,j] = sigmoid(logit(frame_i, frame_j)) = P(i before j)

        Args:
            frames:     (n, 3, H, W)
            frame_nbrs: (n, 3, H, W) — temporal neighbor of each frame
        Returns:
            S: (n, n) float tensor
        """
        n = frames.size(0)
        embeddings = self.encode(frames, frame_nbrs)      # (n, d)

        e_i = embeddings.unsqueeze(1).expand(n, n, -1)   # (n, n, d)
        e_j = embeddings.unsqueeze(0).expand(n, n, -1)   # (n, n, d)

        feat   = torch.cat([e_i - e_j, e_i * e_j], dim=-1)
        logits = self.classifier(feat).squeeze(-1)
        S      = torch.sigmoid(logits)
        S      = (S + (1 - S.T)) / 2                       # enforce anti-symmetry
        S      = S * (1 - torch.eye(n, device=S.device))   # zero diagonal
        return S

    def rank_frames(self, frames: torch.Tensor, frame_nbrs: torch.Tensor) -> torch.Tensor:
        """Return indices that sort frames in temporal order (ascending)."""
        S = self.score_matrix(frames, frame_nbrs)
        return torch.argsort(S.sum(dim=1), descending=True)