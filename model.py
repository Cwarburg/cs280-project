"""
Pairwise temporal ordering model.

Architecture:
  - Shared CNN encoder φ (ResNet-50 or ViT-S/16 via timm)
  - Classifier head: MLP over [e_i || e_j]  →  scalar logit
  - Loss: binary cross-entropy

At inference: build n×n score matrix S[i,j] = P(i before j),
then rank frames by row sums.
"""
import torch
import torch.nn as nn
import timm


class PairwiseOrderingModel(nn.Module):
    """
    Args:
        encoder_name: timm model name, e.g. 'resnet50', 'vit_small_patch16_224'
        pretrained: load ImageNet weights
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

        # Shared encoder
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,   # remove classification head, return features
        )
        feat_dim = self.encoder.num_features

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

        # Classifier: [e_i || e_j] → scalar
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(x))

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """Returns logit (scalar per pair). Positive → a comes before b."""
        e_a = self.encode(img_a)
        e_b = self.encode(img_b)
        return self.classifier(torch.cat([e_a, e_b], dim=-1)).squeeze(-1)

    def score_matrix(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Compute n×n score matrix for a sequence of n frames.
        S[i,j] = sigmoid(logit(frame_i, frame_j)) = P(i before j)

        Args:
            frames: (n, C, H, W)
        Returns:
            S: (n, n) float tensor
        """
        n = frames.size(0)
        embeddings = self.encode(frames)          # (n, d)

        # Broadcast to build all pairwise differences
        e_i = embeddings.unsqueeze(1).expand(n, n, -1)   # (n, n, d)
        e_j = embeddings.unsqueeze(0).expand(n, n, -1)   # (n, n, d)

        logits = self.classifier(torch.cat([e_i, e_j], dim=-1)).squeeze(-1)  # (n, n)
        return torch.sigmoid(logits)

    def rank_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Return indices that sort frames in temporal order (ascending)."""
        S = self.score_matrix(frames)
        scores = S.sum(dim=1)   # row sum = total "before" votes
        return torch.argsort(scores, descending=True)
