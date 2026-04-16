"""
GradCAM for the pairwise ordering model.

Computes the gradient of the ordering logit P(frame_a before frame_b)
w.r.t. the final convolutional feature map of the encoder,
producing a spatial saliency map for each input frame.

Usage:
    from gradcam import GradCAM
    cam = GradCAM(model, target_layer="encoder.layer4")
    saliency_a, saliency_b = cam(img_a, img_b)  # (H, W) tensors in [0,1]
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class GradCAM:
    """
    Args:
        model: PairwiseOrderingModel
        target_layer: dot-separated attribute path to the convolutional layer,
                      e.g. 'encoder.layer4' for ResNet-50
    """

    def __init__(self, model: nn.Module, target_layer: str = "encoder.layer4"):
        self.model = model
        self.model.eval()
        self._hooks: list = []
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Resolve layer by dot path
        layer = model
        for attr in target_layer.split("."):
            layer = getattr(layer, attr)
        self.layer = layer
        self._register_hooks()

    def _register_hooks(self):
        def save_activation(module, input, output):
            self._activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(self.layer.register_forward_hook(save_activation))
        self._hooks.append(self.layer.register_full_backward_hook(save_gradient))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        target: str = "a",   # "a" or "b" — which frame to attribute
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            img_a, img_b: (1, C, H, W) tensors (single pair, with grad)
            target: 'a' or 'b' selects which input to attribute the logit to

        Returns:
            (saliency_a, saliency_b): each (H', W') numpy array in [0, 1]
                                      where H', W' match the input spatial dims
        """
        img_a = img_a.requires_grad_(True)
        img_b = img_b.requires_grad_(True)

        # Forward through the chosen frame first so hooks fire for it
        if target == "a":
            self.model.zero_grad()
            logit = self.model(img_a, img_b)
            logit.backward()
            cam_a = self._compute_cam()
            cam_a = self._resize(cam_a, img_a.shape[-2:])

            self.model.zero_grad()
            logit = self.model(img_b, img_a)   # flip order for img_b attribution
            logit.backward()
            cam_b = self._compute_cam()
            cam_b = self._resize(cam_b, img_b.shape[-2:])
        else:
            self.model.zero_grad()
            logit = self.model(img_b, img_a)
            logit.backward()
            cam_b = self._compute_cam()
            cam_b = self._resize(cam_b, img_b.shape[-2:])

            self.model.zero_grad()
            logit = self.model(img_a, img_b)
            logit.backward()
            cam_a = self._compute_cam()
            cam_a = self._resize(cam_a, img_a.shape[-2:])

        return cam_a, cam_b

    def _compute_cam(self) -> np.ndarray:
        """Pool gradients over spatial dims, weight activations."""
        grads = self._gradients       # (1, C, H, W)
        acts  = self._activations     # (1, C, H, W)
        weights = grads.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1).squeeze(0)     # (H, W)
        cam = torch.relu(cam).cpu().numpy()
        return cam

    @staticmethod
    def _resize(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Resize cam to (H, W) and normalize to [0, 1]."""
        from PIL import Image as PILImage
        h, w = size
        pil = PILImage.fromarray(cam).resize((w, h), PILImage.BILINEAR)
        arr = np.array(pil, dtype=np.float32)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        return arr


def visualize_pair(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    cam_a: np.ndarray,
    cam_b: np.ndarray,
    label: Optional[int] = None,
    pred_logit: Optional[float] = None,
    save_path: Optional[str] = None,
):
    """Overlay GradCAM heatmaps on the input frames and display/save."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    def to_rgb(t):
        img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = img * std + mean
        return np.clip(img, 0, 1)

    rgb_a = to_rgb(img_a)
    rgb_b = to_rgb(img_b)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(rgb_a);         axes[0, 0].set_title("Frame A")
    axes[0, 1].imshow(rgb_b);         axes[0, 1].set_title("Frame B")
    axes[1, 0].imshow(rgb_a)
    axes[1, 0].imshow(cam_a, alpha=0.5, cmap="jet"); axes[1, 0].set_title("GradCAM A")
    axes[1, 1].imshow(rgb_b)
    axes[1, 1].imshow(cam_b, alpha=0.5, cmap="jet"); axes[1, 1].set_title("GradCAM B")

    title_parts = []
    if label is not None:
        title_parts.append(f"Label: {'A before B' if label == 1 else 'B before A'}")
    if pred_logit is not None:
        prob = 1 / (1 + np.exp(-pred_logit))
        title_parts.append(f"P(A before B)={prob:.2f}")
    if title_parts:
        fig.suptitle("  |  ".join(title_parts), fontsize=12)

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
