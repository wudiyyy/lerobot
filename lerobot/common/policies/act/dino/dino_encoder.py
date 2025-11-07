"""DINOv3 Encoder wrapper for ACT policy."""

import torch
import torch.nn as nn
import io
import numpy as np
import math
import os
from lerobot.common.policies.act.dino.config import *
from transformers import AutoModel, AutoImageProcessor
# from modelscope import AutoModel, AutoImageProcessor
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import requests
from PIL import Image
import matplotlib.pyplot as plt
class DINOEncoder(nn.Module):
    """
    DINOv3 encoder that wraps the pretrained model and provides
    an interface compatible with ACT's ResNet backbone.

    Returns output in the same format as IntermediateLayerGetter:
    {"feature_map": tensor of shape (B, C, H, W)}
    """
    def __init__(self, model_name: str, output_dim: int = 512, freeze: bool = True, model_dir: str = DINOV3_LOCATION):
        """
        Args:
            model_name: Name of the DINOv3 model (e.g., "dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16")
            output_dim: Output dimension after projection
            freeze: Whether to freeze the DINO backbone weights
            model_dir: Directory of the DINOv3 model
        """
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.freeze = freeze
        self.model_dir = model_dir

        assert model_name in MODEL_TO_NUM_LAYERS, f"Model name {model_name} not in {MODEL_TO_NUM_LAYERS}"

        # Load pretrained DINOv3 model
        if "dinov3" in model_name:
            # Load from torch hub or local path
            # Adjust the repo path according to your DINOv3 setup
            try:
                # Try loading from local path first (if you have cloned the repo)
                self.dino = AutoModel.from_pretrained(self.model_dir)

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load DINOv3 model '{model_name}'. "
                    f"Please ensure the model is available. Error: {e}"
                )
            
        if freeze:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()

        self.n_layers = MODEL_TO_NUM_LAYERS[self.model_name]
        # Projection layer to match output dimension
        # Use Conv2d to match ResNet's output format
        self.projection = nn.Conv2d(MODEL_TO_HIDDEN_DIM[self.model_name], output_dim, kernel_size=1)
        # Store feature dimension for compatibility with ACT code
        self.fc = nn.Module()  # Dummy module for compatibility
        self.fc.in_features = output_dim


    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input images of shape (B, C, H, W)

        Returns:
            Dictionary with "feature_map" key containing tensor of shape (B, output_dim, h, w)
        """
        B, C, H, W = x.shape
        if self.freeze:
            with torch.no_grad():
                outputs = self.dino(pixel_values=x, output_hidden_states=True)
                hidden_states = outputs.last_hidden_state  # (B, N, D)
                patch_features = hidden_states[:, 5:, :]  # (B*T, num_patches, dinov3_dim)
        else:
            raise NotImplementedError("DINOv3 encoder does not support training")
        patch = getattr(self.dino.config, "patch_size", 16)
        H = x.shape[-2] // patch
        W = x.shape[-1] // patch

        assert H * W == patch_features.shape[1], "grid size mismatch"
        B,N,C = patch_features.shape
        feat_map = patch_features.transpose(1, 2).reshape(B, C, H, W)     # (B, 768, H, W)
        feature_map = self.projection(feat_map)  # (B, num_patches, output_dim)

        # Return in the same format as IntermediateLayerGetter
        return {"feature_map": feature_map}

    def train(self, mode: bool = True):
        """Override train mode to keep DINO frozen if requested."""
        super().train(mode)
        if self.freeze:
            self.dino.eval()
        return self


# --- helper: download image from URL ---
def load_image_from_url(url, resize_short=384):
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    # resize so shorter side == resize_short while keep aspect ratio
    w, h = image.size
    if min(w, h) != resize_short:
        if w < h:
            new_w = resize_short
            new_h = int(h * resize_short / w)
        else:
            new_h = resize_short
            new_w = int(w * resize_short / h)
        image = image.resize((new_w, new_h))
    return image

# --- visualization helpers ---
def upsample_to_image(feature_map: torch.Tensor, image_size: tuple):
    """
    feature_map: (C, h, w) or (1, C, h, w)
    image_size: (H, W)
    returns: numpy array (C, H, W)
    """
    if feature_map.dim() == 3:
        feature_map = feature_map.unsqueeze(0)
    # upsample using bilinear
    up = F.interpolate(feature_map, size=image_size, mode="bilinear", align_corners=False)
    up = up.squeeze(0).cpu().numpy()
    return up  # (C, H, W)
def normalize01(x: np.ndarray):
    x = x - x.min()
    if x.max() > 0:
        x = x / (x.max() + 1e-12)
    return x

def plot_feature_overlays(original_pil: Image.Image, upsampled_feats: np.ndarray, topk=4, out_path="dinov3_feature_maps.png"):
    """
    original_pil: PIL Image (RGB)
    upsampled_feats: (C, H, W) numpy array
    pick topk channels by L2 energy and plot them as overlay heatmaps
    """
    C, H, W = upsampled_feats.shape
    # compute energy per channel
    energies = np.linalg.norm(upsampled_feats.reshape(C, -1), axis=1)
    topk_idx = list(np.argsort(energies)[-topk:][::-1])

    fig_rows = 1 + topk  # original + topk
    fig, axs = plt.subplots(1, fig_rows, figsize=(4 * fig_rows, 4))
    axs[0].imshow(original_pil)
    axs[0].set_title("Original")
    axs[0].axis("off")

    orig_np = np.array(original_pil).astype(np.float32) / 255.0

    for i, ch in enumerate(topk_idx):
        ax = axs[i + 1]
        fm = upsampled_feats[ch]
        fm = normalize01(fm)
        # overlay: convert heatmap to RGBA
        cmap = plt.get_cmap("jet")
        heat_rgba = cmap(fm)  # H,W,4
        # blend heatmap with original
        alpha = 0.5
        blended = (1 - alpha) * orig_np + alpha * heat_rgba[..., :3]
        blended = np.clip(blended, 0, 1)
        ax.imshow(blended)
        ax.set_title(f"Channel {ch} (energy {energies[ch]:.1f})")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")
    plt.close(fig)

# --- main test routine ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # choose model. you can set model_dir to a local checkpoint if you have one.
    model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"  # alternatives: dinov3_vitb16, dinov3_vitl16 (if available)
    output_dim = 512  # projection channels to visualize
    # 初始化图像预处理器（会尝试从 HF 下载 tokenizer/config）
    try:
        model_dir = model_name
        processor = AutoImageProcessor.from_pretrained(model_dir)
    except Exception as e:
        print("Warning: failed to load AutoImageProcessor from", model_dir, "-> trying model_name. Error:", e)
        processor = AutoImageProcessor.from_pretrained(model_name)

    print("Loading model:", model_name)
    encoder = DINOEncoder(model_name=model_name, output_dim=output_dim, freeze=True, model_dir=model_name)
    encoder.to(device)
    # sample image urls (common public images)
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",  # a PNG demo (small)
        "https://images.unsplash.com/photo-1519681393784-d120267933ba",  # unsplash sample (landscape)
    ]
    # pick the first one for simplicity
    url = urls[1]
    print("Downloading image:", url)
    img = load_image_from_url(url, resize_short=480)
    orig_w, orig_h = img.size
    print("Original image size:", img.size)


if __name__ == "__main__":
    main()