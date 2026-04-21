import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

# --- Configuration ---
#dbIr = r"D:\OrthoCrop\tifquery"
dbIr = r"D:\OrthoAshdod2\tifquery"
frames = [1690, 1750, 1870, 1990, 2050, 2710, 2770]

REPO = r"C:\github\RADIO"
MODEL_VERSION = "c-radio_v4-h"


def load_model(repo=REPO, version=MODEL_VERSION, adaptor_names=None):
    """Load RADIO model from local repo, using cached weights (no re-download)."""
    source = "local" if os.path.exists(repo) else "github"
    model = torch.hub.load(
        repo, "radio_model", source=source, version=version,
        progress=True, skip_validation=True,
        adaptor_names=adaptor_names,
    )
    return model.cuda().eval()


def load_images(db_path, frame_ids):
    """Load images from db_path matching frame IDs. Returns list of (frame_id, PIL.Image)."""
    images = []
    for fid in frame_ids:
        pattern = os.path.join(db_path, f"*{fid}*.*")
        matches = glob.glob(pattern)
        if not matches:
            print(f"Warning: no file found for frame {fid}")
            continue
        path = matches[0]
        img = Image.open(path).convert("RGB")
        images.append((fid, img, path))
    return images


def preprocess(model, pil_img):
    """Convert PIL image to model-ready tensor."""
    x = pil_to_tensor(pil_img).to(dtype=torch.float32, device="cuda")
    x.div_(255.0)
    x = x.unsqueeze(0)
    nearest_res = model.get_nearest_supported_resolution(*x.shape[-2:])
    x = F.interpolate(x, nearest_res, mode="bilinear", align_corners=False)
    return x


@torch.no_grad()
def extract_features(model, images):
    """Extract summary features for all images. Returns tensor [N, D]."""
    summaries = []
    for fid, img, path in images:
        x = preprocess(model, img)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            summary, _ = model(x)
        summaries.append(F.normalize(summary, dim=-1))
    return torch.cat(summaries, dim=0)  # [N, D]


def compute_similarity_matrix(features):
    """Compute pairwise cosine similarity matrix [N, N]."""
    return (features @ features.T).cpu().numpy()


def display_images(images):
    """Display all loaded images in a row."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (fid, img, _) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(f"Frame {fid}")
        ax.axis("off")
    fig.suptitle("All Images", fontsize=14)
    plt.tight_layout()
    plt.show()


def display_similarity_matrix(sim_matrix, frame_ids):
    """Display the similarity matrix as a colored heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(frame_ids)))
    ax.set_yticks(range(len(frame_ids)))
    ax.set_xticklabels(frame_ids, rotation=45)
    ax.set_yticklabels(frame_ids)
    for i in range(len(frame_ids)):
        for j in range(len(frame_ids)):
            ax.text(j, i, f"{sim_matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title("Pairwise Cosine Similarity")
    plt.tight_layout()
    plt.show()


def display_query_rankings(images, sim_matrix):
    """For each image as query, show all others ranked by similarity."""
    n = len(images)
    frame_ids = [fid for fid, _, _ in images]

    for qi in range(n):
        sims = sim_matrix[qi]
        order = np.argsort(-sims)  # descending
        # exclude self
        order = [idx for idx in order if idx != qi]

        fig, axes = plt.subplots(1, 1 + len(order), figsize=(4 * (1 + len(order)), 4))
        # Query image
        axes[0].imshow(images[qi][1])
        axes[0].set_title(f"QUERY\nFrame {frame_ids[qi]}", fontweight="bold")
        axes[0].axis("off")

        for col, idx in enumerate(order, start=1):
            axes[col].imshow(images[idx][1])
            axes[col].set_title(f"Frame {frame_ids[idx]}\nsim={sims[idx]:.4f}")
            axes[col].axis("off")

        fig.suptitle(f"Query: Frame {frame_ids[qi]} — matches by similarity (high → low)", fontsize=12)
        plt.tight_layout()
        plt.show()


# --- Main ---
if __name__ == "__main__":
    # 1) Load images
    images = load_images(dbIr, frames)
    print(f"Loaded {len(images)} images")

    # 2) Display all images
    display_images(images)

    # 3) Load model (uses cached weights, no force_reload)
    model = load_model()

    # 4) Extract features and compute similarity
    features = extract_features(model, images)
    sim_matrix = compute_similarity_matrix(features)

    # 5) Display similarity heatmap
    frame_ids = [fid for fid, _, _ in images]
    display_similarity_matrix(sim_matrix, frame_ids)

    # 6) Display per-query rankings
    display_query_rankings(images, sim_matrix)
    print("That's all folks!")