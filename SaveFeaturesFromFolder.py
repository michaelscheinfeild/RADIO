"""
SaveFeaturesFromFolder.py — Extract and save RADIO features for gallery folders.

Processes gallery images from D:\\OrthoIsr\\crop* folders:
  - crop1000  (1000x1000 crops)
  - crop1500  (1500x1500 crops)
  - crop2000  (2000x2000 crops)

For each folder:
  1. Loads every .tif image.
  2. Converts to grayscale (mean across 3 channels), then replicates to 3 identical channels.
  3. Runs RADIO model to extract a normalized summary feature vector.
  4. Saves features as .npy, a companion .txt metadata file (image name, source folder, TFW info),
     and a timing log with average preprocess+model inference time.

Output is saved under D:\\OrthoIsr\\data\\<folder_name>\\ for each crop folder.
"""
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_ROOT = Path(r"D:\OrthoIsr")
DATA_OUT = BASE_ROOT / "data"

CROP_FOLDERS = ["crop1000", "crop1500", "crop2000"]

REPO = r"C:\github\RADIO"
MODEL_VERSION = "c-radio_v4-h"


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(repo: str = REPO, version: str = MODEL_VERSION) -> torch.nn.Module:
    """Load RADIO model from local repo using cached weights (no re-download).

    Returns:
        The RADIO model on CUDA in eval mode.
    """
    source = "local" if os.path.exists(repo) else "github"
    model = torch.hub.load(
        repo, "radio_model", source=source, version=version,
        progress=True, skip_validation=True,
    )
    return model.cuda().eval()


# ── TFW parsing ────────────────────────────────────────────────────────────────
def read_tfw(tfw_path: Path) -> Dict[str, float]:
    """Read a 6-line TFW world file and return the affine parameters.

    TFW format (one value per line):
        A  — pixel size in X (meters/pixel)
        D  — rotation term Y
        B  — rotation term X
        E  — pixel size in Y (negative = north-up)
        C  — X coordinate of upper-left pixel center (UTM easting)
        F  — Y coordinate of upper-left pixel center (UTM northing)

    Returns:
        Dict with keys: A, D, B, E, C, F.
    """
    with open(tfw_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) != 6:
        raise ValueError(f"TFW must contain 6 values, got {len(lines)} in {tfw_path}")
    vals = [float(x) for x in lines]
    return {"A": vals[0], "D": vals[1], "B": vals[2], "E": vals[3], "C": vals[4], "F": vals[5]}


# ── Image discovery ────────────────────────────────────────────────────────────
def discover_tif_tfw_pairs(folder: Path) -> List[Tuple[Path, Path]]:
    """Find all .tif files in a folder that have a matching .tfw file.

    Args:
        folder: Path to the gallery crop folder.

    Returns:
        Sorted list of (tif_path, tfw_path) tuples.
    """
    pairs = []
    for tif_path in sorted(folder.glob("*.tif")):
        tfw_path = tif_path.with_suffix(".tfw")
        if tfw_path.exists():
            pairs.append((tif_path, tfw_path))
        else:
            print(f"  Warning: no .tfw for {tif_path.name}, skipping")
    return pairs


# ── Image loading with gray conversion ─────────────────────────────────────────
def load_image_as_gray3ch(tif_path: Path) -> Image.Image:
    """Load a .tif image, convert to grayscale by mean of 3 channels, then replicate to 3 channels.

    This simulates how an IR-like (single-band) image would look when fed
    to a model expecting 3-channel input.

    Args:
        tif_path: Path to the .tif file.

    Returns:
        PIL Image in RGB mode with all 3 channels identical (grayscale).
    """
    img = Image.open(tif_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)            # (H, W, 3)
    gray = np.round(arr.mean(axis=2)).astype(np.uint8)  # (H, W)
    gray_3ch = np.stack([gray, gray, gray], axis=2)      # (H, W, 3)
    return Image.fromarray(gray_3ch, mode="RGB")


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(model: torch.nn.Module, pil_img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a RADIO-ready tensor on CUDA.

    Steps:
        1. Convert to float32 tensor in [0, 1].
        2. Add batch dimension.
        3. Resize to nearest supported resolution.

    Args:
        model:   The RADIO model (used for get_nearest_supported_resolution).
        pil_img: Input PIL image (RGB).

    Returns:
        Tensor of shape (1, 3, H', W') on CUDA.
    """
    x = pil_to_tensor(pil_img).to(dtype=torch.float32, device="cuda")
    x.div_(255.0)
    x = x.unsqueeze(0)
    nearest_res = model.get_nearest_supported_resolution(*x.shape[-2:])
    x = F.interpolate(x, nearest_res, mode="bilinear", align_corners=False)
    return x


# ── Feature extraction for one image ──────────────────────────────────────────
@torch.no_grad()
def extract_single_feature(model: torch.nn.Module, pil_img: Image.Image) -> Tuple[np.ndarray, float]:
    """Run preprocess + model forward pass and return the normalized summary feature.

    Timing covers only preprocess + model inference (not gray conversion or saving).

    Args:
        model:   The RADIO model.
        pil_img: PIL image (RGB, already gray-converted if needed).

    Returns:
        (feature_vector [1, D] as numpy, elapsed_seconds for preprocess+model).
    """
    t0 = time.perf_counter()
    x = preprocess(model, pil_img)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        summary, _ = model(x)
    summary = F.normalize(summary, dim=-1).cpu().numpy()
    elapsed = time.perf_counter() - t0
    return summary, elapsed


# ── Metadata text file ─────────────────────────────────────────────────────────
def build_metadata_lines(
    pairs: List[Tuple[Path, Path]],
    folder_name: str,
) -> List[str]:
    """Build metadata lines for the companion .txt file.

    Each line contains:
        image_name, source_folder, A, D, B, E, C, F

    Args:
        pairs:       List of (tif_path, tfw_path).
        folder_name: Name of the source crop folder (e.g. 'crop1000').

    Returns:
        List of CSV-formatted strings, one per image.
    """
    lines = []
    for tif_path, tfw_path in pairs:
        tfw = read_tfw(tfw_path)
        line = (
            f"{tif_path.name}, {folder_name}, "
            f"{tfw['A']:.10f}, {tfw['D']:.10f}, {tfw['B']:.10f}, "
            f"{tfw['E']:.10f}, {tfw['C']:.6f}, {tfw['F']:.6f}"
        )
        lines.append(line)
    return lines


# ── Process one crop folder ────────────────────────────────────────────────────
def process_folder(
    model: torch.nn.Module,
    folder: Path,
    out_dir: Path,
) -> None:
    """Extract RADIO features for all images in a gallery crop folder.

    For each image:
        - Convert to grayscale (mean of 3 channels), replicate to 3 channels.
        - Run preprocess + model to get a summary feature vector.
        - Time only the preprocess + model step.

    Saves:
        <out_dir>/radio_features_<folder_name>.npy   — (N, D) feature array
        <out_dir>/radio_features_<folder_name>.txt   — metadata per image
        <out_dir>/radio_timing_<folder_name>.log     — per-image timing + averages

    Args:
        model:   The RADIO model.
        folder:  Path to the crop folder (e.g. D:\\OrthoIsr\\crop1000).
        out_dir: Path to save outputs.
    """
    folder_name = folder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {folder_name}  ({folder})")
    print(f"Output:     {out_dir}")
    print(f"{'='*60}")

    # Discover image pairs
    pairs = discover_tif_tfw_pairs(folder)
    n = len(pairs)
    print(f"  Found {n} .tif/.tfw pairs")
    if n == 0:
        print("  Nothing to process, skipping.")
        return

    # Build metadata
    meta_lines = build_metadata_lines(pairs, folder_name)

    # Extract features
    all_features = []
    timings = []

    for i, (tif_path, tfw_path) in enumerate(pairs):
        if (i + 1) % 200 == 0 or i == 0 or i == n - 1:
            print(f"  [{folder_name}] {i+1}/{n}")

        # Gray conversion (not timed)
        pil_img = load_image_as_gray3ch(tif_path)

        # Feature extraction (timed: preprocess + model only)
        feat, elapsed = extract_single_feature(model, pil_img)

        all_features.append(feat)
        timings.append(elapsed)

    features = np.concatenate(all_features, axis=0)  # (N, D)
    timings_arr = np.array(timings, dtype=np.float64)

    # ── Save outputs ───────────────────────────────────────────────────────────
    npy_path = out_dir / f"radio_features_{folder_name}.npy"
    txt_path = out_dir / f"radio_features_{folder_name}.txt"
    log_path = out_dir / f"radio_timing_{folder_name}.log"

    # Features
    np.save(npy_path, features)
    print(f"  Saved features: {npy_path}  shape={features.shape}")

    # Metadata text
    txt_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    print(f"  Saved metadata: {txt_path}")

    # Timing log
    avg_ms = timings_arr.mean() * 1000
    std_ms = timings_arr.std() * 1000
    total_s = timings_arr.sum()
    log_lines = [
        f"folder={folder_name}",
        f"model={MODEL_VERSION}",
        f"num_images={n}",
        f"feature_dim={features.shape[1]}",
        f"total_inference_time_s={total_s:.2f}",
        f"avg_inference_time_ms={avg_ms:.2f}",
        f"std_inference_time_ms={std_ms:.2f}",
        f"min_inference_time_ms={timings_arr.min()*1000:.2f}",
        f"max_inference_time_ms={timings_arr.max()*1000:.2f}",
        "",
        "# Per-image timing (preprocess + model forward, excludes gray conversion and save):",
        "# index, image_name, time_ms",
    ]
    for i, (tif_path, _) in enumerate(pairs):
        log_lines.append(f"{i}, {tif_path.name}, {timings[i]*1000:.2f}")

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"  Saved timing log: {log_path}")
    print(f"  Avg inference: {avg_ms:.2f} ms/image  (std={std_ms:.2f} ms)")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    """Main entry point: load model once, process each crop folder."""
    print("Loading RADIO model...")
    model = load_model()

    for folder_name in CROP_FOLDERS:
        folder = BASE_ROOT / folder_name
        if not folder.exists():
            print(f"Warning: folder {folder} does not exist, skipping.")
            continue
        out_dir = DATA_OUT / folder_name
        process_folder(model, folder, out_dir)

    print(f"\nAll done! Results saved under: {DATA_OUT}")


if __name__ == "__main__":
    main()
