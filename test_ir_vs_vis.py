"""
IR vs VIS retrieval evaluation using RADIO (C-RADIO v4-H).

- IR query images from D:\OrthoCropAshdodHuge\tifquery
- VIS gallery images from D:\OrthoCropAshdodHuge\tif
- Metadata from test_query.txt / test_gallery.txt
- TFW files for gallery geolocation, query UTM from txt
- Computes cosine similarity with RADIO features
- Saves top-5 panels, similarity histograms, and statistics
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_ROOT = Path(r"D:\OrthoCropAshdodHuge")
DATA_ROOT = BASE_ROOT / "data" / "ortho"

GALLERY_TXT = DATA_ROOT / "test_gallery.txt"
QUERY_TXT = DATA_ROOT / "test_query.txt"

QUERY_DIR = BASE_ROOT / "tifquery"
GALLERY_DIR = BASE_ROOT / "tif"

OUT_ROOT = BASE_ROOT / "RadioStat"
TOP5_DIR = OUT_ROOT / "top5"

REPO = r"C:\github\RADIO"
MODEL_VERSION = "c-radio_v4-h"

GOOD_DIST_M = 1500.0
TOP_PLOT_K = 5
TOP_STATS = (5, 10)


# ── Directory setup ────────────────────────────────────────────────────────────
def ensure_dirs() -> None:
    for d in [OUT_ROOT, TOP5_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(repo=REPO, version=MODEL_VERSION):
    """Load RADIO model from local repo, using cached weights."""
    source = "local" if os.path.exists(repo) else "github"
    model = torch.hub.load(
        repo, "radio_model", source=source, version=version,
        progress=True, skip_validation=True,
    )
    return model.cuda().eval()


# ── Metadata parsing ──────────────────────────────────────────────────────────
def parse_query_txt(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            p = [x.strip() for x in s.split(",")]
            if len(p) < 8:
                continue
            rows.append({
                "rel_path": p[0],
                "frame_id": int(float(p[1])),
                "utm_x": float(p[2]),
                "utm_y": float(p[3]),
                "heading": float(p[4]),
                "altitude": float(p[5]),
                "width": int(float(p[-2])),
                "height": int(float(p[-1])),
            })
    return pd.DataFrame(rows)


def parse_gallery_txt(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            p = [x.strip() for x in s.split(",")]
            if len(p) < 14:
                continue
            rows.append({
                "rel_path": p[0],
                "frame_id": int(float(p[1])),
                "utm_x": float(p[2]),
                "utm_y": float(p[3]),
                "heading": float(p[4]),
                "altitude": float(p[5]),
                "A": float(p[6]),
                "D": float(p[7]),
                "B": float(p[8]),
                "E": float(p[9]),
                "C": float(p[10]),
                "F": float(p[11]),
                "width": int(float(p[-2])),
                "height": int(float(p[-1])),
            })
    return pd.DataFrame(rows)


# ── Image loading & preprocessing ─────────────────────────────────────────────
def load_pil_image(rel_path: str) -> Image.Image:
    full_path = BASE_ROOT / rel_path
    return Image.open(full_path).convert("RGB")


def preprocess(model, pil_img: Image.Image) -> torch.Tensor:
    x = pil_to_tensor(pil_img).to(dtype=torch.float32, device="cuda")
    x.div_(255.0)
    x = x.unsqueeze(0)
    nearest_res = model.get_nearest_supported_resolution(*x.shape[-2:])
    x = F.interpolate(x, nearest_res, mode="bilinear", align_corners=False)
    return x


# ── Feature extraction ────────────────────────────────────────────────────────
@torch.no_grad()
def extract_features_from_df(model, df: pd.DataFrame, label: str = "") -> np.ndarray:
    """Extract normalized summary features for all images in df."""
    summaries = []
    n = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 50 == 0 or i == 0 or i == n - 1:
            print(f"  [{label}] {i+1}/{n}")
        img = load_pil_image(row["rel_path"])
        x = preprocess(model, img)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            summary, _ = model(x)
        summaries.append(F.normalize(summary, dim=-1).cpu())
    return torch.cat(summaries, dim=0).numpy()


# ── Similarity & distance ─────────────────────────────────────────────────────
def compute_similarity_matrix(query_feat: np.ndarray, gallery_feat: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix [Nq x Ng]."""
    q = torch.as_tensor(query_feat, dtype=torch.float32, device="cuda")
    g = torch.as_tensor(gallery_feat, dtype=torch.float32, device="cuda")
    q = F.normalize(q, dim=1)
    g = F.normalize(g, dim=1)
    sim = (q @ g.T).cpu().numpy()
    return sim


def pairwise_distance_matrix(query_df: pd.DataFrame, gallery_df: pd.DataFrame) -> np.ndarray:
    """UTM Euclidean distance [Nq x Ng] in meters."""
    qx = query_df["utm_x"].to_numpy(dtype=np.float64)[:, None]
    qy = query_df["utm_y"].to_numpy(dtype=np.float64)[:, None]
    gx = gallery_df["utm_x"].to_numpy(dtype=np.float64)[None, :]
    gy = gallery_df["utm_y"].to_numpy(dtype=np.float64)[None, :]
    return np.hypot(gx - qx, gy - qy)


# ── Top-K plotting ─────────────────────────────────────────────────────────────
def plot_topk_per_query(
    query_df: pd.DataFrame,
    gallery_df: pd.DataFrame,
    sim: np.ndarray,
    dist: np.ndarray,
    out_dir: Path,
    top_k: int = TOP_PLOT_K,
    hit_dist_m: float = GOOD_DIST_M,
) -> Dict[str, np.ndarray]:
    """Save a 1x(1+K) panel per query: query + top-K gallery images."""
    n_q = sim.shape[0]
    success_top5 = np.zeros(n_q, dtype=np.int32)
    success_top10 = np.zeros(n_q, dtype=np.int32)

    for qi in range(n_q):
        rank = np.argsort(-sim[qi])
        top5_idx = rank[:5]
        top10_idx = rank[:10]

        success_top5[qi] = int(np.any(dist[qi, top5_idx] <= hit_dist_m))
        success_top10[qi] = int(np.any(dist[qi, top10_idx] <= hit_dist_m))

        q_row = query_df.iloc[qi]
        q_img = np.asarray(load_pil_image(q_row["rel_path"]))

        fig, axes = plt.subplots(1, 1 + top_k, figsize=(4 * (1 + top_k), 4.5))

        # Query panel
        axes[0].imshow(q_img)
        axes[0].set_title(
            f"QUERY frame={int(q_row['frame_id'])}\n"
            f"({q_row['utm_x']:.0f}, {q_row['utm_y']:.0f})",
            fontweight="bold", fontsize=9,
        )
        axes[0].axis("off")
        for spine in axes[0].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("blue")
            spine.set_linewidth(3.0)

        # Gallery panels
        for k in range(top_k):
            gi = int(rank[k])
            g_row = gallery_df.iloc[gi]
            g_img = np.asarray(load_pil_image(g_row["rel_path"]))
            d = float(dist[qi, gi])
            s = float(sim[qi, gi])
            is_good = d <= hit_dist_m

            ax = axes[k + 1]
            ax.imshow(g_img)
            ax.axis("off")

            dist_str = f"{d:.0f}m" if d < 10000 else f"{d/1000:.1f}km"
            title_color = "green" if is_good else "red"
            ax.set_title(
                f"Top{k+1} frame={int(g_row['frame_id'])}\n"
                f"sim={s:.4f}  {dist_str}",
                color=title_color, fontsize=9,
            )
            border_color = "green" if is_good else "black"
            border_width = 4.0 if is_good else 1.5
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)

        fig.suptitle(f"Query idx={qi}  frame={int(q_row['frame_id'])}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"query_{qi:04d}.png", dpi=150)
        plt.close(fig)

    return {"success_top5": success_top5, "success_top10": success_top10}


# ── Similarity heatmap ─────────────────────────────────────────────────────────
def save_similarity_matrix_figure(sim: np.ndarray, out_file: Path, title: str) -> None:
    plt.figure(figsize=(12, 8))
    im = plt.imshow(sim, cmap="viridis", aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="cosine similarity")
    plt.xlabel("Gallery index")
    plt.ylabel("Query index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


# ── Statistics & histogram ─────────────────────────────────────────────────────
def save_stats_and_hist(
    query_df: pd.DataFrame,
    gallery_df: pd.DataFrame,
    sim: np.ndarray,
    dist: np.ndarray,
    success: Dict[str, np.ndarray],
    good_dist_m: float = GOOD_DIST_M,
    tag: str = "radio",
) -> None:
    n_q = sim.shape[0]
    top5_rate = float(success["success_top5"].mean()) if n_q else 0.0
    top10_rate = float(success["success_top10"].mean()) if n_q else 0.0

    summary_lines = [
        f"tag={tag}",
        f"model={MODEL_VERSION}",
        f"num_queries={n_q}",
        f"num_gallery={len(gallery_df)}",
        f"hit_threshold_m={good_dist_m}",
        f"top5_hits={int(success['success_top5'].sum())}",
        f"top5_rate={top5_rate:.4f} ({top5_rate*100:.1f}%)",
        f"top10_hits={int(success['success_top10'].sum())}",
        f"top10_rate={top10_rate:.4f} ({top10_rate*100:.1f}%)",
    ]
    summary_path = OUT_ROOT / f"summary_{tag}.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\n{'='*50}")
    print(f"Statistics ({tag}):")
    for line in summary_lines:
        print(f"  {line}")
    print(f"{'='*50}\n")

    # Per-query CSV
    dist_label = str(int(good_dist_m)) if good_dist_m == int(good_dist_m) else f"{good_dist_m}"
    per_query_df = pd.DataFrame({
        "query_idx": np.arange(n_q, dtype=np.int32),
        "frame_id": query_df["frame_id"].to_numpy(dtype=np.int32),
        f"hit_top5_lt{dist_label}m": success["success_top5"],
        f"hit_top10_lt{dist_label}m": success["success_top10"],
    })
    per_query_df.to_csv(OUT_ROOT / f"per_query_{tag}.csv", index=False)

    # Histogram of similarities
    good_mask = dist < good_dist_m
    good_scores = sim[good_mask]
    bad_scores = sim[~good_mask]

    plt.figure(figsize=(9, 6))
    if good_scores.size:
        plt.hist(good_scores, bins=60, alpha=0.65, label=f"good (<{good_dist_m/1000:.1f}km)", color="green")
    if bad_scores.size:
        plt.hist(bad_scores, bins=60, alpha=0.45, label=f"bad (>={good_dist_m/1000:.1f}km)", color="gray")
    plt.title(f"Similarity Histogram — {tag}")
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ROOT / f"hist_similarity_{tag}.png", dpi=180)
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ensure_dirs()

    # 1. Parse metadata
    print("Parsing metadata...")
    query_df = parse_query_txt(QUERY_TXT)
    gallery_df = parse_gallery_txt(GALLERY_TXT)
    print(f"  Queries : {len(query_df)}")
    print(f"  Gallery : {len(gallery_df)}")

    # 2. Load model (cached weights, no re-download)
    print("Loading RADIO model...")
    model = load_model()

    # 3. Extract features
    print("Extracting query features (IR)...")
    query_feat = extract_features_from_df(model, query_df, label="query")
    print(f"  Query features shape: {query_feat.shape}")

    print("Extracting gallery features (VIS)...")
    gallery_feat = extract_features_from_df(model, gallery_df, label="gallery")
    print(f"  Gallery features shape: {gallery_feat.shape}")

    # Save features for later reuse
    np.save(OUT_ROOT / "radio_query_features.npy", query_feat)
    np.save(OUT_ROOT / "radio_gallery_features.npy", gallery_feat)

    # 4. Compute similarity and distance
    print("Computing similarity matrix...")
    sim = compute_similarity_matrix(query_feat, gallery_feat)
    np.save(OUT_ROOT / "similarity_matrix.npy", sim)

    print("Computing distance matrix...")
    dist = pairwise_distance_matrix(query_df, gallery_df)

    # 5. Save similarity heatmap
    save_similarity_matrix_figure(sim, OUT_ROOT / "similarity_matrix.png", "IR Query vs VIS Gallery — RADIO")

    # 6. Plot top-5 panels per query
    print("Plotting top-5 per query...")
    success = plot_topk_per_query(
        query_df=query_df,
        gallery_df=gallery_df,
        sim=sim,
        dist=dist,
        out_dir=TOP5_DIR,
        top_k=TOP_PLOT_K,
        hit_dist_m=GOOD_DIST_M,
    )

    # 7. Save statistics and histogram
    save_stats_and_hist(
        query_df=query_df,
        gallery_df=gallery_df,
        sim=sim,
        dist=dist,
        success=success,
        good_dist_m=GOOD_DIST_M,
        tag="radio_ir_vs_vis",
    )

    print("Done! Results saved to:", OUT_ROOT)


if __name__ == "__main__":
    main()
