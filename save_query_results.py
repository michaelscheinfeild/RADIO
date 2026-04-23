"""
save_query_results.py — Batch query-to-gallery retrieval with RADIO

Given a folder of query images and a gallery (features + metadata),
extract features for each query, compute cosine similarity to the gallery,
and save per-query results (top-10 matches, metadata, etc.) as described.

Usage:
    python save_query_results.py --query_dir <query_folder> --session_name <session_id> --gallery_dir <gallery_data_dir> --crop_folder <crop_folder>

Example:
    python save_query_results.py \
        --query_dir "C:/Users/OPER/OneDrive - Israel Aerospace Industries/Frames_ofek/session_2026-01-04_14-47-20_309/frames" \
        --session_name 309 \
        --gallery_dir "D:/OrthoIsr/data/crop1000" \
        --crop_folder "D:/OrthoIsr/data/crop1000"
"""
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

# --- Helper functions (adapted from RADIO scripts) ---
def load_model(repo, version):
    source = "local" if os.path.exists(repo) else "github"
    model = torch.hub.load(
        repo, "radio_model", source=source, version=version,
        progress=True, skip_validation=True,
    )
    return model.cuda().eval()

def preprocess(model, pil_img):
    x = pil_to_tensor(pil_img).to(dtype=torch.float32, device="cuda")
    x.div_(255.0)
    x = x.unsqueeze(0)
    nearest_res = model.get_nearest_supported_resolution(*x.shape[-2:])
    x = F.interpolate(x, nearest_res, mode="bilinear", align_corners=False)
    return x

def extract_features(model, img_paths, batch_size=1):
    features = []
    batch = []
    n_imgs = len(img_paths)
    last_report = -1
    for idx, img_path in enumerate(img_paths):
        pil_img = Image.open(img_path).convert("RGB")
        x = preprocess(model, pil_img)
        batch.append(x)
        percent = int(100 * (idx + 1) / n_imgs) if n_imgs > 0 else 100
        if percent // 10 != last_report and percent % 10 == 0:
            print(f"[extract_features] {percent}% ({idx + 1}/{n_imgs}) images processed...")
            last_report = percent // 10
        if len(batch) == batch_size:
            x_batch = torch.cat(batch, dim=0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                summary, _ = model(x_batch)
            features.append(F.normalize(summary, dim=-1).detach().cpu().numpy())
            batch = []
    if batch:
        x_batch = torch.cat(batch, dim=0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            summary, _ = model(x_batch)
        features.append(F.normalize(summary, dim=-1).detach().cpu().numpy())
    return np.concatenate(features, axis=0)

def compute_similarity_matrix(query_feat, gallery_feat):
    q = torch.as_tensor(query_feat, dtype=torch.float32, device="cuda")
    g = torch.as_tensor(gallery_feat, dtype=torch.float32, device="cuda")
    q = F.normalize(q, dim=1)
    g = F.normalize(g, dim=1)
    sim = (q @ g.T).cpu().numpy()
    return sim

def parse_gallery_txt(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            p = [x.strip() for x in s.split(",")]
            if len(p) < 8:
                continue
            rows.append({
                "rel_path": p[0],
                "crop_folder": p[1],
                "A": float(p[2]),
                "D": float(p[3]),
                "B": float(p[4]),
                "E": float(p[5]),
                "C": float(p[6]),
                "F": float(p[7]),
            })
    return pd.DataFrame(rows)

def load_query_metadata(csv_path):
    df = pd.read_csv(csv_path)
    # Expect columns: image_name, utm_x, utm_y, heading, height, ...
    return df


def run_query_results(
    query_dir,
    session_name,
    gallery_dir,
    crop_folder,
    repo=r'C:/github/RADIO',
    model_version='c-radio_v4-h',
    query_step=1,
    query_start=-1,
    query_end=-1,
):
    query_dir = Path(query_dir)
    gallery_dir = Path(gallery_dir)
    crop_folder = Path(crop_folder)

    # --- Load gallery features and metadata ---
    gallery_npy = next(gallery_dir.glob('radio*.npy'))
    gallery_txt = next(gallery_dir.glob('radio*.txt'))
    gallery_features = np.load(gallery_npy)
    gallery_df = parse_gallery_txt(gallery_txt)

    # --- Load query images and metadata ---
    query_imgs = sorted(list(query_dir.glob('*.tif')))
    # Select range if specified
    if query_start == -1 and query_end == -1:
        query_imgs = query_imgs[::query_step]
    else:
        # Clamp indices to valid range
        n = len(query_imgs)
        s = max(0, query_start) if query_start >= 0 else 0
        e = min(n, query_end) if query_end >= 0 else n
        query_imgs = query_imgs[s:e:query_step]
    # Try to find the GPS/heading/height CSV
    session_csv = list(query_dir.parent.glob('frame_gps_synchronized.csv'))
    if session_csv:
        query_meta_df = load_query_metadata(session_csv[0])
    else:
        query_meta_df = pd.DataFrame()

    # --- Load model ---
    model = load_model(repo, model_version)

    # --- Extract features for queries ---
    batch_size = 8 # Adjust based on GPU memory
    query_features = extract_features(model, query_imgs, batch_size=batch_size)

    # --- Compute similarity ---
    sim = compute_similarity_matrix(query_features, gallery_features)

    # --- Save all query features ---
    out_npy = crop_folder / f'query_features_{session_name}.npy'
    np.save(out_npy, query_features)

    # --- Save per-query results ---
    for i, img_path in enumerate(query_imgs):
        img_name = img_path.name
        # Normalize for metadata lookup: strip extension and leading zeros
        img_stem = os.path.splitext(img_name)[0].lstrip('0')
        # Find matching metadata row by frame_name (e.g., 'frame1234' matches '1234.tif')
        meta_row = None
        if not query_meta_df.empty:
            # Try to match frame_name ending with the numeric part of img_stem
            matches = query_meta_df[query_meta_df['frame_name'].apply(lambda x: x.lstrip('frame').lstrip('0') == img_stem)]
            if not matches.empty:
                meta_row = matches
        top10_idx = np.argsort(-sim[i])[:10]
        top5_idx = top10_idx[:5]
        # Compose result text
        if meta_row is not None and not meta_row.empty:
            utm_x = meta_row.iloc[0].get('utm_x', '')
            utm_y = meta_row.iloc[0].get('utm_y', '')
            heading = meta_row.iloc[0].get('heading', '')
            height = meta_row.iloc[0].get('altitude', '')
        else:
            utm_x = utm_y = heading = height = ''
        result_txt = f"session: {session_name}\nimage: {img_name}\nutm_x: {utm_x}\nutm_y: {utm_y}\nheading: {heading}\nheight: {height}\ntop5_gallery_idx: {top5_idx.tolist()}\ntop10_gallery_idx: {top10_idx.tolist()}\n"
        img_stem = os.path.splitext(img_name)[0]
        out_txt = crop_folder / f'query_result_{session_name}_{img_stem}.txt'
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write(result_txt)

    # --- Save a summary txt for all queries ---
    summary_txt = crop_folder / f'query_summary_{session_name}.txt'
    with open(summary_txt, 'w', encoding='utf-8') as f:
        for i, img_path in enumerate(query_imgs):
            img_name = img_path.name
            top10_idx = np.argsort(-sim[i])[:10]
            top5_idx = top10_idx[:5]
            img_stem = os.path.splitext(img_name)[0].lstrip('0')
            meta_row = None
            if not query_meta_df.empty:
                matches = query_meta_df[query_meta_df['frame_name'].apply(lambda x: x.lstrip('frame').lstrip('0') == img_stem)]
                if not matches.empty:
                    meta_row = matches
            if meta_row is not None and not meta_row.empty:
                utm_x = meta_row.iloc[0].get('utm_x', '')
                utm_y = meta_row.iloc[0].get('utm_y', '')
                heading = meta_row.iloc[0].get('heading', '')
                height = meta_row.iloc[0].get('altitude', '')
            else:
                utm_x = utm_y = heading = height = ''
            f.write(f"{img_name},{utm_x},{utm_y},{heading},{height},{top5_idx.tolist()},{top10_idx.tolist()}\n")

    print(f"Done. Results saved in {crop_folder}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_dir', required=True, help='Folder with query images')
    parser.add_argument('--session_name', required=True, help='Session name/id (e.g., 309)')
    parser.add_argument('--gallery_dir', required=True, help='Gallery data folder (with .npy and .txt)')
    parser.add_argument('--crop_folder', required=True, help='Crop folder to save results')
    parser.add_argument('--repo', default=r'C:/github/RADIO', help='RADIO repo path')
    parser.add_argument('--model_version', default='c-radio_v4-h', help='RADIO model version')
    parser.add_argument('--query_step', type=int, default=1, help='Step size for selecting query images (default: 1, use 60 for every 60th image)')
    parser.add_argument('--query_start', type=int, default=-1, help='Start index for query images (default: -1, meaning start at 0)')
    parser.add_argument('--query_end', type=int, default=-1, help='End index for query images (default: -1, meaning end at last image)')
    args = parser.parse_args()
    run_query_results(
        query_dir=args.query_dir,
        session_name=args.session_name,
        gallery_dir=args.gallery_dir,
        crop_folder=args.crop_folder,
        repo=args.repo,
        model_version=args.model_version,
        query_step=args.query_step,
        query_start=args.query_start,
        query_end=args.query_end,
    )

if __name__ == "__main__":
    main()
