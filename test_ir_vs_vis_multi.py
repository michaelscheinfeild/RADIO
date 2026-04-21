"""
Multi-dataset IR vs VIS retrieval evaluation using precomputed RADIO features.

- Reads precomputed features from each dataset's RadioStat folder:
  - radio_query_features.npy
  - radio_gallery_features.npy
- Combines all query features into one matrix and all gallery features into one matrix
- Combines metadata from all datasets to preserve GPS/image mapping
- Computes cosine similarity and GPS distance statistics (same style as test_ir_vs_vis.py)
- Saves top-5 panels (regular + filtered within 5km), histograms, bar charts, and CSV/TXT stats


Reads both dataset roots:
D:/OneDrive - Israel Aerospace Industries/Databases/OrthoCropAshdodHuge
D:/OneDrive - Israel Aerospace Industries/Databases/OrthoCropWingateHuge
Loads precomputed features from each RadioStat folder:
radio_query_features.npy
radio_gallery_features.npy
Parses metadata from each dataset:
data/ortho/test_query.txt
data/ortho/test_gallery.txt
Concatenates all query features into one large matrix and all gallery features into one large matrix.
Computes:
Combined cosine similarity matrix
Combined GPS distance matrix
Saves full stats/plots in the same style as test_ir_vs_vis:
similarity heatmap
summary txt
per-query csv
similarity histogram
top-5 and top-10 correctness plots
Adds the extra requirement:
Top-5 plots for filtered candidates within 5km in a dedicated folder top5_filtered_5km.
Separate filtered-5km stats/plots.

"""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.nn import functional as F


# Configuration
DATASET_ROOTS: Sequence[Path] = (
	Path(r"D:\OneDrive - Israel Aerospace Industries\Databases\OrthoCropAshdodHuge"),
	Path(r"D:\OneDrive - Israel Aerospace Industries\Databases\OrthoCropWingateHuge"),
)

OUT_ROOT = Path(r"D:\OneDrive - Israel Aerospace Industries\Databases\RadioStatCombined")
TOP5_DIR = OUT_ROOT / "top5"
TOP5_FILTERED_DIR = OUT_ROOT / "top5_filtered_5km"

GOOD_DIST_M = 1500.0
FILTER_DIST_M = 5000.0
TOP_PLOT_K = 5


def ensure_dirs() -> None:
	for d in (OUT_ROOT, TOP5_DIR, TOP5_FILTERED_DIR):
		d.mkdir(parents=True, exist_ok=True)


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


def _resolve_image_path(base_root: Path, rel_path: str) -> Path:
	return base_root / rel_path


def load_combined_inputs(dataset_roots: Sequence[Path]) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
	query_meta_all: List[pd.DataFrame] = []
	gallery_meta_all: List[pd.DataFrame] = []
	query_feat_all: List[np.ndarray] = []
	gallery_feat_all: List[np.ndarray] = []

	for ds_idx, root in enumerate(dataset_roots):
		data_root = root / "data" / "ortho"
		query_txt = data_root / "test_query.txt"
		gallery_txt = data_root / "test_gallery.txt"
		stat_root = root / "RadioStat"
		query_npy = stat_root / "radio_query_features.npy"
		gallery_npy = stat_root / "radio_gallery_features.npy"

		if not query_txt.exists() or not gallery_txt.exists():
			raise FileNotFoundError(f"Missing metadata files under: {data_root}")
		if not query_npy.exists() or not gallery_npy.exists():
			raise FileNotFoundError(f"Missing feature npy files under: {stat_root}")

		print(f"Loading dataset #{ds_idx + 1}: {root}")

		q_df = parse_query_txt(query_txt)
		g_df = parse_gallery_txt(gallery_txt)
		q_feat = np.load(query_npy)
		g_feat = np.load(gallery_npy)

		if len(q_df) != q_feat.shape[0]:
			raise ValueError(
				f"Query mismatch in {root.name}: rows={len(q_df)} vs feat={q_feat.shape[0]}"
			)
		if len(g_df) != g_feat.shape[0]:
			raise ValueError(
				f"Gallery mismatch in {root.name}: rows={len(g_df)} vs feat={g_feat.shape[0]}"
			)

		q_df = q_df.copy()
		g_df = g_df.copy()
		q_df["dataset_name"] = root.name
		g_df["dataset_name"] = root.name
		q_df["dataset_root"] = str(root)
		g_df["dataset_root"] = str(root)

		query_meta_all.append(q_df)
		gallery_meta_all.append(g_df)
		query_feat_all.append(q_feat)
		gallery_feat_all.append(g_feat)

		print(f"  Queries : {len(q_df)} | feature shape: {q_feat.shape}")
		print(f"  Gallery : {len(g_df)} | feature shape: {g_feat.shape}")

	query_df = pd.concat(query_meta_all, axis=0, ignore_index=True)
	gallery_df = pd.concat(gallery_meta_all, axis=0, ignore_index=True)
	query_feat = np.concatenate(query_feat_all, axis=0)
	gallery_feat = np.concatenate(gallery_feat_all, axis=0)

	if query_feat.shape[1] != gallery_feat.shape[1]:
		raise ValueError(
			f"Feature dimensions do not match: query={query_feat.shape[1]} gallery={gallery_feat.shape[1]}"
		)

	return query_df, gallery_df, query_feat, gallery_feat


def compute_similarity_matrix(query_feat: np.ndarray, gallery_feat: np.ndarray) -> np.ndarray:
	q = torch.as_tensor(query_feat, dtype=torch.float32, device="cuda")
	g = torch.as_tensor(gallery_feat, dtype=torch.float32, device="cuda")
	q = F.normalize(q, dim=1)
	g = F.normalize(g, dim=1)
	sim = (q @ g.T).cpu().numpy()
	return sim


def pairwise_distance_matrix(query_df: pd.DataFrame, gallery_df: pd.DataFrame) -> np.ndarray:
	qx = query_df["utm_x"].to_numpy(dtype=np.float64)[:, None]
	qy = query_df["utm_y"].to_numpy(dtype=np.float64)[:, None]
	gx = gallery_df["utm_x"].to_numpy(dtype=np.float64)[None, :]
	gy = gallery_df["utm_y"].to_numpy(dtype=np.float64)[None, :]
	return np.hypot(gx - qx, gy - qy)


def _load_image_from_row(row: pd.Series) -> np.ndarray:
	img_path = _resolve_image_path(Path(row["dataset_root"]), str(row["rel_path"]))
	return np.asarray(Image.open(img_path).convert("RGB"))


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


def plot_topk_per_query(
	query_df: pd.DataFrame,
	gallery_df: pd.DataFrame,
	sim: np.ndarray,
	dist: np.ndarray,
	out_dir: Path,
	top_k: int,
	hit_dist_m: float,
	caption_suffix: str,
) -> Dict[str, np.ndarray]:
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
		q_img = _load_image_from_row(q_row)
		fig, axes = plt.subplots(1, 1 + top_k, figsize=(4 * (1 + top_k), 4.8))

		axes[0].imshow(q_img)
		axes[0].set_title(
			f"QUERY {q_row['dataset_name']} frame={int(q_row['frame_id'])}\n"
			f"({q_row['utm_x']:.0f}, {q_row['utm_y']:.0f})",
			fontweight="bold",
			fontsize=9,
		)
		axes[0].axis("off")
		for spine in axes[0].spines.values():
			spine.set_visible(True)
			spine.set_edgecolor("blue")
			spine.set_linewidth(3.0)

		for k in range(top_k):
			gi = int(rank[k])
			g_row = gallery_df.iloc[gi]
			g_img = _load_image_from_row(g_row)
			d = float(dist[qi, gi])
			s = float(sim[qi, gi])
			is_good = d <= hit_dist_m

			ax = axes[k + 1]
			ax.imshow(g_img)
			ax.axis("off")

			dist_str = f"{d:.0f}m" if d < 10000 else f"{d/1000:.1f}km"
			title_color = "green" if is_good else "red"
			ax.set_title(
				f"Top{k + 1} {g_row['dataset_name']} frame={int(g_row['frame_id'])}\n"
				f"sim={s:.4f}  {dist_str}",
				color=title_color,
				fontsize=9,
			)

			border_color = "green" if is_good else "black"
			border_width = 4.0 if is_good else 1.5
			for spine in ax.spines.values():
				spine.set_visible(True)
				spine.set_edgecolor(border_color)
				spine.set_linewidth(border_width)

		fig.suptitle(
			f"Query idx={qi} {q_row['dataset_name']} frame={int(q_row['frame_id'])} | {caption_suffix}",
			fontsize=11,
		)
		fig.tight_layout()
		fig.savefig(out_dir / f"query_{qi:05d}.png", dpi=150)
		plt.close(fig)

	return {"success_top5": success_top5, "success_top10": success_top10}


def _plot_topk_correct_per_query(
	query_df: pd.DataFrame,
	sim: np.ndarray,
	dist: np.ndarray,
	good_dist_m: float,
	tag: str,
	top_k: int,
) -> None:
	n_q = sim.shape[0]
	correct_counts = np.zeros(n_q, dtype=np.int32)
	for qi in range(n_q):
		rank = np.argsort(-sim[qi])[:top_k]
		correct_counts[qi] = int(np.sum(dist[qi, rank] <= good_dist_m))

	labels = [
		f"{query_df.iloc[i]['dataset_name']}:{int(query_df.iloc[i]['frame_id'])}"
		for i in range(n_q)
	]
	x = np.arange(n_q)

	colors = ["green" if c > 0 else "red" for c in correct_counts]
	fig, ax = plt.subplots(figsize=(max(8, n_q * 0.4), 5))
	ax.bar(x, correct_counts, color=colors, edgecolor="black", linewidth=0.5)
	ax.set_xticks(x)
	ax.set_xticklabels(labels, rotation=90, fontsize=7)
	ax.set_xlabel("Query (dataset:frame_id)")
	ax.set_ylabel(f"# correct in top-{top_k}")
	ax.set_ylim(0, top_k + 0.5)
	ax.set_yticks(range(top_k + 1))
	mean_correct = correct_counts.mean()
	n_with_hit = int(np.sum(correct_counts > 0))
	ax.set_title(
		f"Correct matches in top-{top_k} per query - {tag}\n"
		f"mean={mean_correct:.2f}  queries with >=1 hit: {n_with_hit}/{n_q} ({n_with_hit/n_q*100:.1f}%)"
	)
	ax.axhline(mean_correct, color="blue", linestyle="--", linewidth=1, label=f"mean={mean_correct:.2f}")
	ax.legend()
	fig.tight_layout()
	fig.savefig(OUT_ROOT / f"correct_top{top_k}_per_query_{tag}.png", dpi=180)
	plt.close(fig)

	fig2, ax2 = plt.subplots(figsize=(7, 5))
	bins = np.arange(top_k + 2) - 0.5
	ax2.hist(correct_counts, bins=bins, color="steelblue", edgecolor="black", rwidth=0.8)
	ax2.set_xticks(range(top_k + 1))
	ax2.set_xlabel(f"# correct in top-{top_k}")
	ax2.set_ylabel("# queries")
	ax2.set_title(f"Distribution of correct matches in top-{top_k} - {tag}")
	fig2.tight_layout()
	fig2.savefig(OUT_ROOT / f"correct_top{top_k}_distribution_{tag}.png", dpi=180)
	plt.close(fig2)


def save_stats_and_hist(
	query_df: pd.DataFrame,
	gallery_df: pd.DataFrame,
	sim: np.ndarray,
	dist: np.ndarray,
	success: Dict[str, np.ndarray],
	good_dist_m: float,
	tag: str,
) -> None:
	n_q = sim.shape[0]
	top5_rate = float(success["success_top5"].mean()) if n_q else 0.0
	top10_rate = float(success["success_top10"].mean()) if n_q else 0.0

	summary_lines = [
		f"tag={tag}",
		f"num_queries={n_q}",
		f"num_gallery={len(gallery_df)}",
		f"hit_threshold_m={good_dist_m}",
		f"top5_hits={int(success['success_top5'].sum())}",
		f"top5_rate={top5_rate:.4f} ({top5_rate * 100:.1f}%)",
		f"top10_hits={int(success['success_top10'].sum())}",
		f"top10_rate={top10_rate:.4f} ({top10_rate * 100:.1f}%)",
	]
	summary_path = OUT_ROOT / f"summary_{tag}.txt"
	summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

	print(f"\n{'=' * 50}")
	print(f"Statistics ({tag}):")
	for line in summary_lines:
		print(f"  {line}")
	print(f"{'=' * 50}\n")

	per_query_df = pd.DataFrame({
		"query_idx": np.arange(n_q, dtype=np.int32),
		"dataset_name": query_df["dataset_name"].to_numpy(),
		"frame_id": query_df["frame_id"].to_numpy(dtype=np.int32),
		"hit_top5_lt1500m": success["success_top5"],
		"hit_top10_lt1500m": success["success_top10"],
	})
	per_query_df.to_csv(OUT_ROOT / f"per_query_{tag}.csv", index=False)

	good_mask = dist < good_dist_m
	good_scores = sim[good_mask]
	bad_scores = sim[~good_mask]

	plt.figure(figsize=(9, 6))
	if good_scores.size:
		plt.hist(good_scores, bins=60, alpha=0.65, label=f"good (<{good_dist_m / 1000:.1f}km)", color="green")
	if bad_scores.size:
		plt.hist(bad_scores, bins=60, alpha=0.45, label=f"bad (>={good_dist_m / 1000:.1f}km)", color="gray")
	plt.title(f"Similarity Histogram - {tag}")
	plt.xlabel("cosine similarity")
	plt.ylabel("count")
	plt.legend()
	plt.tight_layout()
	plt.savefig(OUT_ROOT / f"hist_similarity_{tag}.png", dpi=180)
	plt.close()

	_plot_topk_correct_per_query(query_df, sim, dist, good_dist_m, tag, top_k=5)
	_plot_topk_correct_per_query(query_df, sim, dist, good_dist_m, tag, top_k=10)


def filter_similarity_by_distance(sim: np.ndarray, dist: np.ndarray, max_dist_m: float) -> np.ndarray:
	filtered = sim.copy()
	filtered[dist > max_dist_m] = -1.0
	return filtered


def compute_filtered_success(sim_filtered: np.ndarray, dist: np.ndarray, hit_dist_m: float) -> Dict[str, np.ndarray]:
	n_q = sim_filtered.shape[0]
	success_top5 = np.zeros(n_q, dtype=np.int32)
	success_top10 = np.zeros(n_q, dtype=np.int32)
	for qi in range(n_q):
		rank = np.argsort(-sim_filtered[qi])
		top5_idx = rank[:5]
		top10_idx = rank[:10]
		success_top5[qi] = int(np.any(dist[qi, top5_idx] <= hit_dist_m))
		success_top10[qi] = int(np.any(dist[qi, top10_idx] <= hit_dist_m))
	return {"success_top5": success_top5, "success_top10": success_top10}


def main() -> None:
	ensure_dirs()

	print("Loading metadata + precomputed npy features...")
	query_df, gallery_df, query_feat, gallery_feat = load_combined_inputs(DATASET_ROOTS)
	print(f"Combined queries: {len(query_df)}")
	print(f"Combined gallery: {len(gallery_df)}")
	print(f"Combined query features shape: {query_feat.shape}")
	print(f"Combined gallery features shape: {gallery_feat.shape}")

	np.save(OUT_ROOT / "radio_query_features_combined.npy", query_feat)
	np.save(OUT_ROOT / "radio_gallery_features_combined.npy", gallery_feat)

	print("Computing similarity matrix...")
	sim = compute_similarity_matrix(query_feat, gallery_feat)
	np.save(OUT_ROOT / "similarity_matrix_combined.npy", sim)

	print("Computing GPS distance matrix...")
	dist = pairwise_distance_matrix(query_df, gallery_df)
	np.save(OUT_ROOT / "distance_matrix_combined.npy", dist)

	save_similarity_matrix_figure(
		sim,
		OUT_ROOT / "similarity_matrix_combined.png",
		"Combined IR Query vs VIS Gallery - RADIO",
	)

	print("Plotting top-5 per query (all gallery)...")
	success = plot_topk_per_query(
		query_df=query_df,
		gallery_df=gallery_df,
		sim=sim,
		dist=dist,
		out_dir=TOP5_DIR,
		top_k=TOP_PLOT_K,
		hit_dist_m=GOOD_DIST_M,
		caption_suffix="all gallery",
	)
	save_stats_and_hist(
		query_df=query_df,
		gallery_df=gallery_df,
		sim=sim,
		dist=dist,
		success=success,
		good_dist_m=GOOD_DIST_M,
		tag="radio_ir_vs_vis_combined",
	)

	print(f"Applying distance filter <= {FILTER_DIST_M / 1000:.0f}km...")
	sim_filtered = filter_similarity_by_distance(sim, dist, FILTER_DIST_M)
	np.save(OUT_ROOT / "similarity_matrix_combined_filtered_5km.npy", sim_filtered)

	print("Plotting top-5 per query (filtered 5km)...")
	success_filtered = plot_topk_per_query(
		query_df=query_df,
		gallery_df=gallery_df,
		sim=sim_filtered,
		dist=dist,
		out_dir=TOP5_FILTERED_DIR,
		top_k=TOP_PLOT_K,
		hit_dist_m=GOOD_DIST_M,
		caption_suffix="gallery candidates filtered <=5km",
	)

	success_filtered_stats = compute_filtered_success(sim_filtered, dist, hit_dist_m=GOOD_DIST_M)
	save_stats_and_hist(
		query_df=query_df,
		gallery_df=gallery_df,
		sim=sim_filtered,
		dist=dist,
		success=success_filtered_stats,
		good_dist_m=GOOD_DIST_M,
		tag="radio_ir_vs_vis_combined_filtered_5km",
	)

	print("Done. Results saved to:")
	print(f"  {OUT_ROOT}")
	print(f"  top5 all: {TOP5_DIR}")
	print(f"  top5 filtered: {TOP5_FILTERED_DIR}")


if __name__ == "__main__":
	main()
