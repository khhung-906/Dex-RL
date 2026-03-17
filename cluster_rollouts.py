#!/usr/bin/env python3
"""
K-means clustering of rollout trajectories.

Loads rollouts from rollouts.hdf5, builds trajectory-level features,
runs k-means, and visualizes clusters (via t-SNE projection).

Usage:
    python cluster_rollouts.py --data_dir data/collected_rollouts/sr_75a7a_2
    python cluster_rollouts.py --data_dir data/collected_rollouts/sr_75a7a_2 -k 5
    python cluster_rollouts.py --h5_path path/to/rollouts.hdf5 -k 10 --output clusters.png
"""

import argparse
import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


def load_rollouts(h5_path: str):
    """Load all rollouts from rollouts.hdf5 into a list of dicts."""
    rollouts = []
    with h5py.File(h5_path, "r") as f:
        if "rollouts/successful" not in f:
            raise KeyError(f"No 'rollouts/successful' group in {h5_path}")
        s_grp = f["rollouts/successful"]
        for name in sorted(s_grp.keys()):
            grp = s_grp[name]
            rollout = {k: np.array(grp[k][:]) for k in grp.keys()}
            rollout["_name"] = name
            rollouts.append(rollout)
    return rollouts


def build_feature_matrix(rollouts, feature_keys, unit="trajectory"):
    """Build (N, D) feature matrix. unit='trajectory' -> one row per rollout (mean over time)."""
    all_rows = []
    for r in rollouts:
        parts = []
        for fk in feature_keys:
            if fk in r:
                parts.append(r[fk])
        if not parts:
            continue
        T = min(p.shape[0] for p in parts)
        vec = np.concatenate([p[:T].mean(axis=0).flatten() for p in parts])
        all_rows.append(vec)
    if not all_rows:
        raise ValueError("No valid feature rows extracted from rollouts")
    return np.stack(all_rows)


def main():
    parser = argparse.ArgumentParser(description="K-means clustering of rollout trajectories")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/collected_rollouts/sr_75a7a_2",
        help="Directory containing rollouts.hdf5",
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default=None,
        help="Direct path to rollouts.hdf5 (overrides data_dir)",
    )
    parser.add_argument(
        "-k", "--n_clusters",
        type=int,
        default=40,
        help="Number of clusters (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output figure path (default: <data_dir>/clusters_k{N}.png)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature keys (default: auto-detect)",
    )
    parser.add_argument(
        "--max_rollouts",
        type=int,
        default=None,
        help="Max rollouts to load (for quick test)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for k-means and t-SNE",
    )
    parser.add_argument(
        "--save_labels",
        type=str,
        default=None,
        help="Save cluster labels to .npy file",
    )
    parser.add_argument(
        "--save_results",
        type=str,
        default=None,
        help="Save cluster stats (mean reward, indices sorted by reward) to JSON file",
    )
    args = parser.parse_args()

    if args.h5_path:
        h5_path = args.h5_path
    else:
        for fname in ["rollouts.hdf5", "rollout.hdf5"]:
            candidate = os.path.join(args.data_dir, fname)
            if os.path.isfile(candidate):
                h5_path = candidate
                break
        else:
            h5_path = os.path.join(args.data_dir, "rollouts.hdf5")

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f"rollouts.hdf5 / rollout.hdf5 not found at {args.data_dir}. "
            "Run rollout collection first or provide --h5_path."
        )

    print(f"Loading rollouts from {h5_path}")
    rollouts = load_rollouts(h5_path)
    print(f"Loaded {len(rollouts)} rollouts")

    if args.max_rollouts is not None:
        rollouts = rollouts[: args.max_rollouts]
        print(f"Using first {len(rollouts)} rollouts")

    all_keys = set()
    for r in rollouts:
        all_keys.update(k for k in r.keys() if not k.startswith("_"))
    if args.features:
        feature_keys = [k.strip() for k in args.features.split(",") if k.strip()]
        missing = [k for k in feature_keys if k not in all_keys]
        if missing:
            raise ValueError(f"Unknown feature keys: {missing}. Available: {sorted(all_keys)}")
    else:
        preferred = [
            "state_rh", 
            "state_manip_obj_rh",
            "q_rh", "dq_rh",
        ]
        feature_keys = [k for k in preferred if k in all_keys] or list(all_keys - {"_name"})
    print(f"Using feature keys: {feature_keys}")

    X = build_feature_matrix(rollouts, feature_keys, unit="trajectory")
    print(f"Feature matrix shape: {X.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Running k-means with k={args.n_clusters}...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    if args.save_labels:
        np.save(args.save_labels, labels)
        print(f"Saved cluster labels to {args.save_labels}")

    # Compute rewards per rollout
    rewards = np.array([r["reward"][:].sum() if "reward" in r else 0.0 for r in rollouts])

    # Mean reward, indices (sorted high to low), and within-cluster similarity per cluster
    rollout_names = [r["_name"] for r in rollouts]

    cluster_mean_rewards = []
    cluster_indices = []
    cluster_names = []
    cluster_similarities = []
    for i in range(args.n_clusters):
        mask = labels == i
        indices = np.where(mask)[0]
        cluster_rewards = rewards[indices]
        # Sort indices by reward descending (high to low)
        order = np.argsort(-cluster_rewards)
        sorted_indices = indices[order].tolist()
        sorted_names = [rollout_names[j] for j in sorted_indices]
        mean_rew = float(cluster_rewards.mean()) if len(cluster_rewards) > 0 else 0.0
        cluster_mean_rewards.append(mean_rew)
        cluster_indices.append(sorted_indices)
        cluster_names.append(sorted_names)

        # Within-cluster similarity: mean pairwise cosine similarity (higher = more similar)
        if len(indices) <= 1:
            sim = 1.0
        else:
            pts = X_scaled[indices]
            sim_mat = cosine_similarity(pts)
            # Exclude diagonal, take mean of upper triangle
            n = len(indices)
            triu_sum = sim_mat[np.triu_indices(n, k=1)].sum()
            n_pairs = n * (n - 1) // 2
            sim = float(triu_sum / n_pairs)
        cluster_similarities.append(sim)

    # Sort clusters by mean reward (high to low)
    cluster_order = np.argsort(-np.array(cluster_mean_rewards))

    # Output format: "1": {idx, mean_reward, similarity}, "2": {...}, ...
    result = {
        "n_clusters": args.n_clusters,
        "description": "Clusters sorted by mean reward (high to low). idx/names sorted by reward (high to low). names = HDF5 group keys. similarity = mean pairwise cosine sim within cluster.",
    }
    for rank, orig_i in enumerate(cluster_order, start=1):
        result[str(rank)] = {
            "idx": cluster_indices[orig_i],
            "names": cluster_names[orig_i],
            "mean_reward": round(cluster_mean_rewards[orig_i], 4),
            "similarity": round(cluster_similarities[orig_i], 4),
        }

    out_json = args.save_results or os.path.join(
        os.path.dirname(h5_path),
        f"clusters_k{args.n_clusters}_results.json",
    )
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved cluster results to {out_json}")

    # Print mean reward and similarity per cluster
    print("\nClusters (sorted by mean reward, high to low):")
    for rank, orig_i in enumerate(cluster_order, start=1):
        print(f"  {rank}: mean_reward={cluster_mean_rewards[orig_i]:.2f}, similarity={cluster_similarities[orig_i]:.4f}, size={len(cluster_indices[orig_i])}")

    # t-SNE for visualization
    print("Running t-SNE for visualization...")
    perplexity = min(30, X.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=args.random_state)
    X_2d = tsne.fit_transform(X_scaled)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    scatter = ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7,
        s=20,
    )
    ax.set_title(f"K-means clusters (k={args.n_clusters})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.colorbar(scatter, ax=ax, label="cluster")
    ax.set_aspect("equal")
    plt.tight_layout()

    out_path = args.output or os.path.join(
        os.path.dirname(h5_path),
        f"clusters_k{args.n_clusters}.png",
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {out_path}")

    # Print cluster sizes
    for i in range(args.n_clusters):
        count = (labels == i).sum()
        print(f"  Cluster {i}: {count} trajectories ({100 * count / len(labels):.1f}%)")


if __name__ == "__main__":
    main()
