#!/usr/bin/env python3
"""
Load rollout.hdf5 from collected rollouts and visualize with t-SNE.

Usage:
    python analyze_data.py
    python analyze_data.py --data_dir data/collected_rollouts/sr_75a7a_2
    python analyze_data.py --h5_path path/to/rollouts.hdf5
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
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


def build_feature_matrix(rollouts, feature_keys=None, unit="timestep"):
    """
    Build (N, D) feature matrix from rollouts.

    unit="timestep": each row = one timestep from one rollout. Returns (X, rollout_ids, timestep_ids).
    unit="trajectory": each row = one trajectory (mean over time). Returns (X, None, None).
    """
    if feature_keys is None:
        all_keys = set()
        for r in rollouts:
            all_keys.update(k for k in r.keys() if not k.startswith("_"))
        preferred = [
            "state_rh", "state_lh",
            "state_manip_obj_rh", "state_manip_obj_lh",
            "q_rh", "q_lh", "dq_rh", "dq_lh",
            # "actions",
        ]
        feature_keys = [k for k in preferred if k in all_keys] or list(all_keys)

    if unit == "trajectory":
        all_rows = []
        for r in rollouts:
            parts = []
            for fk in feature_keys:
                if fk in r:
                    parts.append(r[fk])
            if not parts:
                continue
            T = min(p.shape[0] for p in parts)
            # Mean over time: (T, D) -> (D,)
            vec = np.concatenate([p[:T].mean(axis=0).flatten() for p in parts])
            all_rows.append(vec)
        if not all_rows:
            raise ValueError("No valid feature rows extracted from rollouts")
        X = np.stack(all_rows)
        return X, None, None

    # unit="timestep": one row per timestep
    all_rows = []
    rollout_ids = []
    timestep_ids = []
    for rid, r in enumerate(rollouts):
        parts = []
        for fk in feature_keys:
            if fk in r:
                parts.append(r[fk])
        if not parts:
            continue
        T = min(p.shape[0] for p in parts)
        for t in range(T):
            vec = np.concatenate([p[t].flatten() for p in parts])
            all_rows.append(vec)
            rollout_ids.append(rid)
            timestep_ids.append(t)
    if not all_rows:
        raise ValueError("No valid feature rows extracted from rollouts")
    X = np.stack(all_rows)
    return X, np.array(rollout_ids), np.array(timestep_ids)


def run_tsne(X, perplexity=30, n_iter=1000, random_state=42):
    """Run t-SNE on feature matrix."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=min(perplexity, X.shape[0] - 1), n_iter=n_iter, random_state=random_state)
    X_2d = tsne.fit_transform(X_scaled)
    return X_2d


def plot_tsne(X_2d, rollout_ids, timestep_ids, rollouts, out_path, unit="timestep"):
    """Create t-SNE visualization figure."""
    if unit == "trajectory":
        # One point per rollout
        rewards = []
        for r in rollouts:
            if "reward" in r:
                rewards.append(r["reward"][:].sum())
            else:
                rewards.append(0.0)
        rewards = np.array(rewards)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sc = ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=rewards,
            cmap="viridis",
            alpha=0.7,
            s=20,
        )
        ax.set_title("t-SNE: each point = one trajectory")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        plt.colorbar(sc, ax=ax, label="total reward")
        ax.set_aspect("equal")
    else:
        # One point per timestep
        n_rollouts = len(rollouts)
        colors = plt.cm.tab20(np.linspace(0, 1, max(n_rollouts, 1)))
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        for rid in range(n_rollouts):
            mask = rollout_ids == rid
            if mask.sum() == 0:
                continue
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=[colors[rid % len(colors)]],
                label=f"rollout_{rid}",
                alpha=0.6,
                s=8,
            )
        ax.set_title("t-SNE: colored by rollout")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        if n_rollouts <= 20:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.set_aspect("equal")

        ax = axes[1]
        T_max = timestep_ids.max() + 1
        t_norm = timestep_ids.astype(float) / max(T_max - 1, 1)
        sc = ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=t_norm,
            cmap="viridis",
            alpha=0.6,
            s=8,
        )
        ax.set_title("t-SNE: colored by timestep (normalized)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        plt.colorbar(sc, ax=ax, label="timestep (norm)")
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization of rollout data")
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
        "--output",
        type=str,
        default=None,
        help="Output figure path (default: <data_dir>/tsne_rollouts.png)",
    )
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity")
    parser.add_argument("--n_iter", type=int, default=1000, help="t-SNE iterations")
    parser.add_argument("--max_rollouts", type=int, default=None, help="Max rollouts to load (for quick test)")
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature keys (e.g. q_rh,dq_rh,state_rh). Default: auto-detect.",
    )
    parser.add_argument(
        "--list_features",
        action="store_true",
        help="Print available keys in the HDF5 and exit.",
    )
    parser.add_argument(
        "--unit",
        type=str,
        choices=["timestep", "trajectory"],
        default="trajectory",
        help="Unit for t-SNE: 'trajectory' = one point per rollout (default), 'timestep' = one point per timestep.",
    )
    args = parser.parse_args()

    if args.h5_path:
        h5_path = args.h5_path
    else:
        # Try rollouts.hdf5 first, then rollout.hdf5
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

    print("\nIdx | Name     | Total Reward")
    print("-" * 36)
    for idx, r in enumerate(rollouts):
        total_reward = r["reward"][:].sum() if "reward" in r else float("nan")
        print(f"  {idx:4d} | {r['_name']:8s} | {total_reward:.4f}")
    print()

    all_keys = set()
    for r in rollouts:
        all_keys.update(k for k in r.keys() if not k.startswith("_"))

    if args.list_features:
        print("Available feature keys:")
        for k in sorted(all_keys):
            shp = rollouts[0].get(k)
            print(f"  {k}: shape {shp.shape if shp is not None else 'N/A'}")
        return

    if args.max_rollouts is not None:
        rollouts = rollouts[: args.max_rollouts]
        print(f"Using first {len(rollouts)} rollouts")

    if args.features:
        feature_keys = [k.strip() for k in args.features.split(",") if k.strip()]
        missing = [k for k in feature_keys if k not in all_keys]
        if missing:
            raise ValueError(f"Unknown feature keys: {missing}. Available: {sorted(all_keys)}")
    else:
        preferred = [
            "state_rh", # , "state_lh",
            "state_manip_obj_rh", # "state_manip_obj_lh",
            "q_rh", "dq_rh", # "q_lh" "dq_lh"
            # "actions",
        ]
        feature_keys = [k for k in preferred if k in all_keys] or list(all_keys - {"_name"})
    print(f"Using feature keys: {feature_keys}")

    X, rollout_ids, timestep_ids = build_feature_matrix(rollouts, feature_keys, unit=args.unit)
    print(f"Feature matrix shape: {X.shape} ({X.shape[0]} points)")

    print("Running t-SNE...")
    X_2d = run_tsne(X, perplexity=args.perplexity, n_iter=args.n_iter)

    out_path = args.output or os.path.join(os.path.dirname(h5_path), "tsne_rollouts.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plot_tsne(X_2d, rollout_ids, timestep_ids, rollouts, out_path, unit=args.unit)


if __name__ == "__main__":
    main()
