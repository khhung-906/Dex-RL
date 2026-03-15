#!/usr/bin/env python3
"""Collect successful residual-policy trajectories and register them by data_id.

This script wraps `main/rl/train.py` in test mode with rollout dumping enabled,
then creates/updates a local registry so collected trajectories can be referenced
via a stable data_id in future data collection workflows.

Notes:
- It does not modify the core training dataloader.
- It provides a data_id registry for collected rollout files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
DUMPS_DIR = ROOT / "dumps"
COLLECT_DIR = ROOT / "data" / "collected_rollouts"
REGISTRY_PATH = COLLECT_DIR / "data_id_registry.json"


@dataclass
class RolloutStats:
    successful: int
    failed: int


@dataclass
class SuccessfulRolloutInfo:
    name: str
    total_reward: float


def _ensure_dirs() -> None:
    DUMPS_DIR.mkdir(parents=True, exist_ok=True)
    COLLECT_DIR.mkdir(parents=True, exist_ok=True)


def _load_registry() -> Dict[str, Any]:
    if REGISTRY_PATH.exists():
        with REGISTRY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid registry format in {REGISTRY_PATH}")
        data.setdefault("version", 1)
        data.setdefault("items", {})
        return data
    return {"version": 1, "items": {}}


def _save_registry(registry: Dict[str, Any]) -> None:
    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=True)


def _list_dump_dirs() -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not DUMPS_DIR.exists():
        return out
    for p in DUMPS_DIR.iterdir():
        if p.is_dir():
            out[p.name] = p.stat().st_mtime
    return out


def _sanitize_data_id(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9_\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _default_collection_id(source_data_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"sr_{_sanitize_data_id(source_data_id)}_{ts}"


def _find_newest_rollout_dump(before: Dict[str, float]) -> Path:
    after = _list_dump_dirs()
    created = [name for name in after.keys() if name not in before]
    candidates: List[Path] = []

    for name in created:
        p = DUMPS_DIR / name / "rollouts.hdf5"
        if p.exists():
            candidates.append(p.parent)

    if not candidates:
        # Fallback: pick newest dump folder that contains rollouts.hdf5.
        for name, _ in sorted(after.items(), key=lambda x: x[1], reverse=True):
            p = DUMPS_DIR / name / "rollouts.hdf5"
            if p.exists():
                candidates.append(p.parent)
                break

    if not candidates:
        raise FileNotFoundError("No dump folder with rollouts.hdf5 was found under dumps/")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _count_rollouts(h5_path: Path) -> RolloutStats:
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required to read rollouts.hdf5. Install it with `pip install h5py`."
        ) from e

    with h5py.File(h5_path, "r") as f:
        successful = len(f["rollouts/successful"].keys())
        failed = len(f["rollouts/failed"].keys())
    return RolloutStats(successful=successful, failed=failed)


def _list_successful_rollouts(h5_path: Path) -> List[SuccessfulRolloutInfo]:
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required to read rollouts.hdf5. Install it with `pip install h5py`."
        ) from e

    out: List[SuccessfulRolloutInfo] = []
    with h5py.File(h5_path, "r") as f:
        s_grp = f["rollouts/successful"]
        for name in s_grp.keys():
            reward = float(s_grp[name]["reward"][:].sum())
            out.append(SuccessfulRolloutInfo(name=name, total_reward=reward))
    out.sort(key=lambda x: x.total_reward, reverse=True)
    return out


def _copy_dump_artifacts(src_dump_dir: Path, dst_dir: Path) -> Dict[str, str]:
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied: Dict[str, str] = {}
    for name in ["rollouts.hdf5", "config.yaml"]:
        src = src_dump_dir / name
        if src.exists():
            dst = dst_dir / name
            shutil.copy2(src, dst)
            copied[name] = str(dst.relative_to(ROOT))
    return copied


def _build_train_command(args: argparse.Namespace, experiment_name: str) -> List[str]:
    # Keep values as Hydra overrides to align with existing scripts.
    cmd = [
        sys.executable,
        "main/rl/train.py",
        f"task={args.task}",
        f"dexhand={args.dexhand}",
        f"side={args.side}",
        f"headless={str(args.headless).lower()}",
        f"num_envs={args.num_envs}",
        f"learning_rate={args.learning_rate}",
        "test=true",
        f"randomStateInit={str(args.random_state_init).lower()}",
        f"dataIndices=[{args.source_data_id}]",
        f"actionsMovingAverage={args.actions_moving_average}",
        f"checkpoint={args.residual_checkpoint}",
        f"rh_base_model_checkpoint={args.rh_base_model_checkpoint}",
        f"lh_base_model_checkpoint={args.lh_base_model_checkpoint}",
        "save_rollouts=true",
        "save_successful_rollouts_only=true",
        f"num_rollouts_to_save={args.num_rollouts_to_save}",
        f"num_rollouts_to_run={args.num_rollouts_to_run}",
        f"min_episode_length={args.min_episode_length}",
        f"wandb_activate={str(args.wandb_activate).lower()}",
        f"experiment={experiment_name}",
    ]

    if args.extra_override:
        cmd.extend(args.extra_override)

    return cmd


def _run_collect(args: argparse.Namespace) -> None:
    _ensure_dirs()

    if args.max_child_rollouts is not None and args.max_child_rollouts < 1:
        raise ValueError("--max-child-rollouts must be >= 1")

    collection_id = args.collection_id or _default_collection_id(args.source_data_id)
    safe_collection_id = _sanitize_data_id(collection_id)
    if not safe_collection_id:
        raise ValueError("collection_id is empty after sanitization")

    registry = _load_registry()
    items = registry["items"]
    if safe_collection_id in items and not args.overwrite:
        raise ValueError(
            f"collection_id '{safe_collection_id}' already exists in registry. "
            "Use --overwrite or choose another --collection-id."
        )

    before = _list_dump_dirs()
    experiment_name = args.experiment_name or f"collect_{safe_collection_id}"
    cmd = _build_train_command(args, experiment_name=experiment_name)

    print("[info] Running rollout collection command:")
    print("       " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    dump_dir = _find_newest_rollout_dump(before)
    rollout_path = dump_dir / "rollouts.hdf5"
    stats = _count_rollouts(rollout_path)
    successful_rollouts = _list_successful_rollouts(rollout_path)
    if args.max_child_rollouts is not None:
        successful_rollouts = successful_rollouts[: args.max_child_rollouts]

    if stats.successful == 0:
        print("[warn] 0 successful rollouts were collected.")

    output_dir = COLLECT_DIR / safe_collection_id
    copied = _copy_dump_artifacts(dump_dir, output_dir)

    now = datetime.now().isoformat(timespec="seconds")
    items[safe_collection_id] = {
        "collection_id": safe_collection_id,
        "created_at": now,
        "source_data_id": args.source_data_id,
        "task": args.task,
        "dexhand": args.dexhand,
        "side": args.side,
        "residual_checkpoint": args.residual_checkpoint,
        "rh_base_model_checkpoint": args.rh_base_model_checkpoint,
        "lh_base_model_checkpoint": args.lh_base_model_checkpoint,
        "num_rollouts_to_save": args.num_rollouts_to_save,
        "num_rollouts_to_run": args.num_rollouts_to_run,
        "min_episode_length": args.min_episode_length,
        "max_child_rollouts": args.max_child_rollouts,
        "registered_child_rollouts": len(successful_rollouts),
        "successful_rollouts": stats.successful,
        "failed_rollouts": stats.failed,
        "dump_dir": str(dump_dir.relative_to(ROOT)),
        "artifacts": copied,
        "child_rollout_ids": [],
    }

    # Register each successful rollout as an independent data_id for ablation.
    child_ids: List[str] = []
    for i, rollout in enumerate(successful_rollouts):
        child_id = f"{safe_collection_id}_r{i:03d}"
        if child_id in items and not args.overwrite:
            raise ValueError(
                f"child data_id '{child_id}' already exists in registry. "
                "Use --overwrite or choose another --collection-id."
            )
        items[child_id] = {
            "collection_id": child_id,
            "created_at": now,
            "source_data_id": args.source_data_id,
            "task": args.task,
            "dexhand": args.dexhand,
            "side": args.side,
            "residual_checkpoint": args.residual_checkpoint,
            "rh_base_model_checkpoint": args.rh_base_model_checkpoint,
            "lh_base_model_checkpoint": args.lh_base_model_checkpoint,
            "num_rollouts_to_save": args.num_rollouts_to_save,
            "num_rollouts_to_run": args.num_rollouts_to_run,
            "min_episode_length": args.min_episode_length,
            "max_child_rollouts": args.max_child_rollouts,
            "successful_rollouts": stats.successful,
            "failed_rollouts": stats.failed,
            "dump_dir": str(dump_dir.relative_to(ROOT)),
            "artifacts": copied,
            "parent_collection_id": safe_collection_id,
            "rollout_name": rollout.name,
            "rollout_rank": i,
            "rollout_return": rollout.total_reward,
        }
        child_ids.append(child_id)

    items[safe_collection_id]["child_rollout_ids"] = child_ids

    _save_registry(registry)

    print("[done] Collection completed.")
    print(f"       data_id: {safe_collection_id}")
    print(f"       successful_rollouts: {stats.successful}")
    print(f"       failed_rollouts: {stats.failed}")
    print(f"       registered_child_rollouts: {len(child_ids)}")
    print(f"       rollouts: {copied.get('rollouts.hdf5', 'N/A')}")
    print(f"       child_rollout_ids: {len(child_ids)}")
    print(f"       registry: {REGISTRY_PATH.relative_to(ROOT)}")


def _run_list(_: argparse.Namespace) -> None:
    _ensure_dirs()
    reg = _load_registry()
    items = reg.get("items", {})
    if not items:
        print("[info] Registry is empty.")
        return

    print("data_id\tsource_data_id\trollout\tcreated_at")
    for k in sorted(items.keys()):
        v = items[k]
        rollout_name = v.get("rollout_name", "-")
        print(
            f"{k}\t{v.get('source_data_id','-')}\t{rollout_name}\t{v.get('created_at','-')}"
        )


def _run_show(args: argparse.Namespace) -> None:
    _ensure_dirs()
    reg = _load_registry()
    item = reg.get("items", {}).get(args.data_id)
    if item is None:
        raise KeyError(f"data_id '{args.data_id}' not found in registry")
    print(json.dumps(item, indent=2, ensure_ascii=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect and register successful residual-policy trajectories")
    sub = parser.add_subparsers(dest="cmd", required=True)

    collect = sub.add_parser("collect", help="Run policy, save successful rollouts, and register as data_id")
    collect.add_argument("--source-data-id", required=True, help="Original task id, e.g. 75a7a@1")
    collect.add_argument("--residual-checkpoint", required=True, help="Residual checkpoint path (.pth)")
    collect.add_argument("--rh-base-model-checkpoint", required=True, help="RH imitation checkpoint path (.pth)")
    collect.add_argument("--lh-base-model-checkpoint", default="assets/imitator_lh_inspire.pth")

    collect.add_argument("--task", default="ResDexHand")
    collect.add_argument("--dexhand", default="inspire")
    collect.add_argument("--side", default="RH")
    collect.add_argument("--headless", type=lambda x: str(x).lower() == "true", default=True)
    collect.add_argument("--num-envs", type=int, default=4)
    collect.add_argument("--learning-rate", default="2e-4")
    collect.add_argument("--random-state-init", type=lambda x: str(x).lower() == "true", default=True)
    collect.add_argument("--actions-moving-average", default="0.4")

    collect.add_argument("--num-rollouts-to-save", type=int, default=200)
    collect.add_argument("--num-rollouts-to-run", type=int, default=1000)
    collect.add_argument("--min-episode-length", type=int, default=20)
    collect.add_argument(
        "--max-child-rollouts",
        type=int,
        default=1000,
        help="Max number of top-reward successful rollouts to register as child data_id entries",
    )
    collect.add_argument("--wandb-activate", type=lambda x: str(x).lower() == "true", default=False)

    collect.add_argument(
        "--collection-id",
        default=None,
        help="Registered data_id for this collected set. Default: sr_<source>_<timestamp>",
    )
    collect.add_argument("--experiment-name", default=None, help="Hydra experiment name for the dump run")
    collect.add_argument("--overwrite", action="store_true", help="Overwrite existing collection_id in registry")
    collect.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Extra Hydra override (repeatable), e.g. --extra-override usePIDControl=True",
    )
    collect.set_defaults(func=_run_collect)

    list_cmd = sub.add_parser("list", help="List registered collected trajectory data_id entries")
    list_cmd.set_defaults(func=_run_list)

    show = sub.add_parser("show", help="Show one registered entry")
    show.add_argument("data_id")
    show.set_defaults(func=_run_show)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
