#!/usr/bin/env python3
"""Directly replay collected rollout trajectories (no policy inference).

This tool visualizes the actual trajectory saved in rollouts.hdf5 for IDs like
sr_75a7a_2_r000, sr_75a7a_2_r001, ...
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from datetime import datetime
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List

import h5py


@dataclass
class ReplayTarget:
    data_id: str
    source_data_id: str
    rollout_name: str
    rollouts_h5_path: str
    dexhand: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay collected rollouts from registry")
    parser.add_argument("--registry", type=str, default="data/collected_rollouts/data_id_registry.json")
    parser.add_argument("--prefix", type=str, default="sr_75a7a_2_r")
    parser.add_argument(
        "--ids",
        type=str,
        default="",
        help="Comma-separated data_id list. If set, ignores --prefix/--start/--end/--step.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=499)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--side", type=str, choices=["rh", "lh"], default="rh")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means replay one full trajectory")
    parser.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between trajectories")
    parser.add_argument(
        "--save-frames",
        dest="save_frames",
        action="store_true",
        help="Save replay frames to --save-dir",
    )
    parser.add_argument(
        "--no-save-frames",
        dest="save_frames",
        action="store_false",
        help="Disable replay frame saving",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="dumps/replay_sr_75a7a_2",
        help="Directory to store replay frames",
    )
    parser.add_argument("--frame-every", type=int, default=1, help="Save one frame every N sim steps")
    parser.add_argument(
        "--wandb",
        dest="wandb",
        action="store_true",
        help="Upload replay videos to Weights & Biases",
    )
    parser.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="Disable Weights & Biases upload",
    )
    parser.add_argument("--wandb-project", type=str, default="Dex-RL")
    parser.add_argument("--wandb-entity", type=str, default="maxliu01-stanford-university")
    parser.add_argument("--wandb-group", type=str, default="ReplayCollected")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--wandb-fps", type=int, default=10)
    parser.add_argument("--wandb-key-prefix", type=str, default="replay_video")
    parser.add_argument(
        "--cleanup-local",
        dest="cleanup_local",
        action="store_true",
        help="Delete local frames/mp4 after each replay upload",
    )
    parser.add_argument(
        "--keep-local",
        dest="cleanup_local",
        action="store_false",
        help="Keep local frames/mp4 on disk",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.set_defaults(save_frames=True, wandb=True, cleanup_local=True)
    return parser.parse_args()


def _load_registry(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", {})
    if not isinstance(items, dict):
        raise ValueError(f"Invalid items field in registry: {path}")
    return items


def _collect_targets(args: argparse.Namespace, items: Dict) -> List[ReplayTarget]:
    out: List[ReplayTarget] = []
    for i in range(args.start, args.end + 1, args.step):
        data_id = f"{args.prefix}{i:03d}"
        item = items.get(data_id)
        if item is None:
            raise KeyError(f"Missing data_id in registry: {data_id}")

        source_data_id = item.get("source_data_id")
        rollout_name = item.get("rollout_name")
        dexhand = item.get("dexhand", "inspire")
        rel_h5 = item.get("artifacts", {}).get("rollouts.hdf5")
        if not source_data_id or not rollout_name or not rel_h5:
            raise ValueError(f"Registry item for {data_id} is missing source_data_id/rollout_name/artifacts.rollouts.hdf5")

        h5_path = rel_h5 if os.path.isabs(rel_h5) else os.path.join(os.getcwd(), rel_h5)
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"rollouts.hdf5 not found for {data_id}: {h5_path}")

        out.append(
            ReplayTarget(
                data_id=data_id,
                source_data_id=source_data_id,
                rollout_name=rollout_name,
                rollouts_h5_path=h5_path,
                dexhand=dexhand,
            )
        )
    return out


def _collect_targets_by_ids(data_ids: List[str], items: Dict) -> List[ReplayTarget]:
    out: List[ReplayTarget] = []
    for data_id in data_ids:
        item = items.get(data_id)
        if item is None:
            raise KeyError(f"Missing data_id in registry: {data_id}")

        source_data_id = item.get("source_data_id")
        rollout_name = item.get("rollout_name")
        dexhand = item.get("dexhand", "inspire")
        rel_h5 = item.get("artifacts", {}).get("rollouts.hdf5")
        if not source_data_id or not rollout_name or not rel_h5:
            raise ValueError(f"Registry item for {data_id} is missing source_data_id/rollout_name/artifacts.rollouts.hdf5")

        h5_path = rel_h5 if os.path.isabs(rel_h5) else os.path.join(os.getcwd(), rel_h5)
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"rollouts.hdf5 not found for {data_id}: {h5_path}")

        out.append(
            ReplayTarget(
                data_id=data_id,
                source_data_id=source_data_id,
                rollout_name=rollout_name,
                rollouts_h5_path=h5_path,
                dexhand=dexhand,
            )
        )
    return out


def _load_source_template(source_data_id: str, side: str, dexhand_name: str, device: str):
    import torch

    from main.dataset.factory import ManipDataFactory
    from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

    side_name = "right" if side == "rh" else "left"
    dexhand = DexHandFactory.create_hand(dexhand_name, side_name)
    mujoco2gym_transf = torch.eye(4, device=device)

    dataset_type = ManipDataFactory.dataset_type(source_data_id)
    dataset = ManipDataFactory.create_data(
        manipdata_type=dataset_type,
        side=side_name,
        device=device,
        mujoco2gym_transf=mujoco2gym_transf,
        max_seq_len=1200,
        dexhand=dexhand,
        embodiment=dexhand_name,
    )
    return dataset[source_data_id]


def _load_rollout_arrays(h5_path: str, rollout_name: str, side: str) -> Dict:
    import torch

    side_key = "rh" if side == "rh" else "lh"
    q_key = f"q_{side_key}"
    dq_key = f"dq_{side_key}"
    state_key = f"state_{side_key}"
    obj_state_key = f"state_manip_obj_{side_key}"

    with h5py.File(h5_path, "r") as f:
        group = f[f"rollouts/successful/{rollout_name}"]
        required = [q_key, dq_key, state_key, obj_state_key]
        for k in required:
            if k not in group:
                raise KeyError(f"Key '{k}' not found in rollout '{rollout_name}' ({h5_path})")

        return {
            q_key: torch.tensor(group[q_key][:], dtype=torch.float32),
            dq_key: torch.tensor(group[dq_key][:], dtype=torch.float32),
            state_key: torch.tensor(group[state_key][:], dtype=torch.float32),
            obj_state_key: torch.tensor(group[obj_state_key][:], dtype=torch.float32),
        }


def _build_rollout_seq(template: Dict, rollout_arr: Dict, side: str, dexhand_name: str) -> Dict:
    side_key = "rh" if side == "rh" else "lh"

    obj_id_key = "obj_id"
    obj_urdf_key = "obj_urdf_path"
    if obj_id_key not in template or obj_urdf_key not in template:
        raise KeyError("Template data missing obj_id or obj_urdf_path")

    rollout_seq = {
        "dexhand": dexhand_name,
        f"q_{side_key}": rollout_arr[f"q_{side_key}"],
        f"dq_{side_key}": rollout_arr[f"dq_{side_key}"],
        f"state_{side_key}": rollout_arr[f"state_{side_key}"],
        f"state_manip_obj_{side_key}": rollout_arr[f"state_manip_obj_{side_key}"],
        f"oid_{side_key}": template["obj_id"],
        f"obj_{side_key}_path": template["obj_urdf_path"],
        "scene_objs": template.get("scene_objs", []),
    }
    return rollout_seq


def _build_vis_args(headless: bool) -> SimpleNamespace:
    from isaacgym import gymapi

    return SimpleNamespace(
        headless=headless,
        use_gpu=True,
        use_gpu_pipeline=True,
        sim_device="cuda:0",
        compute_device_id=0,
        graphics_device_id=0,
        physics_engine=gymapi.SIM_PHYSX,
    )


def _frame_paths(frame_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))


def _save_mp4_from_frames(frame_dir: str, output_mp4_path: str, fps: int) -> str:
    frame_paths = _frame_paths(frame_dir)
    if len(frame_paths) == 0:
        return "no_frames"

    try:
        import imageio.v2 as imageio
    except Exception:
        return "missing_imageio"

    os.makedirs(os.path.dirname(output_mp4_path), exist_ok=True)
    with imageio.get_writer(output_mp4_path, fps=fps, codec="libx264") as writer:
        for p in frame_paths:
            writer.append_data(imageio.imread(p))
    return "ok"


def _init_wandb(args: argparse.Namespace, n_targets: int):
    if not args.wandb or args.dry_run:
        return None

    try:
        import wandb
    except Exception as exc:
        print(f"[warn] wandb import failed, skip wandb logging: {exc}")
        return None

    run_name = args.wandb_name.strip()
    if not run_name:
        run_name = f"replay_{datetime.now().strftime('%m-%d-%H-%M-%S')}"

    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=run_name,
            reinit=True,
        )
    except Exception as exc:
        print(f"[warn] wandb.init failed, skip wandb logging: {exc}")
        return None

    wandb.config.update(
        {
            "replay_registry": args.registry,
            "replay_side": args.side,
            "replay_num_targets": n_targets,
            "replay_max_steps": args.max_steps,
            "replay_save_dir": args.save_dir,
            "replay_frame_every": args.frame_every,
        },
        allow_val_change=True,
    )
    return run


def _log_video_to_wandb(run, mp4_path: str, key: str, step: int, data_id: str, fps: int) -> None:
    if run is None:
        return
    try:
        import wandb

        run.log(
            {
                key: wandb.Video(mp4_path, fps=fps, format="mp4"),
                "replay/data_id": data_id,
            },
            step=step,
        )
    except Exception as exc:
        print(f"[warn] wandb log failed for {data_id}: {exc}")


def main() -> None:
    args = _parse_args()
    if args.start < 0 or args.end < 0 or args.start > args.end:
        raise ValueError("Invalid range. Require 0 <= start <= end")
    if args.step < 1:
        raise ValueError("step must be >= 1")
    if args.frame_every < 1:
        raise ValueError("frame_every must be >= 1")
    if args.wandb_fps < 1:
        raise ValueError("wandb-fps must be >= 1")

    items = _load_registry(args.registry)
    if args.ids.strip():
        data_ids = [x.strip() for x in args.ids.split(",") if x.strip()]
        if len(data_ids) == 0:
            raise ValueError("--ids is set but empty after parsing")
        targets = _collect_targets_by_ids(data_ids, items)
    else:
        targets = _collect_targets(args, items)

    print(f"[info] replay targets: {len(targets)} ({targets[0].data_id} .. {targets[-1].data_id})")

    source_cache: Dict[str, Dict] = {}
    vis_args = None

    if args.headless and args.save_frames:
        print("[warn] headless mode cannot capture viewer frames; disabling frame save and wandb upload")
        args.save_frames = False
        args.wandb = False

    wandb_run = _init_wandb(args, n_targets=len(targets))

    if args.save_frames and not args.dry_run:
        os.makedirs(args.save_dir, exist_ok=True)

    for idx, t in enumerate(targets, start=1):
        print(f"[run {idx}/{len(targets)}] {t.data_id} -> {t.rollout_name} (source={t.source_data_id})")

        if args.dry_run:
            continue

        if vis_args is None:
            vis_args = _build_vis_args(headless=args.headless)

        if t.source_data_id not in source_cache:
            source_cache[t.source_data_id] = _load_source_template(
                source_data_id=t.source_data_id,
                side=args.side,
                dexhand_name=t.dexhand,
                device=args.device,
            )

        rollout_arr = _load_rollout_arrays(t.rollouts_h5_path, t.rollout_name, args.side)
        rollout_seq = _build_rollout_seq(source_cache[t.source_data_id], rollout_arr, args.side, t.dexhand)

        total_len = len(rollout_seq[f"q_{args.side}"])
        max_steps = args.max_steps if args.max_steps > 0 else total_len

        if args.side == "rh":
            from DexManipNet.dexmanip_sh import DexManipSH_RH

            vis_env = DexManipSH_RH(vis_args, rollout_seq)
        else:
            from DexManipNet.dexmanip_sh import DexManipSH_LH

            vis_env = DexManipSH_LH(vis_args, rollout_seq)

        frame_dir = None
        if args.save_frames:
            frame_dir = os.path.join(args.save_dir, t.data_id)

        vis_env.play(max_steps=max_steps, frame_dir=frame_dir, frame_every=args.frame_every)

        mp4_path = None
        if frame_dir is not None and wandb_run is not None:
            mp4_path = os.path.join(args.save_dir, f"{t.data_id}.mp4")
            save_status = _save_mp4_from_frames(frame_dir, mp4_path, fps=args.wandb_fps)
            if save_status == "ok":
                wandb_key = f"{args.wandb_key_prefix}/{t.data_id}"
                _log_video_to_wandb(
                    wandb_run,
                    mp4_path,
                    key=wandb_key,
                    step=idx,
                    data_id=t.data_id,
                    fps=args.wandb_fps,
                )
            elif save_status == "no_frames":
                print(
                    f"[warn] no frames captured for {t.data_id}. Viewer may have failed to initialize (DISPLAY/GLFW issue)."
                )
            elif save_status == "missing_imageio":
                print(f"[warn] imageio/imageio-ffmpeg missing, cannot build mp4 for {t.data_id}.")
            else:
                print(f"[warn] cannot build mp4 for {t.data_id}. status={save_status}")

        if args.cleanup_local and frame_dir is not None:
            if os.path.isdir(frame_dir):
                shutil.rmtree(frame_dir, ignore_errors=True)
            if mp4_path is not None and os.path.isfile(mp4_path):
                try:
                    os.remove(mp4_path)
                except OSError:
                    pass

        if args.sleep > 0:
            import time

            time.sleep(args.sleep)

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
