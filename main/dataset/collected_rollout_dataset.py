import copy
import json
import os
from functools import lru_cache
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from main.dataset.factory import ManipDataFactory
from main.dataset.transform import quat_to_aa, quat_to_rotmat
from .decorators import register_manipdata


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CollectedRolloutDatasetBase(Dataset):
    """Load collected successful rollouts and expose them as regular `data_id` samples.

    data_id format is produced by `main/tools/collect_success_trajectories.py` and
    currently follows `sr_*`.
    """

    _side_name = None  # right/left

    def __init__(
        self,
        *,
        data_dir: str = "data/collected_rollouts",
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.device = torch.device(device)
        self.mujoco2gym_transf = mujoco2gym_transf
        self.max_seq_len = max_seq_len
        self.dexhand = dexhand
        self.kwargs = kwargs

        if self._side_name not in ["right", "left"]:
            raise ValueError("_side_name must be 'right' or 'left'")

        self.registry_path = os.path.join(self.data_dir, "data_id_registry.json")
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r", encoding="utf-8") as f:
                reg = json.load(f)
        else:
            reg = {"version": 1, "items": {}}

        self.registry_items = reg.get("items", {})
        self.data_ids = sorted(self.registry_items.keys())

        self.source_dataset_cache: Dict[str, Dataset] = {}

    def __len__(self):
        return len(self.data_ids)

    @staticmethod
    def _compute_velocity(x: torch.Tensor, time_delta: float) -> torch.Tensor:
        vel = np.gradient(x.detach().cpu().numpy(), axis=0) / time_delta
        return torch.from_numpy(vel).to(x)

    @staticmethod
    def _compute_dof_velocity(x: torch.Tensor, time_delta: float) -> torch.Tensor:
        vel = np.gradient(x.detach().cpu().numpy(), axis=0) / time_delta
        return torch.from_numpy(vel).to(x)

    def _clone_template(self, d: Dict):
        out = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.clone()
            elif isinstance(v, dict):
                out[k] = self._clone_template(v)
            elif isinstance(v, list):
                out[k] = copy.deepcopy(v)
            else:
                out[k] = v
        return out

    def _get_source_dataset(self, source_data_id: str):
        source_type = ManipDataFactory.dataset_type(source_data_id)
        if source_type == "collected":
            raise ValueError("Nested collected data_id is not supported")

        if source_type not in self.source_dataset_cache:
            self.source_dataset_cache[source_type] = ManipDataFactory.create_data(
                manipdata_type=source_type,
                side=self._side_name,
                device=self.device,
                mujoco2gym_transf=self.mujoco2gym_transf,
                max_seq_len=self.max_seq_len,
                dexhand=self.dexhand,
                **self.kwargs,
            )
        return self.source_dataset_cache[source_type]

    def _pick_key(self, rollout: Dict[str, torch.Tensor], keys: List[str]) -> str:
        for k in keys:
            if k in rollout:
                return k
        raise KeyError(f"None of keys {keys} exist in rollout. Available keys: {list(rollout.keys())}")

    def _load_successful_rollout(self, rollouts_h5_path: str, rollout_name: str = None) -> Dict[str, torch.Tensor]:
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "h5py is required to load collected rollouts. Install it with `pip install h5py`."
            ) from e

        with h5py.File(rollouts_h5_path, "r") as f:
            s_grp = f["rollouts/successful"]
            rollout_names = list(s_grp.keys())
            if len(rollout_names) == 0:
                raise ValueError(f"No successful rollout found in {rollouts_h5_path}")

            if rollout_name is None:
                # Backward compatibility: pick highest-return successful rollout.
                returns = [s_grp[name]["reward"][:].sum() for name in rollout_names]
                selected_name = rollout_names[int(np.argmax(np.asarray(returns)))]
            else:
                if rollout_name not in s_grp:
                    raise KeyError(
                        f"rollout_name '{rollout_name}' not found in {rollouts_h5_path}. "
                        f"Available: {rollout_names}"
                    )
                selected_name = rollout_name
            best = s_grp[selected_name]

            out = {}
            for k in best.keys():
                out[k] = torch.tensor(best[k][:], dtype=torch.float32, device=self.device)
        return out

    def _build_from_rollout(self, template: Dict, rollout: Dict[str, torch.Tensor]) -> Dict:
        q_key = self._pick_key(rollout, ["q_rh", "q_lh"])
        dq_key = self._pick_key(rollout, ["dq_rh", "dq_lh"])
        state_key = self._pick_key(rollout, ["state_rh", "state_lh"])
        obj_state_key = self._pick_key(rollout, ["state_manip_obj_rh", "state_manip_obj_lh"])
        joint_state_key = self._pick_key(rollout, ["joint_state_rh", "joint_state_lh"])

        q = rollout[q_key]
        dq = rollout[dq_key]
        state = rollout[state_key]
        obj_state = rollout[obj_state_key]
        joint_state = rollout[joint_state_key]

        T = q.shape[0]
        if T > self.max_seq_len:
            T = self.max_seq_len
            q = q[:T]
            dq = dq[:T]
            state = state[:T]
            obj_state = obj_state[:T]
            joint_state = joint_state[:T]

        wrist_pos = state[:, :3]
        wrist_quat_xyzw = state[:, 3:7]
        wrist_quat_wxyz = wrist_quat_xyzw[:, [3, 0, 1, 2]]
        wrist_rot = quat_to_aa(wrist_quat_wxyz)
        wrist_velocity = state[:, 7:10]
        wrist_angular_velocity = state[:, 10:13]

        obj_pos = obj_state[:, :3]
        obj_quat_xyzw = obj_state[:, 3:7]
        obj_quat_wxyz = obj_quat_xyzw[:, [3, 0, 1, 2]]
        obj_rot = quat_to_rotmat(obj_quat_wxyz)

        obj_trajectory = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(T, 1, 1)
        obj_trajectory[:, :3, :3] = obj_rot
        obj_trajectory[:, :3, 3] = obj_pos

        obj_velocity = obj_state[:, 7:10]
        obj_angular_velocity = obj_state[:, 10:13]

        mano_joints = {}
        mano_joints_velocity = {}
        for i, body_name in enumerate(self.dexhand.body_names):
            hand_name = self.dexhand.to_hand(body_name)[0]
            body_state = joint_state[:, i * 13 : (i + 1) * 13]
            if hand_name not in mano_joints:
                mano_joints[hand_name] = body_state[:, :3]
                mano_joints_velocity[hand_name] = body_state[:, 7:10]

        required_keys = [
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
            "pinky_tip",
        ]
        for k in required_keys:
            if k not in mano_joints:
                raise KeyError(f"Missing joint key '{k}' in reconstructed rollout joints")

        obj_verts = template["obj_verts"]
        obj_verts_transf = (obj_trajectory[:, :3, :3] @ obj_verts.T[None]).transpose(-1, -2) + obj_trajectory[
            :, :3, 3
        ][:, None]
        tips = torch.stack([mano_joints[k] for k in required_keys], dim=1)
        tips_distance = torch.cdist(tips, obj_verts_transf).min(dim=-1).values

        dt = 1 / 60.0
        opt_wrist_pos = wrist_pos
        opt_wrist_rot = wrist_rot
        opt_dof_pos = q
        opt_wrist_velocity = wrist_velocity
        opt_wrist_angular_velocity = wrist_angular_velocity
        opt_dof_velocity = self._compute_dof_velocity(opt_dof_pos, dt)

        data = self._clone_template(template)
        data.update(
            {
                "obj_trajectory": obj_trajectory,
                "wrist_pos": wrist_pos,
                "wrist_rot": wrist_rot,
                "mano_joints": mano_joints,
                "tips_distance": tips_distance,
                "obj_velocity": obj_velocity,
                "obj_angular_velocity": obj_angular_velocity,
                "wrist_velocity": wrist_velocity,
                "wrist_angular_velocity": wrist_angular_velocity,
                "mano_joints_velocity": mano_joints_velocity,
                "opt_wrist_pos": opt_wrist_pos,
                "opt_wrist_rot": opt_wrist_rot,
                "opt_dof_pos": opt_dof_pos,
                "opt_wrist_velocity": opt_wrist_velocity,
                "opt_wrist_angular_velocity": opt_wrist_angular_velocity,
                "opt_dof_velocity": opt_dof_velocity,
            }
        )

        return data

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        if isinstance(index, int):
            data_id = self.data_ids[index]
        elif isinstance(index, str):
            data_id = index
        else:
            raise TypeError("index must be int or str")

        if data_id not in self.registry_items:
            raise KeyError(
                f"Collected data_id '{data_id}' not found in {self.registry_path}. "
                "Use `python main/tools/collect_success_trajectories.py list` to inspect available IDs."
            )

        item = self.registry_items[data_id]
        source_data_id = item["source_data_id"]
        source_dataset = self._get_source_dataset(source_data_id)
        template = source_dataset[source_data_id]

        rel_h5 = item.get("artifacts", {}).get("rollouts.hdf5")
        if rel_h5 is None:
            rel_h5 = os.path.join(self.data_dir, data_id, "rollouts.hdf5")
        h5_path = rel_h5 if os.path.isabs(rel_h5) else os.path.join(PROJECT_ROOT, rel_h5)
        if not os.path.exists(h5_path):
            # Fallback to path rooted at data_dir.
            h5_path = os.path.join(self.data_dir, data_id, "rollouts.hdf5")

        rollout_name = item.get("rollout_name", None)
        rollout = self._load_successful_rollout(h5_path, rollout_name=rollout_name)
        data = self._build_from_rollout(template, rollout)

        return data


@register_manipdata("collected_rh")
class CollectedRolloutDatasetRH(CollectedRolloutDatasetBase):
    _side_name = "right"


@register_manipdata("collected_lh")
class CollectedRolloutDatasetLH(CollectedRolloutDatasetBase):
    _side_name = "left"
