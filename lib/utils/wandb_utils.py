import gym
import torch
import wandb
from rl_games.common.algo_observer import AlgoObserver

from lib.utils.utils import retry
from lib.utils.reformat import omegaconf_to_dict


class WandbAlgoObserver(AlgoObserver):
    """Need this to propagate the correct experiment name after initialization."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.algo = None

    def after_init(self, algo):
        self.algo = algo

    def before_init(self, base_name, config, experiment_name):
        """
        Must call initialization of Wandb before RL-games summary writer is initialized, otherwise
        sync_tensorboard does not work.
        """

        import wandb
        import os

        wandb_unique_id = f"uid_{experiment_name}"
        print(f"Wandb using unique id {wandb_unique_id}")

        cfg = self.cfg

        # Construct the summaries directory path to patch tensorboard before init
        # This matches the path structure in lib/rl/base.py
        train_dir = config.get("train_dir", "runs")
        summaries_dir = os.path.join(train_dir, experiment_name, "summaries")
        
        # Patch tensorboard before wandb.init() to avoid the warning about multiple log directories
        try:
            wandb.tensorboard.patch(root_logdir=summaries_dir)
        except Exception as e:
            print(f"Warning: Could not patch tensorboard: {e}")

        # this can fail occasionally, so we try a couple more times
        @retry(3, exceptions=(Exception,))
        def init_wandb():
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                group=cfg.wandb_group,
                tags=cfg.wandb_tags,
                sync_tensorboard=True,
                id=wandb_unique_id,
                name=experiment_name,
                resume="never",
                settings=wandb.Settings(start_method="fork"),
            )

            if cfg.wandb_logcode_dir:
                wandb.run.log_code(root=cfg.wandb_logcode_dir)
                print("wandb running directory........", wandb.run.dir)

        print("Initializing WandB...")
        wandb_initialized = False
        try:
            init_wandb()
            wandb_initialized = True
        except Exception as exc:
            print(f"Could not initialize WandB! {exc}")

        if wandb_initialized:
            if isinstance(self.cfg, dict):
                wandb.config.update(self.cfg, allow_val_change=True)
            else:
                wandb.config.update(omegaconf_to_dict(self.cfg), allow_val_change=True)

    def after_print_stats(self, frame, epoch_num, total_time):
        """Explicitly log success_rate, fail_rate, episode_lengths to wandb (tensorboard sync can miss these)."""
        try:
            if not getattr(wandb, "run", None) or self.algo is None:
                return
            log_dict = {}
            if self.algo.game_success.current_size > 0:
                log_dict["success_rate"] = float(self.algo.game_success.get_mean())
            if self.algo.game_fails.current_size > 0:
                log_dict["fail_rate"] = float(self.algo.game_fails.get_mean())
            if self.algo.game_rewards.current_size > 0:
                log_dict["episode_reward"] = float(self.algo.game_rewards.get_mean()[0])
                log_dict["episode_length"] = float(self.algo.game_lengths.get_mean())
            if log_dict:
                wandb.log(log_dict, step=frame)
        except Exception:
            pass


class WandbVideoCaptureWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        n_parallel_recorders: int = 1,
        n_successful_videos_to_record: int = 50,
    ):
        super().__init__(env)
        n_parallel_recorders = min(n_parallel_recorders, env.num_envs)
        self._n_recorders = n_parallel_recorders
        self._videos = [[] for _ in range(n_parallel_recorders)]
        self._rcd_idxs = [i for i in range(env.num_envs) if i % (env.num_envs // n_parallel_recorders) == 0][
            :n_parallel_recorders
        ]
        self._n_video_saved = 0
        self._n_successful_video_saved = 0
        self._n_successful_videos_to_record = n_successful_videos_to_record

    def reset(self, **kwargs):
        self._videos = [[] for _ in range(self._n_recorders)]
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        for i, idx in enumerate(self._rcd_idxs):
            self._videos[i].append(self.env.camera_obs[idx].clone())
        if torch.any(done):
            for i, idx in enumerate(self._rcd_idxs):
                if done[idx]:
                    video = torch.stack(self._videos[i])[..., :-1]  # (T, H, W, C), RGBA -> RGB
                    video = video.to(dtype=torch.uint8)
                    video = video.permute(0, 3, 1, 2).detach().cpu().numpy()  # (T, C, H, W)
                    video = wandb.Video(video, fps=10, format="mp4")
                    succeeded = self.env.success_buf
                    failed = self.env.failure_buf
                    status = "timeout"
                    if succeeded[idx]:
                        status = "success"
                        self._n_successful_video_saved += 1
                    elif failed[idx]:
                        status = "failure"
                    wandb.log({f"test_video/video-{self._n_video_saved}_{status}": video})
                    self._n_video_saved += 1
                    self._videos[i] = []
                    if self._n_successful_video_saved >= self._n_successful_videos_to_record:
                        exit()
        return obs, reward, done, info
