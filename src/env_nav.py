"""nav: a high-level MLP flies a frozen robust_goto policy (env.low_level, a model_*.pt whose run dir holds its
cfgs.pkl) to a goal (yellow sphere) by giving it waypoints (pink sphere).

Every env.decimation low-level steps the high-level action in [-1, 1]^3 sets the waypoint at
base_pos + action * waypoint_range, a cube around the drone (world-aligned axes). It observes the absolute drone and
goal positions, normalized to the safety box. A reached goal is replaced by a new one sampled in the command ranges
(the spawn region too). The episode ends on the ground, outside the safety box (red wireframe) or at the time limit.

Each env flies its own random drone (drawn once at startup, as in the low-level run). The drone, controller, dt,
randomization and low-level observations come from the low-level cfgs.pkl; the env keys of
hydra_configs/task/nav.yaml override its env config.
"""
import os
import copy
import pickle
import torch
from tensordict import TensorDict
from rsl_rl.utils import resolve_class

import genesis as gs
from src.env_robust_goto import RobustGotoEnv, gs_rand_float


class NavEnv(RobustGotoEnv):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        with open(os.path.join(os.path.dirname(env_cfg["low_level"]), "cfgs.pkl"), "rb") as f:
            _, low_env_cfg, low_obs_cfg, _, _, low_train_cfg = pickle.load(f)

        self.decimation = env_cfg["decimation"]
        self.waypoint_range = env_cfg["waypoint_range"]
        box = torch.tensor(env_cfg["safety_box"], device=gs.device, dtype=gs.tc_float)
        self.box_lo, self.box_hi = box[:, 0], box[:, 1]
        self.box_center, self.box_half = (self.box_hi + self.box_lo) / 2, (self.box_hi - self.box_lo) / 2
        self.visualize = env_cfg["visualize_target"]
        self.goal_marker = None
        self.goal = torch.zeros((num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.goal_dist = torch.zeros((num_envs,), device=gs.device, dtype=gs.tc_float)
        self.progress = torch.zeros_like(self.goal_dist)
        self.reached = torch.zeros((num_envs,), device=gs.device, dtype=torch.bool)

        super().__init__(num_envs, {**low_env_cfg, **env_cfg}, low_obs_cfg, reward_cfg, command_cfg, show_viewer)

        # rewards per high-level step (per m of progress, per goal, per crash), not per second
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])
        self.num_actions = 3  # waypoint; self.actions holds the low-level ones

        # frozen low-level policy
        actor_class, actor_cfg = resolve_class(low_train_cfg["actor"])
        self.low_policy = actor_class(
            super().get_observations(), low_train_cfg["obs_groups"], "actor", env_cfg["num_actions"], **actor_cfg
        ).to(gs.device)
        state = torch.load(env_cfg["low_level"], map_location=gs.device, weights_only=False)
        self.low_policy.load_state_dict(state["actor_state_dict"])
        self.low_policy.eval()

        if self.visualize:
            self.scene.draw_debug_box(box.T, color=(1.0, 0.0, 0.0, 0.4), wireframe_radius=0.02)
        cam_pos  = (0.0, 0.0, 5.0)
        cam_lookat =  (0.0, 0.0, 1.0)
        if show_viewer:
            self.scene.viewer.set_camera_pose(pos=cam_pos, lookat=cam_lookat)
        if self.env_cfg["visualize_camera"]:
            self.cam.set_pose(pos=cam_pos, lookat=cam_lookat)

    def _resample_commands(self, envs_idx):
        """New goals; self.commands holds the waypoints."""
        self.goal[envs_idx] = self._sample_region(len(envs_idx))
        if self.visualize and (envs_idx < self.rendered_env_num).any():
            if self.goal_marker is not None:
                self.scene.clear_debug_object(self.goal_marker)
            self.goal_marker = self.scene.draw_debug_spheres(
                self.goal[: self.rendered_env_num], radius=0.15, color=(1.0, 1.0, 0.0, 1.0)
            )

    def _sample_region(self, n):
        c = self.command_cfg
        return torch.stack(
            [gs_rand_float(*c[k], (n,), gs.device) for k in ("pos_x_range", "pos_y_range", "pos_z_range")], dim=1
        )

    def step(self, actions):
        self.commands[:] = self.base_pos + torch.clip(actions, -1.0, 1.0) * self.waypoint_range
        self.rel_pos = self.commands - self.base_pos

        self.rew_buf[:] = 0.0
        done = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)
        for _ in range(self.decimation):
            self._step_obs_bias()
            self._update_observation()
            with torch.no_grad():
                low_actions = self.low_policy(super().get_observations())
            self._sim_step(low_actions)
            self.last_actions[:] = self.actions

            # the rest of the high-level step of a terminated env is ignored
            live = ~done
            goal_dist = torch.norm(self.goal - self.base_pos, dim=1)
            self.progress = (self.goal_dist - goal_dist) * live
            self.reached = (goal_dist < self.env_cfg["goal_threshold"]) & live
            self.crash_condition = (
                (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
                | (self.base_pos < self.box_lo).any(dim=1)
                | (self.base_pos > self.box_hi).any(dim=1)
            ) & live
            for name, reward_func in self.reward_functions.items():
                rew = reward_func() * self.reward_scales[name]
                self.rew_buf += rew
                self.episode_sums[name] += rew
            done |= self.crash_condition

            reached_idx = self.reached.nonzero(as_tuple=False).reshape((-1,))
            self._resample_commands(reached_idx)
            goal_dist[reached_idx] = torch.norm(self.goal[reached_idx] - self.base_pos[reached_idx], dim=1)
            self.goal_dist = goal_dist

        time_out = self.episode_length_buf > self.max_episode_length
        self.reset_buf = done | time_out
        self.extras["time_outs"] = (time_out & ~done).to(gs.tc_float)
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        super().reset_idx(envs_idx)  # also samples the goals

        # spawn in the goal region with the waypoint on the drone
        pos = self._sample_region(len(envs_idx))
        self.drone.set_pos(pos, zero_velocity=False, envs_idx=envs_idx)
        self.base_pos[envs_idx] = pos
        self.last_base_pos[envs_idx] = pos
        self.commands[envs_idx] = pos
        self.rel_pos = self.commands - self.base_pos
        self.goal_dist[envs_idx] = torch.norm(self.goal[envs_idx] - pos, dim=1)

    def get_observations(self):
        obs = torch.cat(
            [(self.base_pos - self.box_center) / self.box_half, (self.goal - self.box_center) / self.box_half], dim=-1
        )
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    # ------------ reward functions----------------
    def _reward_progress(self):
        return self.progress

    def _reward_reach(self):
        return self.reached.float()
