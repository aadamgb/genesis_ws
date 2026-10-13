"""nav_evader / nav_pursuer: a high-level MLP flies a frozen low-level policy (env.low_level, a model_*.pt whose run dir
holds its cfgs.pkl) to a goal (yellow sphere) by giving it waypoints (the low-level target sphere).

    nav_evader   one drone, an evader (robust_goto) policy; the waypoint is for the drone
    nav_pursuer  two drones carrying a rod, a pursuer (rod2) policy; the waypoint is for the rod center

Every env.decimation low-level steps the high-level action in [-1, 1]^3 sets the waypoint at
pos + action * waypoint_range, a cube around the controlled point (world-aligned axes). It observes the absolute
positions of the controlled point and of the goal, normalized to the safety box. A reached goal is replaced by a new
one sampled in the command ranges (the spawn region too). The episode ends when a body is on the ground or outside the
safety box (red wireframe), on a low-level failure (pursuer: rod tilt, drones too close, ...) or at the time limit.

Each env flies its own random drone(s) (drawn once at startup, as in the low-level run). The drones, controller, dt,
randomization and low-level observations come from the low-level cfgs.pkl; the env keys of
hydra_configs/task/nav_*.yaml override its env config.
"""
import os
import copy
import pickle
import numpy as np
import torch
from tensordict import TensorDict

import genesis as gs
import genesis.utils.geom as gu
from src.env_evader import EvaderEnv, gs_rand_float
from src.env_pursuer import PursuerEnv
from src.models import load_policy


def set_level_camera(env, show_viewer, lookat, distance):
    """Viewer (and env.cam if visualize_camera) at the height of lookat, distance away along x and -y, level."""
    pos = (lookat[0] + distance, lookat[1] - distance, lookat[2])
    if show_viewer:  # a full pose: with pos / lookat only the viewer keeps the up vector of its initial pose
        pose = gu.pos_lookat_up_to_T(*(np.array(v, dtype=float) for v in (pos, lookat, (0.0, 0.0, 1.0))))
        env.scene.viewer.set_camera_pose(pose=pose)
    if env.env_cfg["visualize_camera"]:
        env.cam.set_pose(pos=pos, lookat=lookat, up=(0.0, 0.0, 1.0))


class _Nav:
    """High-level env over a low-level env class (the next base in the MRO), which provides _sim_step,
    _update_observation, get_observations and reset_idx; the subclasses give _pos, _bodies and _translate."""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        with open(os.path.join(os.path.dirname(env_cfg["low_level"]), "cfgs.pkl"), "rb") as f:
            _, low_env_cfg, low_obs_cfg, _, _, _ = pickle.load(f)

        self.decimation = env_cfg["decimation"]
        self.waypoint_range = env_cfg["waypoint_range"]
        box = torch.tensor(env_cfg["safety_box"], device=gs.device, dtype=gs.tc_float)
        self.box_lo, self.box_hi = box[:, 0], box[:, 1]
        self.box_center, self.box_half = (self.box_hi + self.box_lo) / 2, (self.box_hi - self.box_lo) / 2
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
        self.low_policy = load_policy(env_cfg["low_level"], super().get_observations(), low_env_cfg["num_actions"])

        if self.env_cfg["visualize_target"]:
            self.scene.draw_debug_box(box.T, color=(1.0, 0.0, 0.0, 0.4), wireframe_radius=0.02)
        set_level_camera(self, show_viewer, self.box_center.tolist(), 14.0)

    def _failure(self):
        """Low-level failures besides the ground and the safety box."""
        return torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)

    def _resample_commands(self, envs_idx):
        """New goals; self.commands holds the waypoints."""
        self.goal[envs_idx] = self._sample_region(len(envs_idx))
        if self.env_cfg["visualize_target"] and (envs_idx < 10).any():  # the rendered envs
            if self.goal_marker is not None:
                self.scene.clear_debug_object(self.goal_marker)
            self.goal_marker = self.scene.draw_debug_spheres(self.goal[:10], radius=0.15, color=(1.0, 1.0, 0.0, 1.0))

    def _sample_region(self, n):
        c = self.command_cfg
        return torch.stack(
            [gs_rand_float(*c[k], (n,), gs.device) for k in ("pos_x_range", "pos_y_range", "pos_z_range")], dim=1
        )

    def step(self, actions):
        self.commands[:] = self._pos() + torch.clip(actions, -1.0, 1.0) * self.waypoint_range
        self.rel_pos = self.commands - self._pos()

        self.rew_buf[:] = 0.0
        done = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)
        for _ in range(self.decimation):
            self._update_observation()
            with torch.no_grad():
                low_actions = self.low_policy(super().get_observations())
            self._sim_step(low_actions)
            self.last_actions[:] = self.actions

            # the rest of the high-level step of a terminated env is ignored
            live = ~done
            pos = self._pos()
            goal_dist = torch.norm(self.goal - pos, dim=1)
            self.progress = (self.goal_dist - goal_dist) * live
            self.reached = (goal_dist < self.env_cfg["goal_threshold"]) & live
            bodies = torch.stack(self._bodies(), dim=1)  # [envs, bodies, 3]
            self.crash_condition = (
                (bodies[..., 2] < self.env_cfg["termination_if_close_to_ground"]).any(dim=1)
                | (bodies < self.box_lo).any(dim=2).any(dim=1)
                | (bodies > self.box_hi).any(dim=2).any(dim=1)
                | self._failure()
            ) & live
            for name, reward_func in self.reward_functions.items():
                rew = reward_func() * self.reward_scales[name]
                self.rew_buf += rew
                self.episode_sums[name] += rew
            done |= self.crash_condition

            reached_idx = self.reached.nonzero(as_tuple=False).reshape((-1,))
            self._resample_commands(reached_idx)
            goal_dist[reached_idx] = torch.norm(self.goal[reached_idx] - pos[reached_idx], dim=1)
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

        # spawn in the goal region with the waypoint on the controlled point
        spawn = self._sample_region(len(envs_idx))
        self._translate(envs_idx, spawn - self._pos()[envs_idx])
        self.commands[envs_idx] = spawn
        self.rel_pos = self.commands - self._pos()
        self.goal_dist[envs_idx] = torch.norm(self.goal[envs_idx] - spawn, dim=1)

    def get_observations(self):
        obs = torch.cat(
            [(self._pos() - self.box_center) / self.box_half, (self.goal - self.box_center) / self.box_half], dim=-1
        )
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    # ------------ reward functions----------------
    def _reward_progress(self):
        return self.progress

    def _reward_reach(self):
        return self.reached.float()


class NavEvaderEnv(_Nav, EvaderEnv):
    def _pos(self):
        return self.base_pos

    def _bodies(self):
        return [self.base_pos]

    def _update_observation(self):
        self._step_obs_bias()
        super()._update_observation()

    def _translate(self, envs_idx, delta):
        pos = self.base_pos[envs_idx] + delta
        self.drone.set_pos(pos, zero_velocity=False, envs_idx=envs_idx)
        self.base_pos[envs_idx] = pos
        self.last_base_pos[envs_idx] = pos


class NavPursuerEnv(_Nav, PursuerEnv):
    def _pos(self):
        return self.rod_pos

    def _bodies(self):
        return [self.rod_pos] + [d.pos for d in self.drones]

    def _failure(self):
        return PursuerEnv._failure(self)

    def _translate(self, envs_idx, delta):
        for d in self.drones:
            d.entity.set_pos(d.entity.get_pos()[envs_idx] + delta, zero_velocity=False, envs_idx=envs_idx)
            d.update_state()
        self.rod.set_pos(self.rod.get_pos()[envs_idx] + delta, zero_velocity=False, envs_idx=envs_idx)
        self.rod_pos[envs_idx] += delta
        self.last_rod_pos[envs_idx] = self.rod_pos[envs_idx]
