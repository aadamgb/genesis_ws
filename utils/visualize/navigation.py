"""Visualize a nav_pursuer and a nav_evader (nav) policy flying in the same scene, no training: the pursuers (two drones
and the rod) go to blue goals, the evader to red goals, each with its high-level and low-level policies. A side that
crashes respawns on its own.

    python visualize_nav.py --pursuer logs/nav_pursuer/SRT_general_check/model_79.pt \
                            --evader logs/nav/SRT_a300_check/model_149.pt --t 60
"""
import os
import pickle
import argparse
import torch
from tensordict import TensorDict
from rsl_rl.utils import resolve_class

import genesis as gs


def load_cfgs(model_path):
    with open(os.path.join(os.path.dirname(model_path), "cfgs.pkl"), "rb") as f:
        return pickle.load(f)


def load_actor(model_path, obs, num_actions):
    train_cfg = load_cfgs(model_path)[5]
    actor_class, actor_cfg = resolve_class(train_cfg["actor"])
    actor = actor_class(obs, train_cfg["obs_groups"], "actor", num_actions, **actor_cfg).to(gs.device)
    actor.load_state_dict(torch.load(model_path, map_location=gs.device, weights_only=False)["actor_state_dict"])
    return actor.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pursuer", default="logs/nav_pursuer/SRT_general_check/model_79.pt")
    parser.add_argument("--evader", default="logs/nav/SRT_a300_check/model_149.pt")
    parser.add_argument("--t", type=float, default=60.0, help="[s]")
    args = parser.parse_args()

    gs.init(logging_level="warning")
    from src.env_nav import NavPursuerEnv
    from src.quadrotor import Quadrotor
    from utils.design_informed_dr import DesignInformedDR

    class DuoEnv(NavPursuerEnv):
        """nav_pursuer with an evader drone added to the scene, flown by its own nav and low-level policies."""

        def __init__(self, env_cfg, obs_cfg, reward_cfg, command_cfg, evader_path):
            _, self.ev_cfg, _, _, self.ev_command_cfg, _ = load_cfgs(evader_path)
            _, ev_low_env, self.ev_obs_cfg, _, _, _ = load_cfgs(self.ev_cfg["low_level"])
            assert self.ev_cfg["decimation"] == env_cfg["decimation"]
            assert ev_low_env["dt"] == load_cfgs(env_cfg["low_level"])[1]["dt"]
            dr = DesignInformedDR(ev_low_env["drone"], ev_low_env["domain_rand"], 1)
            self.ev = Quadrotor(ev_low_env["drone"], ev_low_env, 1, ev_low_env["dt"], dr)

            def buf(*shape):
                return torch.zeros((1, *shape), device=gs.device, dtype=gs.tc_float)

            self.ev_goal, self.ev_waypoint, self.ev_last_actions, self.ev_params = buf(3), buf(3), buf(4), buf(17)
            self.ev_hist = buf(self.ev_obs_cfg["history"], 17)
            self.ev_markers = {}
            self.all_envs = torch.arange(1, device=gs.device)

            # the evader drone joins the scene the pursuer env builds
            build = gs.Scene.build

            def build_with_evader(scene, *a, **k):
                self.ev.add_to(scene, ev_low_env["drone"]["urdf"], (0.0, 0.0, 1.0))
                build(scene, *a, **k)
                self.ev.build()

            gs.Scene.build = build_with_evader
            super().__init__(1, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True)
            gs.Scene.build = build

            self.ev_low = load_actor(self.ev_cfg["low_level"], self._ev_low_obs(), 4)
            self.ev_nav = load_actor(evader_path, self._ev_nav_obs(), 3)

        # ------------ markers ----------------
        def _marker(self, name, pos, radius, color):
            if name in self.ev_markers:
                self.scene.clear_debug_object(self.ev_markers[name])
            self.ev_markers[name] = self.scene.draw_debug_spheres(pos, radius=radius, color=color)

        def _resample_commands(self, envs_idx):
            self.goal[envs_idx] = self._sample_region(len(envs_idx))
            self._marker("pursuer_goal", self.goal, 0.15, (0.0, 0.3, 1.0, 1.0))  # blue

        # ------------ evader ----------------
        def _ev_low_obs(self):
            return TensorDict({"policy": self.ev_hist.reshape(1, -1), "params": self.ev_params}, batch_size=[1])

        def _ev_nav_obs(self):
            norm = lambda p: (p - self.box_center) / self.box_half
            return TensorDict({"policy": torch.cat([norm(self.ev.pos), norm(self.ev_goal)], dim=-1)}, batch_size=[1])

        def _ev_observe(self):
            """The robust_goto frame: rel_pos, quat, body lin / ang vel, last actions, with its observation noise."""
            s, noise = self.ev_obs_cfg["obs_scales"], self.ev_obs_cfg.get("obs_noise", {})
            noisy = lambda x, key: x + noise.get(key, 0.0) * torch.randn_like(x)
            quat = noisy(self.ev.quat, "quat")
            frame = torch.cat(
                [
                    torch.clip(noisy(self.ev_waypoint - self.ev.pos, "rel_pos") * s["rel_pos"], -1, 1),
                    quat / quat.norm(dim=1, keepdim=True),
                    torch.clip(noisy(self.ev.lin_vel, "lin_vel") * s["lin_vel"], -1, 1),
                    torch.clip(noisy(self.ev.ang_vel, "ang_vel") * s["ang_vel"], -1, 1),
                    self.ev_last_actions,
                ],
                dim=-1,
            )
            self.ev_hist = torch.roll(self.ev_hist, 1, dims=1)
            self.ev_hist[:, 0] = frame

        def _ev_sample(self):
            """A point of the evader's goal / spawn region."""
            c = self.ev_command_cfg
            lo, hi = (torch.tensor([c[k][i] for k in ("pos_x_range", "pos_y_range", "pos_z_range")], device=gs.device)
                      for i in (0, 1))
            return lo + (hi - lo) * torch.rand((1, 3), device=gs.device)

        def _ev_new_goal(self):
            self.ev_goal[:] = self._ev_sample()
            self._marker("evader_goal", self.ev_goal, 0.15, (1.0, 0.0, 0.0, 1.0))  # red

        def _ev_reset(self):
            spawn = self._ev_sample()
            self._ev_new_goal()
            e = self.ev.entity
            e.set_pos(spawn, zero_velocity=True, envs_idx=self.all_envs)
            e.set_quat(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=gs.device), zero_velocity=True, envs_idx=self.all_envs)
            self.ev.reset_idx(self.all_envs)
            self.ev.update_state()
            self.ev_params[:] = self._drone_params(self.ev.domain_rand, self.all_envs)
            self.ev_waypoint[:] = spawn
            self.ev_last_actions[:] = 0.0
            self._ev_observe()
            self.ev_hist[:] = self.ev_hist[:, :1]  # filled with the first frame

        def reset(self):
            obs = super().reset()
            self._ev_reset()
            return obs

        def step(self, actions):
            with torch.no_grad():
                a = torch.clip(self.ev_nav(self._ev_nav_obs()), -1.0, 1.0)
            self.ev_waypoint[:] = self.ev.pos + a * self.ev_cfg["waypoint_range"]
            self._marker("evader_waypoint", self.ev_waypoint, 0.05, (1.0, 0.5, 0.5, 1.0))
            return super().step(actions)

        def _sim_step(self, actions):
            with torch.no_grad():
                self.ev_last_actions[:] = torch.clip(self.ev_low(self._ev_low_obs()), 0.0, 1.0)
            self.ev.apply(self.ev_last_actions)
            super()._sim_step(actions)  # scene step
            self.ev.update_state()
            self._ev_observe()

            pos = self.ev.pos
            if torch.norm(self.ev_goal - pos) < self.ev_cfg["goal_threshold"]:
                self._ev_new_goal()
            if (pos[0, 2] < self.ev_cfg["termination_if_close_to_ground"]
                    or (pos < self.box_lo).any() or (pos > self.box_hi).any()):
                self._ev_reset()

    _, env_cfg, obs_cfg, reward_cfg, command_cfg, _ = load_cfgs(args.pursuer)
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = False
    env_cfg["episode_length_s"] = args.t
    env = DuoEnv(env_cfg, obs_cfg, reward_cfg, command_cfg, args.evader)
    pursuer_nav = load_actor(args.pursuer, env.get_observations(), 3)

    obs = env.reset()
    with torch.no_grad():
        for _ in range(int(args.t / (env.dt * env.decimation))):
            obs, _, _, _ = env.step(pursuer_nav(obs))


if __name__ == "__main__":
    main()
