"""pursuit_evasion: two learning agents in one scene (hydra_configs/task/pursuit_evasion.yaml, trained by
train_marl.py), each a high-level MLP giving waypoints to its frozen low-level policy, as in env_nav.py:

    pursuer  the pursuer system (two drones carrying a rod with a net, env.pursuer_low_level); its waypoint is for the
             rod center. It starts at its home in the hemisphere and holds it until the evader enters the hemisphere,
             then chases: rewarded for closing in on the evader and for catching it.
    evader   one drone (env.evader_low_level), spawned outside the hemisphere; greedy for now: rewarded for progress
             to its target inside the hemisphere and for reaching it, penalized when caught.

Catch: the evader crosses the net, the vertical rectangle hanging below the rod (rod length x net_depth). The episode
ends on a catch, the evader at its target, a crash of either side (a body on the ground or outside the safety box, a
pursuer failure) or at the time limit; each side is penalized for its own crash only.

Observations (normalized; velocities world frame): "evader" and "pursuer" for the actors, "critic" the joint state
for both critics (centralized critic, MAPPO). The step takes both actions and returns {agent: reward}.

Visualization: hemisphere (white wireframe, commented out), evader target (yellow), net outline (green), the pursuer
waypoint (blue, low-level target) and the evader one (red).
"""
import math
import os
import pickle
import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import transform_by_quat
from src.env_nav import set_level_camera
from src.env_pursuer import PursuerEnv
from src.models import load_policy
from src.quadrotor import Quadrotor
from utils.design_informed_dr import DesignInformedDR

AGENTS = ("pursuer", "evader")


def load_cfgs(model_path):
    with open(os.path.join(os.path.dirname(model_path), "cfgs.pkl"), "rb") as f:
        return pickle.load(f)


class PursuitEvasionEnv(PursuerEnv):
    target_color = (0.0, 0.3, 1.0)  # pursuer waypoint

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        _, p_env_cfg, p_obs_cfg, _, _, _ = load_cfgs(env_cfg["pursuer_low_level"])
        _, self.ev_env_cfg, self.ev_obs_cfg, _, _, _ = load_cfgs(env_cfg["evader_low_level"])
        assert self.ev_env_cfg["dt"] == p_env_cfg["dt"] and not self.ev_obs_cfg.get("obs_bias")
        env_cfg = {**p_env_cfg, **env_cfg}

        def buf(*shape, dtype=gs.tc_float):
            return torch.zeros((num_envs, *shape), device=gs.device, dtype=dtype)

        # evader drone, added to the scene by _add_entities
        ev_drone = self.ev_env_cfg["drone"]
        self.ev = Quadrotor(ev_drone, self.ev_env_cfg, num_envs, self.ev_env_cfg["dt"],
                            DesignInformedDR(ev_drone, self.ev_env_cfg["domain_rand"], num_envs))
        self.ev_target, self.ev_waypoint, self.ev_vel, self.ev_last_pos = buf(3), buf(3), buf(3), buf(3)
        self.ev_last_actions, self.ev_params = buf(4), buf(17)
        self.ev_hist = buf(self.ev_obs_cfg["history"], 17)
        self.ev_hist_empty = torch.ones((num_envs,), device=gs.device, dtype=torch.bool)

        # game
        self.decimation = env_cfg["decimation"]
        self.waypoint_range = env_cfg["waypoint_range"]
        box = torch.tensor(env_cfg["safety_box"], device=gs.device, dtype=gs.tc_float)
        self.box_lo, self.box_hi = box[:, 0], box[:, 1]
        self.box_center, self.box_half = (self.box_hi + self.box_lo) / 2, (self.box_hi - self.box_lo) / 2
        h = env_cfg["hemisphere"]
        self.hemi_center = torch.tensor(h["center"], device=gs.device, dtype=gs.tc_float)
        self.hemi_radius = h["radius"]
        self.chase = buf(dtype=torch.bool)
        self.target_dist = buf()  # evader - its target
        self.gap = buf()          # evader - rod center
        self.markers = {}
        # reward scales per agent, per high-level step: events count 1, distances in m (home: m s)
        self.scales = {a: dict(reward_cfg[a]) for a in AGENTS}

        super().__init__(num_envs, env_cfg, p_obs_cfg, {"reward_scales": {}}, command_cfg, show_viewer)

        self.num_actions = 3  # per agent
        self.home = self.rod_init_pos  # the pursuer spawns and hovers here
        self.episode_sums = {f"{a}/{k}": buf() for a in AGENTS for k in self.scales[a]}
        self.reset()

        # frozen low-level policies
        self.p_low = load_policy(env_cfg["pursuer_low_level"], PursuerEnv.get_observations(self), p_env_cfg["num_actions"])
        self.ev_low = load_policy(env_cfg["evader_low_level"], self._ev_low_obs(), 4)

        # if self.visualize:
            # self.scene.draw_debug_box(box.T, color=(1.0, 0.0, 0.0, 0.4), wireframe_radius=0.02)  # safety box
            # self._draw_hemisphere()
        set_level_camera(self, show_viewer, self.box_center.tolist(), 16.0)

    # ------------ scene ----------------
    def _add_entities(self):
        self.ev.add_to(self.scene, self.ev_env_cfg["drone"]["urdf"], (0.0, 0.0, 1.0))

    def _built(self):
        self.ev.build()

    def _resample_commands(self, envs_idx):
        """The pursuer waypoint (self.commands) starts at home."""
        self.commands[envs_idx] = self.rod_init_pos

    # ------------ evader ----------------
    def _ev_low_obs(self):
        return TensorDict({"policy": self.ev_hist.reshape(self.num_envs, -1), "params": self.ev_params},
                          batch_size=[self.num_envs])

    def _ev_observe(self):
        """The robust_goto frame (rel_pos, quat, body lin / ang vel, last actions) with its observation noise."""
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
        empty = self.ev_hist_empty
        self.ev_hist[empty] = frame[empty].unsqueeze(1).expand(-1, self.ev_hist.shape[1], -1)
        self.ev_hist_empty[:] = False

    def _uniform(self, lo, hi, n):
        return lo + (hi - lo) * torch.rand(n, device=gs.device, dtype=gs.tc_float)

    def _ev_reset(self, envs_idx):
        """Spawn outside the hemisphere (horizontal distance and height ranges), target inside (a cylinder around
        the hemisphere axis)."""
        n, c = len(envs_idx), self.hemi_center
        spawn, target = self.env_cfg["evader_spawn"], self.env_cfg["evader_target"]
        phi, d = self._uniform(0.0, 2.0 * math.pi, n), self._uniform(*spawn["distance"], n)
        pos = torch.stack([c[0] + d * torch.cos(phi), c[1] + d * torch.sin(phi), self._uniform(*spawn["z"], n)], dim=1)
        phi, r = self._uniform(0.0, 2.0 * math.pi, n), target["radius"] * torch.rand(n, device=gs.device).sqrt()
        self.ev_target[envs_idx] = torch.stack(
            [c[0] + r * torch.cos(phi), c[1] + r * torch.sin(phi), self._uniform(*target["z"], n)], dim=1
        )

        e = self.ev.entity
        e.set_pos(pos, zero_velocity=True, envs_idx=envs_idx)
        e.set_quat(torch.tensor((1.0, 0.0, 0.0, 0.0), device=gs.device).expand(n, -1), zero_velocity=True, envs_idx=envs_idx)
        self.ev.reset_idx(envs_idx)
        self.ev.update_state()
        self.ev_vel[envs_idx] = 0.0
        self.ev_last_pos[envs_idx] = pos
        self.ev_params[envs_idx] = self._drone_params(self.ev.domain_rand, envs_idx)
        self.ev_waypoint[envs_idx] = pos
        self.ev_last_actions[envs_idx] = 0.0
        self.ev_hist_empty[envs_idx] = True
        if self.visualize and (envs_idx < len(self.rendered_envs)).any():
            self._marker("evader_target", self.ev_target[self.rendered_envs], 0.15, (1.0, 1.0, 0.0, 1.0))

    # ------------ net ----------------
    def _net_frame(self):
        """Horizontal unit vectors along the rod (a) and normal to the net (n)."""
        a = transform_by_quat(self.rod_axis, self.rod_quat)
        a = torch.stack([a[:, 0], a[:, 1], torch.zeros_like(a[:, 0])], dim=1)
        a = a / a.norm(dim=1, keepdim=True).clamp(min=1e-6)
        n = torch.stack([-a[:, 1], a[:, 0], torch.zeros_like(a[:, 0])], dim=1)
        return a, n

    def _caught(self):
        """The evader segment of this step crosses the net: the vertical rectangle below the rod center, rod length
        wide and net_depth deep."""
        a, n = self._net_frame()
        p0, p1, c = self.ev_last_pos, self.ev.pos, self.rod_pos
        s0, s1 = ((p0 - c) * n).sum(dim=1), ((p1 - c) * n).sum(dim=1)
        crossed = (s0 * s1 <= 0.0) & (s0 != s1)
        x = p0 + (s0 / torch.where(crossed, s0 - s1, torch.ones_like(s0))).unsqueeze(1) * (p1 - p0)
        along, depth = ((x - c) * a).sum(dim=1).abs(), c[:, 2] - x[:, 2]
        return crossed & (along <= self.rod_length / 2.0) & (depth >= 0.0) & (depth <= self.env_cfg["net_depth"])

    # ------------ env ----------------
    def step(self, actions_pursuer, actions_evader):
        self.commands[:] = self.rod_pos + torch.clip(actions_pursuer, -1.0, 1.0) * self.waypoint_range
        self.ev_waypoint[:] = self.ev.pos + torch.clip(actions_evader, -1.0, 1.0) * self.waypoint_range
        self.rel_pos = self.commands - self.rod_pos
        if self.visualize:
            self._marker("evader_waypoint", self.ev_waypoint[self.rendered_envs], 0.05, (1.0, 0.0, 0.0, 1.0))

        cfg, dt = self.env_cfg, self.dt
        rew = {a: torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float) for a in AGENTS}
        outcomes = torch.zeros(4, device=gs.device)  # caught, reached, pursuer crash, evader crash
        done = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)
        for _ in range(self.decimation):
            self._update_observation()
            self._ev_observe()
            with torch.no_grad():
                p_actions = self.p_low(PursuerEnv.get_observations(self))
                self.ev_last_actions[:] = torch.clip(self.ev_low(self._ev_low_obs()), 0.0, 1.0)
            self.ev.apply(self.ev_last_actions)
            self.ev_last_pos[:] = self.ev.pos
            self._sim_step(p_actions)  # scene step, pursuer state
            self.last_actions[:] = self.actions
            self.ev.update_state()
            self.ev_vel[:] = self.ev.entity.get_vel()

            # the rest of the high-level step of a finished game is ignored
            live = ~done
            e, r = self.ev.pos, self.rod_pos
            target_dist, gap = (self.ev_target - e).norm(dim=1), (e - r).norm(dim=1)
            progress, closing = self.target_dist - target_dist, self.gap - gap
            self.target_dist, self.gap = target_dist, gap
            self.chase |= ((e - self.hemi_center).norm(dim=1) < self.hemi_radius) & live

            caught = self._caught() & live
            reached = (target_dist < cfg["goal_threshold"]) & live
            ground = cfg["termination_if_close_to_ground"]
            out = lambda p: (p[..., 2] < ground) | (p < self.box_lo).any(dim=-1) | (p > self.box_hi).any(dim=-1)
            p_bodies = torch.stack([r] + [d.pos for d in self.drones], dim=1)
            p_crash = (out(p_bodies).any(dim=1) | self._failure()) & live
            e_crash = out(e) & live

            terms = {
                "pursuer": {
                    "home": (r - self.home).norm(dim=1) * dt * ~self.chase,  # [m s] before the chase
                    "approach": closing * self.chase,                         # [m] during the chase
                    "catch": caught,
                    "evader_reached": reached,
                    "crash": p_crash,
                },
                "evader": {"progress": progress, "reach": reached, "caught": caught, "crash": e_crash},
            }
            for agent, agent_terms in terms.items():
                for name, term in agent_terms.items():
                    r_term = term.to(gs.tc_float) * live * self.scales[agent][name]
                    rew[agent] += r_term
                    self.episode_sums[f"{agent}/{name}"] += r_term
            outcomes += torch.stack([caught, reached, p_crash, e_crash]).sum(dim=1)
            done |= caught | reached | p_crash | e_crash

        time_out = (self.episode_length_buf > self.max_episode_length) & ~done
        self.reset_buf = done | time_out
        self.extras["time_outs"] = time_out.to(gs.tc_float)
        counts = torch.cat([outcomes, time_out.sum().unsqueeze(0)]).tolist()
        self.extras["outcomes"] = dict(zip(("caught", "reached", "pursuer_crash", "evader_crash", "time_out"), counts))
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape(-1))
        return self.get_observations(), rew, self.reset_buf, self.extras

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        super().reset_idx(envs_idx)
        self._ev_reset(envs_idx)
        self.chase[envs_idx] = False
        self.target_dist[envs_idx] = (self.ev_target[envs_idx] - self.ev.pos[envs_idx]).norm(dim=1)
        self.gap[envs_idx] = (self.ev.pos[envs_idx] - self.rod_pos[envs_idx]).norm(dim=1)

    def reset(self):
        if hasattr(self, "home"):  # not from PursuerEnv.__init__, before the game is set up
            return super().reset()

    def get_observations(self):
        norm = lambda p: (p - self.box_center) / self.box_half
        vel = lambda v: torch.clip(v * self.env_cfg["vel_scale"], -1.0, 1.0)
        a, n = self._net_frame()
        e, r, chase = self.ev.pos, self.rod_pos, self.chase.to(gs.tc_float).unsqueeze(1)
        rel = e - r  # evader in the net frame: along the rod, normal to the net, height
        net = torch.clip(torch.stack([(rel * a).sum(dim=1), (rel * n).sum(dim=1), rel[:, 2]], dim=1) * 0.25, -1.0, 1.0)
        rod = [norm(r), vel(self.rod_vel), a[:, :2]]
        evader = [norm(e), vel(self.ev_vel)]
        obs = {
            "pursuer": torch.cat(rod + evader + [chase], dim=-1),
            "evader": torch.cat(evader + [norm(self.ev_target)] + rod[:2], dim=-1),
            "critic": torch.cat(rod + evader + [norm(self.ev_target), chase, net], dim=-1),
        }
        return TensorDict(obs, batch_size=[self.num_envs])

    # ------------ visualization ----------------
    def _marker(self, name, pos, radius, color):
        if name in self.markers:
            self.scene.clear_debug_object(self.markers[name])
        self.markers[name] = self.scene.draw_debug_spheres(pos, radius=radius, color=color)

    def _draw_hemisphere(self, n_meridians=12, n_parallels=4, n_points=48):
        c, R, dev = self.hemi_center, self.hemi_radius, gs.device
        color = (1.0, 1.0, 1.0, 0.4)
        polar = torch.linspace(0.0, math.pi / 2, n_points, device=dev)  # from the top down to the ground
        for phi in torch.arange(n_meridians, device=dev) * 2.0 * math.pi / n_meridians:
            pts = torch.stack([R * polar.sin() * phi.cos(), R * polar.sin() * phi.sin(), R * polar.cos()], dim=1)
            self.scene.draw_debug_trajectory((pts + c).cpu().numpy(), radius=0.01, color=color)
        azimuth = torch.linspace(0.0, 2.0 * math.pi, 2 * n_points, device=dev)
        for k in range(n_parallels):
            theta = math.pi / 2 * (k + 1) / n_parallels  # polar angle, the last one on the ground
            pts = torch.stack([R * math.sin(theta) * azimuth.cos(), R * math.sin(theta) * azimuth.sin(),
                               torch.full_like(azimuth, R * math.cos(theta))], dim=1)
            self.scene.draw_debug_trajectory((pts + c).cpu().numpy(), radius=0.01, color=color)

    def _draw(self):
        """The rod (PursuerEnv) and the net outline of the rendered envs."""
        super()._draw()
        a, _ = self._net_frame()
        envs = self.rendered_envs
        c, half = self.rod_pos[envs], a[envs] * self.rod_length[envs, None] / 2.0
        down = torch.tensor((0.0, 0.0, self.env_cfg["net_depth"]), device=gs.device)
        for corners in zip((c - half).tolist(), (c + half).tolist(), (c + half - down).tolist(), (c - half - down).tolist()):
            pts = list(corners) + [corners[0]]
            self.debug_objs.append(self.scene.draw_debug_trajectory(pts, radius=0.008, color=(0.1, 0.7, 0.2, 0.8)))
