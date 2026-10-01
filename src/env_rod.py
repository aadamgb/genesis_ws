import copy
import math

import torch
from tensordict import TensorDict

import genesis as gs

from src.quadrotor import Quadrotor


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class RodEnv:
    """Two drones carry a rod, hanging from their ropes, and bring its center to a target (the sprind task) with the
    actuation and noise of the goto task. One policy commands the 8 motors: [drone 0, drone 1]."""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device
        self.dt = env_cfg["dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        drone_cfg = env_cfg["drone"]
        if "rope_urdf" not in drone_cfg:
            raise ValueError(f"drone {drone_cfg['name']} has no rope_urdf for the rod task")
        self.drones = [Quadrotor(drone_cfg, env_cfg["domain_rand"], env_cfg, num_envs, self.dt) for _ in range(2)]

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=env_cfg["substeps"]),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(min(10, num_envs)))),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                batch_links_info=any(d.domain_rand.batch_links_info for d in self.drones),
            ),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())

        if env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.05, fixed=False, collision=False),
                surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.5, 1.0, 0.5))),
            )
        else:
            self.target = None

        # spawn: the drones side by side along x, the rod hanging below them from the rope tips
        rod = env_cfg["rod"]
        half = rod["drone_spacing"] / 2.0
        z = env_cfg["spawn_height"]
        self.drone_init_pos = [
            torch.tensor((-half, 0.0, z), device=gs.device),
            torch.tensor((half, 0.0, z), device=gs.device),
        ]
        self.rod_init_pos = torch.tensor((0.0, 0.0, z - rod["hang"]), device=gs.device)
        self.rod_init_quat = torch.tensor((0.7071068, 0.0, 0.7071068, 0.0), device=gs.device)  # axis along x
        self.init_quat = torch.tensor((1.0, 0.0, 0.0, 0.0), device=gs.device)

        for drone, pos in zip(self.drones, self.drone_init_pos):
            drone.add_to(self.scene, drone_cfg["rope_urdf"], tuple(pos.tolist()))
        volume = math.pi * rod["radius"] ** 2 * rod["length"]
        self.rod = self.scene.add_entity(
            morph=gs.morphs.Cylinder(
                radius=rod["radius"], height=rod["length"], pos=tuple(self.rod_init_pos.tolist()),
                quat=tuple(self.rod_init_quat.tolist()), collision=False,
            ),
            material=gs.materials.Rigid(rho=rod["mass"] / volume),
            surface=gs.surfaces.Default(color=(0.1, 0.1, 0.1, 1.0)),
        )

        self.scene.build(n_envs=num_envs)
        for drone in self.drones:
            drone.build()
            # the weld keeps the relative pose of the rope tip and the rod at this moment
            self.scene.sim.rigid_solver.add_weld_constraint(drone.entity.get_link(rod["tip_link"]).idx, self.rod.base_link.idx)
        self.rope_dofs = torch.arange(6, self.drones[0].entity.n_dofs, device=gs.device)
        # actions in [-1, 1] around hover for SRTHover, motor commands in [0, 1] otherwise
        hover_centered = hasattr(self.drones[0].controller, "hover_u")
        self.action_clip = env_cfg.get("action_clip", [-1.0, 1.0] if hover_centered else [0.0, 1.0])

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((num_envs,), device=gs.device, dtype=gs.tc_float)

        def buf(*shape, dtype=gs.tc_float):
            return torch.zeros((num_envs, *shape), device=gs.device, dtype=dtype)

        self.rew_buf = buf()
        self.reset_buf = torch.ones((num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = buf(dtype=gs.tc_int)
        self.commands = buf(self.num_commands)
        self.actions = buf(self.num_actions)
        self.last_actions = buf(self.num_actions)
        self.rod_pos = buf(3)
        self.last_rod_pos = buf(3)
        self.rod_quat = buf(4)
        self.rod_vel = buf(3)  # world frame
        self.rel_pos = buf(3)
        self.last_rel_pos = buf(3)
        self.crash_condition = torch.zeros((num_envs,), device=gs.device, dtype=torch.bool)

        # initial state: the whole formation is offset and moving (uniform half-ranges, world frame)
        self.init_pos_offset = torch.tensor(env_cfg.get("init_pos_offset", [0.0, 0.0, 0.0]), device=gs.device)
        self.init_lin_vel = env_cfg.get("init_lin_vel", 0.0)

        # gaussian observation noise in physical units (before scaling), e.g. {"ang_vel": 0.1}
        self.obs_noise = obs_cfg.get("obs_noise", {})

        # observation history: the current frame followed by the previous history-1 frames (newest first)
        self.obs_history = obs_cfg.get("history", 1)
        self.frame_size = 10 + 2 * 13 + self.num_actions
        self.obs_hist = buf(self.obs_history, self.frame_size)
        self.obs_hist_empty = torch.ones((num_envs,), device=gs.device, dtype=torch.bool)

        self.extras = dict()
        self.reset()

    def _resample_commands(self, envs_idx):
        n = (len(envs_idx),)
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], n, gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], n, gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], n, gs.device)

    def step(self, actions):
        self.actions = torch.clip(actions, *self.action_clip)
        for i, drone in enumerate(self.drones):
            drone.apply(self.actions[:, 4 * i : 4 * i + 4])

        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)
        self.scene.step()

        self.episode_length_buf += 1
        for drone in self.drones:
            drone.update_state()
        self.last_rod_pos[:] = self.rod_pos
        self._update_rod()

        at_target = (self.rel_pos.norm(dim=1) < self.env_cfg["at_target_threshold"]).nonzero(as_tuple=False).reshape(-1)
        self._resample_commands(at_target)

        cfg = self.env_cfg
        separation = (self.drones[0].pos - self.drones[1].pos).norm(dim=1)
        self.crash_condition = (
            (self.rel_pos[:, :2].abs() > cfg["termination_if_xy_greater_than"]).any(dim=1)
            | (self.rel_pos[:, 2].abs() > cfg["termination_if_z_greater_than"])
            | (separation < cfg["termination_if_drones_closer_than"])
            | self.scene.rigid_solver.get_error_envs_mask()
        )
        for drone in self.drones:
            self.crash_condition |= (
                (drone.euler[:, :2].abs() > cfg["termination_if_tilt_greater_than"]).any(dim=1)
                | (drone.pos[:, 2] < cfg["termination_if_close_to_ground"])
            )
        timed_out = self.episode_length_buf > self.max_episode_length
        self.reset_buf = timed_out | self.crash_condition
        self.extras["time_outs"] = timed_out.to(gs.tc_float)

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape(-1))

        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # the observation carries the action just applied; a new episode starts from zero
        self.last_actions[:] = self.actions
        self.last_actions[self.reset_buf.bool()] = 0.0

        self._update_observation()
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def _update_rod(self):
        self.rod_pos[:] = self.rod.get_pos()
        self.rod_quat[:] = self.rod.get_quat()
        self.rod_vel[:] = self.rod.get_vel()
        self.rel_pos[:] = self.commands - self.rod_pos
        self.last_rel_pos[:] = self.commands - self.last_rod_pos

    def _noisy(self, x, key):
        return x + self.obs_noise.get(key, 0.0) * torch.randn_like(x) if self.obs_noise else x

    def _update_observation(self):
        s = self.obs_scales
        clip = lambda x, scale: torch.clip(x * scale, -1, 1)

        def unit(q):
            return q / q.norm(dim=1, keepdim=True)

        parts = [
            clip(self._noisy(self.rel_pos, "rel_pos"), s["rel_pos"]),
            unit(self._noisy(self.rod_quat, "quat")),
            clip(self._noisy(self.rod_vel, "lin_vel"), s["lin_vel"]),
        ]
        for drone in self.drones:
            parts += [
                clip(self._noisy(self.commands - drone.pos, "rel_pos"), s["rel_pos"]),
                unit(self._noisy(drone.quat, "quat")),
                clip(self._noisy(drone.lin_vel, "lin_vel"), s["lin_vel"]),
                clip(self._noisy(drone.ang_vel, "ang_vel"), s["ang_vel"]),
            ]
        frame = torch.cat(parts + [self.last_actions], dim=-1)

        if self.obs_history > 1:
            self.obs_hist = torch.roll(self.obs_hist, 1, dims=1)
            self.obs_hist[:, 0] = frame
            empty = self.obs_hist_empty
            self.obs_hist[empty] = frame[empty].unsqueeze(1).expand(-1, self.obs_history, -1)
            self.obs_hist_empty[:] = False
            self.obs_buf = self.obs_hist.reshape(self.num_envs, -1)
        else:
            self.obs_buf = frame

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        n = len(envs_idx)

        # the formation in its welded configuration, offset and moving as one body
        offset = (2.0 * torch.rand((n, 3), device=gs.device) - 1.0) * self.init_pos_offset
        lin_vel = gs_rand_float(-self.init_lin_vel, self.init_lin_vel, (n, 3), gs.device)
        zeros = torch.zeros((n, 3), device=gs.device)
        for drone, pos in zip(self.drones, self.drone_init_pos):
            e = drone.entity
            e.set_pos(pos + offset, zero_velocity=True, envs_idx=envs_idx)
            e.set_quat(self.init_quat.expand(n, -1), zero_velocity=True, envs_idx=envs_idx)
            e.set_dofs_position(torch.zeros((n, len(self.rope_dofs)), device=gs.device), dofs_idx_local=self.rope_dofs, envs_idx=envs_idx)
            e.zero_all_dofs_velocity(envs_idx)
            e.set_dofs_velocity(torch.cat([lin_vel, zeros], dim=1), dofs_idx_local=list(range(6)), envs_idx=envs_idx)
            drone.reset_idx(envs_idx)
        self.rod.set_pos(self.rod_init_pos + offset, zero_velocity=True, envs_idx=envs_idx)
        self.rod.set_quat(self.rod_init_quat.expand(n, -1), zero_velocity=True, envs_idx=envs_idx)
        self.rod.set_dofs_velocity(torch.cat([lin_vel, zeros], dim=1), envs_idx=envs_idx)
        for drone in self.drones:
            drone.update_state()

        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.obs_hist_empty[envs_idx] = True

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)
        self.rod_pos[envs_idx] = self.rod_init_pos + offset
        self.last_rod_pos[envs_idx] = self.rod_pos[envs_idx]
        self._update_rod()

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        self._update_observation()
        return self.get_observations()

    # ------------ reward functions ----------------
    def _reward_target(self):
        # progress of the rod center towards the target
        return self.last_rel_pos.square().sum(dim=1) - self.rel_pos.square().sum(dim=1)

    def _reward_smooth(self):
        return (self.actions - self.last_actions).square().sum(dim=1)

    def _reward_angular(self):
        return sum((d.ang_vel / math.pi).norm(dim=1) for d in self.drones)

    def _reward_tilt(self):
        # 1 - cos(tilt) = 2 (qx^2 + qy^2), summed over the drones
        return sum(2.0 * (d.quat[:, 1].square() + d.quat[:, 2].square()) for d in self.drones)

    def _reward_crash(self):
        return self.crash_condition.to(gs.tc_float)
