"""pursuer (was rod2 on the rod branch): two general drones carry a rod with a net, hanging from their ropes, and bring
its center to a target (hydra_configs/task/pursuer.yaml).

    drones   design-informed (utils/design_informed_dr.py), sampled once per env: both drones of an env share the
             design (size c and the size-independent draws), each gets its own noise
    rod      length, mass and radius sampled once per env (log-uniform); the drone spacing follows the length and the
             max rpm are raised so that the max thrust of both drones is min_system_twr x the weight of the system
    policy   direct rotor commands in [0, 1] (SRT); a "params" observation group (drones and rod) for an encoder
    starts   every drone tilted with body rates, upset.fraction of the episodes with the upset ranges; the ropes are
             bent so that the rod stays in place
    crash    rod tilt from the horizontal, drone tilt is permissive
"""
import copy
import math

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import transform_by_quat

from src.quadrotor import Quadrotor
from utils.design_informed_dr import DesignInformedDR


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class PursuerEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device
        self.dt = env_cfg["dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.cfg = env_cfg  # read by rsl_rl's runner
        self.rendered_envs = list(range(min(10, num_envs)))
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        # both drones of an env from the same design, each with its own noise
        drone_cfg = env_cfg["drone"]
        drs = [DesignInformedDR(drone_cfg, env_cfg["domain_rand"], num_envs) for _ in range(2)]
        shared = drs[0].draw(num_envs)
        for dr in drs:
            dr.shared = shared
        self.drones = [Quadrotor(drone_cfg, env_cfg, num_envs, self.dt, dr) for dr in drs]

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=env_cfg["substeps"]),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=env_cfg["max_visualize_FPS"],
                camera_pos=(5.0, 0.0, 4.0),
                camera_lookat=(0.0, 0.0, 2.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=self.rendered_envs),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                batch_links_info=True,  # per-env drones and rod
            ),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())

        # spawn: the drones side by side along x, the rod hanging below them from the rope tips
        rod = env_cfg["rod"]
        z = env_cfg["spawn_height"]
        self.rod_init_pos = torch.tensor((0.0, 0.0, z - rod["hang"]), device=gs.device)
        self.rod_init_quat = torch.tensor((0.7071068, 0.0, 0.7071068, 0.0), device=gs.device)  # axis along x
        self.rod_ref = {k: math.sqrt(rod[k][0] * rod[k][1]) for k in ("length", "mass", "radius")}
        for drone in self.drones:
            drone.add_to(self.scene, drone_cfg["rope_urdf"], (0.0, 0.0, z))
        # mass and inertia are set per env after the build; the geometry is only a stub, drawn over by _draw
        self.rod = self.scene.add_entity(
            morph=gs.morphs.Cylinder(
                radius=0.005, height=0.01, pos=tuple(self.rod_init_pos.tolist()),
                quat=tuple(self.rod_init_quat.tolist()), collision=False,
            ),
            surface=gs.surfaces.Default(color=(0.1, 0.1, 0.1, 1.0)),
        )

        self.visualize = env_cfg["visualize_target"]
        if self.visualize:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.05, fixed=False, collision=False),
                surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.5, 1.0, 0.5))),
            )
            # visual only: the net hangs from the rod (the cloth does not act on the rigid bodies)
            half = self.rod_ref["length"] / 2.0
            # self.net = self.scene.add_entity(
            #     material=gs.materials.PBD.Cloth(),
            #     morph=gs.morphs.Mesh(
            #         file="utils/models/net.obj", scale=half, euler=(180.0, 0.0, 0.0),
            #         pos=tuple((self.rod_init_pos - torch.tensor((0.0, 0.0, half), device=gs.device)).tolist()),
            #     ),
            #     surface=gs.surfaces.Default(color=(0.2, 0.6, 0.2, 1.0)),
            # )

        self.scene.build(n_envs=num_envs)
        for drone in self.drones:
            drone.build()
        self._sample_rod(rod)
        self._limit_twr(env_cfg["min_system_twr"])

        # rope: dofs after the free joint, the first two are the y and x hinges at the drone
        e = self.drones[0].entity
        self.rope_dofs = torch.arange(6, e.n_dofs, device=gs.device)
        self.rope_anchor = (e.get_link("segment_1_sphere").get_pos() - e.get_pos())[0]  # body frame, drone level
        self.init_quat = torch.tensor((1.0, 0.0, 0.0, 0.0), device=gs.device)
        self.init_pos_offset = torch.tensor(env_cfg["init_pos_offset"], device=gs.device)
        self.init = env_cfg["init"]
        self.upset = env_cfg["upset"]

        # the welds keep the relative pose of the rope tips and the rod of each env at this moment
        all_envs = torch.arange(num_envs, device=gs.device)
        self._place(all_envs, torch.zeros(num_envs, device=gs.device, dtype=torch.bool), random=False)
        for drone in self.drones:
            self.scene.sim.rigid_solver.add_weld_constraint(drone.entity.get_link(rod["tip_link"]).idx, self.rod.base_link.idx)
        # if self.visualize:
        #     P = self.net.get_particles_pos()[0]
        #     top = torch.where(P[:, 2] > P[:, 2].max() - 0.02)[0]
        #     self.net.fix_particles_to_link(self.rod.base_link.idx, particles_idx_local=top)
        #     self.net_init_pos = P

        self.action_clip = [0.0, 1.0]
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
        self.rod_axis = torch.tensor((0.0, 0.0, 1.0), device=gs.device).expand(num_envs, -1)  # cylinder axis, local
        self.max_rod_axis_z = math.sin(math.radians(env_cfg["termination_if_rod_tilt_greater_than"]))
        arms = sum(d.domain_rand.arm.mean(dim=1) for d in self.drones) / 2.0
        self.min_separation = env_cfg["termination_if_drones_closer_than"] * arms

        # gaussian observation noise in physical units (before scaling), e.g. {"ang_vel": 0.1}
        self.obs_noise = obs_cfg.get("obs_noise", {})

        # observation history: the current frame followed by the previous history-1 frames (newest first)
        self.obs_history = obs_cfg.get("history", 1)
        self.frame_size = 10 + 2 * 13 + self.num_actions
        self.obs_hist = buf(self.obs_history, self.frame_size)
        self.obs_hist_empty = torch.ones((num_envs,), device=gs.device, dtype=torch.bool)

        # encoder input, set at every reset: per drone 17 (_drone_params), rod length, mass, radius
        self.params_buf = buf(2 * 17 + 3)

        self.extras = dict()
        self.reset()

    # ------------ drones and rod ----------------
    def _sample_rod(self, rod):
        """Length, mass and radius per env, log-uniform; the length is raised to fit the drones."""
        n = self.num_envs

        def log_uniform(lo, hi):
            return lo * (hi / lo) ** torch.rand(n, device=gs.device, dtype=gs.tc_float)

        arm = torch.maximum(*(d.domain_rand.arm.mean(dim=1) for d in self.drones))
        self.rod_length = torch.maximum(log_uniform(*rod["length"]), rod["min_spacing"] * arm + 2.0 * rod["tip_inset"])
        self.rod_mass = log_uniform(*rod["mass"])
        self.rod_radius = log_uniform(*rod["radius"])
        self.half_spacing = self.rod_length / 2.0 - rod["tip_inset"]

        # solid cylinder about its axis (z) and a diameter
        m, r, l = self.rod_mass, self.rod_radius, self.rod_length
        j_d = m * (3.0 * r**2 + l**2) / 12.0
        inertia = torch.diag_embed(torch.stack([j_d, j_d, m * r**2 / 2.0], dim=1))
        all_envs = torch.arange(n, device=gs.device)
        self.rod.set_links_mass(m[:, None], [0], all_envs)
        self.rod.set_links_inertia(inertia[:, None], [0], all_envs)

    def _limit_twr(self, min_twr):
        """Raise the max rpm of both drones where their max thrust is below min_twr x the weight of drones + rod."""
        weight = 9.81 * (self.rod_mass + sum(d.entity.get_links_mass().sum(dim=-1) for d in self.drones))
        thrust = sum(4.0 * d.domain_rand.kf[:, 0] * d.domain_rand.max_rpm[:, 0] ** 2 for d in self.drones)
        scale = torch.sqrt((min_twr * weight / thrust).clamp(min=1.0))
        for d in self.drones:
            d.domain_rand.max_rpm *= scale[:, None]

    def _drone_params(self, dr, envs_idx):
        """17 parameters: log ratios to the middle design, the per-episode ones scaled to ~[-1, 1]."""
        log_ratio = lambda name, x: torch.log(x / dr.nominal(name))
        return torch.cat(
            [
                log_ratio("mass", dr.mass[envs_idx, None]),
                log_ratio("inertia", dr.inertia[envs_idx]),
                log_ratio("arm", dr.arm[envs_idx]),
                log_ratio("kf", dr.kf[envs_idx]),
                log_ratio("max_rpm", dr.max_rpm[envs_idx]),
                log_ratio("motor_tau", dr.motor_tau[envs_idx]),
                (dr.motor_eff[envs_idx] - 1.0) * 10.0,
                dr.action_delay[envs_idx, None] / 0.02,
            ],
            dim=-1,
        )

    def _update_params(self, envs_idx):
        rod = [torch.log(getattr(self, "rod_" + k)[envs_idx] / ref)[:, None] for k, ref in self.rod_ref.items()]
        self.params_buf[envs_idx] = torch.cat([self._drone_params(d.domain_rand, envs_idx) for d in self.drones] + rod, dim=-1)

    def _place(self, envs_idx, upset, random=True):
        """The formation of envs_idx at its spawn, offset and moving as one body (random); each drone tilted
        (roll, pitch) with body rates, its rope bent back so that the tip and the rod keep their welded pose."""
        n = len(envs_idx)
        zeros = torch.zeros((n, 3), device=gs.device)
        if random:
            rng = lambda key: torch.where(upset, self.upset[key], self.init[key])[:, None]
            offset = (2.0 * torch.rand((n, 3), device=gs.device) - 1.0) * self.init_pos_offset
            lin_vel = rng("lin_vel") * (2.0 * torch.rand((n, 3), device=gs.device) - 1.0)
        else:
            offset, lin_vel = zeros, zeros
        for i, drone in enumerate(self.drones):
            pos = self.rod_init_pos + offset
            pos[:, 2] = self.env_cfg["spawn_height"] + offset[:, 2]
            pos[:, 0] += (2 * i - 1) * self.half_spacing[envs_idx]
            quat = self.init_quat.expand(n, -1)
            rope = torch.zeros((n, len(self.rope_dofs)), device=gs.device)
            ang_vel = zeros
            if random:
                # R = Rx(roll) Ry(pitch), the rope hinges Ry(-pitch) Rx(-roll) undo it
                roll, pitch = (math.radians(1.0) * rng("tilt")[:, 0] * (2.0 * torch.rand(n, device=gs.device) - 1.0) for _ in range(2))
                cr, sr, cp, sp = torch.cos(roll / 2), torch.sin(roll / 2), torch.cos(pitch / 2), torch.sin(pitch / 2)
                quat = torch.stack([cr * cp, sr * cp, cr * sp, sr * sp], dim=1)
                rope[:, 0], rope[:, 1] = -pitch, -roll
                pos = pos + self.rope_anchor - transform_by_quat(self.rope_anchor.expand(n, -1), quat)
                ang_vel = rng("ang_vel") * (2.0 * torch.rand((n, 3), device=gs.device) - 1.0)
            e = drone.entity
            e.set_pos(pos, zero_velocity=True, envs_idx=envs_idx)
            e.set_quat(quat, zero_velocity=True, envs_idx=envs_idx)
            e.set_dofs_position(rope, dofs_idx_local=self.rope_dofs, envs_idx=envs_idx)
            e.zero_all_dofs_velocity(envs_idx)
            e.set_dofs_velocity(torch.cat([lin_vel, ang_vel], dim=1), dofs_idx_local=list(range(6)), envs_idx=envs_idx)
        self.rod.set_pos(self.rod_init_pos + offset, zero_velocity=True, envs_idx=envs_idx)
        self.rod.set_quat(self.rod_init_quat.expand(n, -1), zero_velocity=True, envs_idx=envs_idx)
        self.rod.set_dofs_velocity(torch.cat([lin_vel, zeros], dim=1), envs_idx=envs_idx)
        # if self.visualize and random:
        #     self.net.set_particles_pos(self.net_init_pos + offset[:, None], envs_idx=envs_idx)
        return offset

    # ------------ env ----------------
    def _resample_commands(self, envs_idx):
        n = (len(envs_idx),)
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], n, gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], n, gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], n, gs.device)

    def step(self, actions):
        self.actions = torch.clip(actions, *self.action_clip)
        for i, drone in enumerate(self.drones):
            drone.apply(self.actions[:, 4 * i : 4 * i + 4])

        if self.visualize:
            self.target.set_pos(self.commands, zero_velocity=True)
        self.scene.step()

        self.episode_length_buf += 1
        for drone in self.drones:
            drone.update_state()
        self.last_rod_pos[:] = self.rod_pos
        self._update_rod()
        if self.visualize:
            self._draw()

        at_target = (self.rel_pos.norm(dim=1) < self.env_cfg["at_target_threshold"]).nonzero(as_tuple=False).reshape(-1)
        self._resample_commands(at_target)

        cfg = self.env_cfg
        separation = (self.drones[0].pos - self.drones[1].pos).norm(dim=1)
        rod_axis_z = transform_by_quat(self.rod_axis, self.rod_quat)[:, 2]
        self.crash_condition = (
            (self.rel_pos[:, :2].abs() > cfg["termination_if_xy_greater_than"]).any(dim=1)
            | (self.rel_pos[:, 2].abs() > cfg["termination_if_z_greater_than"])
            | (rod_axis_z.abs() > self.max_rod_axis_z)
            | (self.rod_pos[:, 2] < cfg["termination_if_close_to_ground"])
            | (separation < self.min_separation)
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

    def _draw(self):
        """The rod of the rendered envs at its sampled length and radius, and its center (red)."""
        self.scene.clear_debug_objects()
        envs = self.rendered_envs
        pos = self.rod_pos[envs]
        half = transform_by_quat(self.rod_axis[envs], self.rod_quat[envs]) * self.rod_length[envs, None] / 2.0
        for a, b, r in zip((pos - half).tolist(), (pos + half).tolist(), self.rod_radius[envs].tolist()):
            self.scene.draw_debug_line(a, b, radius=r, color=(0.1, 0.1, 0.1, 1.0))
        self.scene.draw_debug_spheres(pos.cpu().numpy(), radius=0.03, color=(1.0, 0.1, 0.1, 1.0))

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
        return TensorDict({"policy": self.obs_buf, "params": self.params_buf}, batch_size=[self.num_envs])

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        upset = torch.rand(len(envs_idx), device=gs.device) < self.upset["fraction"]
        offset = self._place(envs_idx, upset)
        for drone in self.drones:
            drone.update_state()
            drone.reset_idx(envs_idx)
        self._update_params(envs_idx)

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

    def _reward_crash(self):
        return self.crash_condition.to(gs.tc_float)
