import torch
import math
import copy
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
    xyz_to_quat,
    quat_to_R
)

from src.controller import build_controller

L = 1.0                 
Z = 1.5 

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class SprindEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.rendered_env_num = min(5, self.num_envs)
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=5),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                world_frame_size=0.5,
                # show_link_frame=True,
                lights=[gs.options.vis.DirectionalLight(dir=(0.2, 0.4, -1), color=(1.0, 1.0, 1.0), intensity=5.0)],
                rendered_envs_idx=list(range(self.rendered_env_num))
                ),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            # renderer=gs.renderers.RayTracer()
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add target
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(0.5, 1.0, 0.5),
                    ),
                ),
            )
        else:
            self.target = None

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                # res=(1920, 1080),
                pos=(0.0, 4.5, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # add drones and rod
        self.bambi_1_init_pos = torch.tensor((-L/2, 0.0, Z), device=gs.device)
        self.bambi_2_init_pos = torch.tensor((L/2, 0.0, Z), device=gs.device)
        self.rod_init_pos     = torch.tensor((0, 0, Z - 0.08), device=gs.device)
        self.rod_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)

        self.rod_init_quat = xyz_to_quat(
            torch.tensor([[0.0, 90.0, 0.0]], device=gs.device), rpy=True, degrees=True
        ).squeeze(0)
        
        self.init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_quat)

        self.bambi_1 = self.scene.add_entity(
            morph=gs.morphs.Drone(
                file="utils/models/urdf/bros300.urdf",
                propellers_spin=(-1, -1, 1, 1),
                pos=self.bambi_1_init_pos,
            ),
        )
        self.bambi_2 = self.scene.add_entity(
            morph=gs.morphs.Drone(
                file="utils/models/urdf/bros300.urdf",
                propellers_spin=(-1, -1, 1, 1),
                pos=self.bambi_2_init_pos,
            ),
        )

        self.rod = self.scene.add_entity(
            morph=gs.morphs.Cylinder(
                radius=0.01,
                height= L + 0.1,
                euler=(0, 90, 0),
                pos=self.rod_init_pos,
                collision=False,
            ),
            material=gs.materials.Rigid(needs_coup=True, coup_friction=0.0),
            surface=gs.surfaces.Default(color=(0.0, 0.0, 0.0, 1.0)),
        )

        # self.rod_cg = self.scene.add_entity(
        #     morph=gs.morphs.Sphere(
        #         radius=0.015,
        #         collision=False,
        #         fixed=False,
        #         pos=(0, 0, Z - 0.08),
        #     ),
        #     surface=gs.surfaces.Default(color=(1.0, 0.2, 0.2, 1.0)),
        # )
        # NOTE: Add only for visuals
        # self.net = self.scene.add_entity(
        #     material=gs.materials.PBD.Cloth(),
        #     morph=gs.morphs.Mesh(
        #         file="utils/models/net.obj",
        #         scale=0.5,
        #         pos=(0.0, 0.0, 0.9),
        #         euler=(180.0, 0.0, 0.0),
        #     ),
        #     surface=gs.surfaces.Default(
        #         color=(0.2, 0.6, 0.2, 1.0),
        #     ),
        # )

        # build controller
        self.bambi_1_controller = build_controller(
            self.env_cfg["controller_type"], drone=self.bambi_1, num_envs=self.num_envs, dt=self.dt, cfg=self.env_cfg
        )
        self.bambi_2_controller = build_controller(
            self.env_cfg["controller_type"], drone=self.bambi_2, num_envs=self.num_envs, dt=self.dt, cfg=self.env_cfg
        )

        # build scene
        self.scene.build(n_envs=num_envs)
        solver  = self.scene.sim.rigid_solver
        tip1    = self.bambi_1.get_link("segment_5_cylinder")
        tip2    = self.bambi_2.get_link("segment_5_cylinder")
        rod_lnk = self.rod.base_link

        solver.add_weld_constraint(tip1.idx, rod_lnk.idx)
        solver.add_weld_constraint(tip2.idx, rod_lnk.idx)
        # rod cg
        # solver.add_weld_constraint(rod_lnk.idx, self.rod_cg.base_link.idx)

        # net NOTE: Add only for visuals
        # P = self.net.get_particles_pos()
        # P0 = P[0] if P.dim() == 3 else P          
        # top = torch.where(P0[:, 2] > P0[:, 2].max() - 0.02)[0]
        # self.net.fix_particles_to_link(rod_lnk.idx, particles_idx_local=top.tolist())

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.time_out_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_bool)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.bambi_1_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.bambi_2_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        self.bambi_1_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.bambi_2_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)

        self.bambi_1_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.bambi_2_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        self.bambi_1_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.bambi_2_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        
        self.rod_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_rod_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        self.rod_rel_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_rod_rel_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        self.bambi_1_rel_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.bambi_2_rel_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        self.extras = dict()  # extra information for logging

        self.rope_dofs_local = torch.arange(6, self.bambi_1.n_dofs, device=gs.device)
        self.reset()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), gs.device)

    def _at_target(self):
        return (
            (torch.norm(self.rod_rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )

    def step(self, actions):
        torch.clamp(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"], out=self.actions)

        bambi_1_actions = self.actions[:, 0:4]
        bambi_2_actions = self.actions[:, 4:8]
        self.bambi_1.set_propellers_rpm(self.bambi_1_controller.update(bambi_1_actions))
        self.bambi_2.set_propellers_rpm(self.bambi_2_controller.update(bambi_2_actions))

        # update target pos
        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.bambi_1_pos[:] = self.bambi_1.get_pos()
        self.bambi_2_pos[:] = self.bambi_2.get_pos()
        self.last_rod_pos[:] = self.rod_pos[:]
        self.rod_pos[:] = self.rod.get_pos()

        self.bambi_1_rel_pos = self.commands - self.bambi_1_pos
        self.bambi_2_rel_pos = self.commands - self.bambi_2_pos
        
        self.last_rod_rel_pos = self.commands - self.last_rod_pos
        self.rod_rel_pos = self.commands - self.rod_pos
        
        self.bambi_1_quat[:] = self.bambi_1.get_quat()
        self.bambi_2_quat[:] = self.bambi_2.get_quat()
        self.rod_quat[:] = self.rod.get_quat()
        self.bambi_1_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.bambi_1_quat), rpy=True, degrees=True
        )
        self.bambi_2_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.bambi_2_quat), rpy=True, degrees=True
        )
        inv_bambi_1_quat = inv_quat(self.bambi_1_quat)
        inv_bambi_2_quat = inv_quat(self.bambi_2_quat)
        

        self.bambi_1_lin_vel[:] = transform_by_quat(self.bambi_1.get_vel(), inv_bambi_1_quat)
        self.bambi_2_lin_vel[:] = transform_by_quat(self.bambi_2.get_vel(), inv_bambi_2_quat)

        self.bambi_1_ang_vel[:] = transform_by_quat(self.bambi_1.get_ang(), inv_bambi_1_quat)
        self.bambi_2_ang_vel[:] = transform_by_quat(self.bambi_2.get_ang(), inv_bambi_2_quat)

        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

        # check termination and reset
        self.termination_conditions = (
              (torch.abs(self.bambi_1_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.bambi_2_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])

            | (torch.abs(self.bambi_1_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.bambi_2_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])

            | (torch.abs(self.bambi_1_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.bambi_2_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])

            | (torch.abs(self.bambi_1_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.bambi_2_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])

            | (self.bambi_1_pos[:, 2] > self.env_cfg["termination_if_z_greater_than"])
            | (self.bambi_2_pos[:, 2] > self.env_cfg["termination_if_z_greater_than"])

            | (self.bambi_1_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
            | (self.bambi_2_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
            
            | self.scene.rigid_solver.get_error_envs_mask()
        )
        timed_out = self.episode_length_buf > self.max_episode_length
        self.reset_buf = timed_out | self.termination_conditions
        self.time_out_buf.copy_(timed_out)
        self.extras["time_outs"] = self.time_out_buf

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self._update_observation()

        self.last_actions[:] = self.actions[:]

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    # ------------ Get Observation ----------------
    def _update_observation(self):
        self.obs_buf = torch.cat(
            [
                # Rod rel pos
                torch.clip(self.rod_rel_pos * self.obs_scales["rel_pos"], -1, 1),
                # Drones state (rel_pos, q, v, w)
                torch.clip(self.bambi_1_rel_pos * self.obs_scales["rel_pos"], -1, 1),
                torch.clip(self.bambi_2_rel_pos * self.obs_scales["rel_pos"], -1, 1),
                self.bambi_1_quat,
                self.bambi_2_quat,
                torch.clip(self.bambi_1_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.bambi_2_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.bambi_1_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                torch.clip(self.bambi_2_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                # Last commands
                self.last_actions,
            ],
            axis=-1,
        )

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])
    
    # ------------ Reset Environment ----------------
    
    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        
        # reset base
        self.bambi_1_pos[envs_idx]  = self.bambi_1_init_pos
        self.bambi_2_pos[envs_idx]  = self.bambi_2_init_pos
        self.rod_pos[envs_idx]      = self.rod_init_pos
        self.last_rod_pos[envs_idx] = self.rod_init_pos

        self.bambi_1.set_pos(self.bambi_1_pos[envs_idx], envs_idx=envs_idx)
        self.bambi_2.set_pos(self.bambi_2_pos[envs_idx], envs_idx=envs_idx)
        self.rod.set_pos(self.rod_pos[envs_idx], envs_idx=envs_idx)

        self.bambi_1_quat[envs_idx] = self.init_quat.reshape(1, -1)
        self.bambi_2_quat[envs_idx] = self.init_quat.reshape(1, -1)
        self.rod_quat[envs_idx] = self.rod_init_quat.reshape(1, -1)
        self.bambi_1.set_quat(self.bambi_1_quat[envs_idx], envs_idx=envs_idx)
        self.bambi_2.set_quat(self.bambi_2_quat[envs_idx], envs_idx=envs_idx)
        self.rod.set_quat(self.rod_quat[envs_idx], envs_idx=envs_idx)
    
        self.bambi_1_lin_vel[envs_idx] = 0
        self.bambi_2_lin_vel[envs_idx] = 0
        self.bambi_1_ang_vel[envs_idx] = 0
        self.bambi_2_ang_vel[envs_idx] = 0

        zeros = torch.zeros(
            (len(envs_idx), len(self.rope_dofs_local)), device=gs.device, dtype=gs.tc_float
        )
        self.bambi_1.set_dofs_position(zeros, dofs_idx_local=self.rope_dofs_local, envs_idx=envs_idx)
        self.bambi_2.set_dofs_position(zeros, dofs_idx_local=self.rope_dofs_local, envs_idx=envs_idx)
        self.bambi_1.zero_all_dofs_velocity(envs_idx)
        self.bambi_2.zero_all_dofs_velocity(envs_idx)
        self.rod.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.bambi_1_controller.reset_idx(envs_idx)
        self.bambi_2_controller.reset_idx(envs_idx)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]) / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)
        torch.sub(self.commands, self.rod_pos, out=self.rod_rel_pos)
        torch.sub(self.commands, self.last_rod_pos, out=self.last_rod_rel_pos)
        torch.sub(self.commands, self.bambi_1_pos, out=self.bambi_1_rel_pos)
        torch.sub(self.commands, self.bambi_2_pos, out=self.bambi_2_rel_pos)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        self._update_observation()
        return self.get_observations()

    # ------------ reward functions----------------
    def _reward_progress(self):
        progress_rew = torch.sum(torch.square(self.last_rod_rel_pos), dim=1) - torch.sum(torch.square(self.rod_rel_pos), dim=1)
        return progress_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    # def _reward_yaw(self):
    #     yaw = self.base_euler[:, 2]
    #     yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # use rad for yaw_reward
    #     yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
    #     return yaw_rew

    def _reward_angular(self):
        angular_rew_1 = torch.norm(self.bambi_1_ang_vel / 3.14159, dim=1)
        angular_rew_2 = torch.norm(self.bambi_2_ang_vel / 3.14159, dim=1)
        return angular_rew_1 + angular_rew_2

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.termination_conditions] = 1
        return crash_rew
