import torch
import math
import copy
import yaml
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


def load_track(track_path):
    with open(track_path, "r") as f:
        track = yaml.safe_load(f)
    return {
        "gates_pos": [g["position"] for g in track["gates"]],
        "gates_rpy": [g["rpy"] for g in track["gates"]],
        "limits": {k: v for d in track["limits"] for k, v in d.items()},
    }
class RaceEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        track = load_track(command_cfg["track_path"])
        # track = load_track("/home/adame/genesis_ws/misc/fig8.yaml") # TODO: Remove
        env_cfg.update(track["limits"])

        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
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
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                world_frame_size=0.5,
                show_link_frame=True,
                lights=[gs.options.vis.DirectionalLight(dir=(0.2, 0.4, -1), color=(1.0, 1.0, 1.0), intensity=5.0)],
                rendered_envs_idx=list(range(self.rendered_env_num))
                ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add target
        if self.env_cfg["visualize_target"]:
            # static gate meshes for the whole track
            for gate_pos, gate_rpy in zip(track["gates_pos"], track["gates_rpy"]):
                self.scene.add_entity(
                    morph=gs.morphs.Mesh(
                        file="misc/gate.obj",
                        euler=( gate_rpy[0], gate_rpy[1], gate_rpy[2]),
                        pos=tuple(gate_pos),
                        fixed=True,
                        collision=False,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(
                            color=(0.0, 0.5, 0.5),
                        ),
                    ),
                )
            # sphere marking the current target gate
            # self.target = self.scene.add_entity(
            #     morph=gs.morphs.Mesh(
            #         file="meshes/sphere.obj",
            #         scale=0.1,
            #         fixed=False,
            #         collision=False,
            #     ),
            #     surface=gs.surfaces.Rough(
            #         diffuse_texture=gs.textures.ColorTexture(
            #             color=(1.0, 1.0, 0.0),
            #         ),
            #     ),
            # )
        else:
            self.target = None

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # add drone
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone = self.scene.add_entity(
            gs.morphs.Drone(
                file="misc/urdf/a300.urdf",
                propellers_spin=(-1, -1, 1, 1)
            )
        )

        # build scene
        self.scene.build(n_envs=num_envs)
        self.controller = build_controller(
            self.env_cfg["controller_type"], drone=self.drone, num_envs=self.num_envs, dt=self.dt, cfg=self.env_cfg
        )

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

        # racing track gates: each env targets the gates sequentially
        self.gates_position = torch.tensor(track["gates_pos"], device=gs.device, dtype=gs.tc_float)
        self.gates_rpy = torch.tensor(track["gates_rpy"], device=gs.device, dtype=gs.tc_float)
        self.gates_R = quat_to_R(xyz_to_quat(self.gates_rpy, rpy=True, degrees=True))  
        self.num_gates = self.gates_position.shape[0]
        self.gate_idx = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.long)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        
        self.rel_pos = torch.zeros_like(self.base_pos)
        self.last_rel_pos = torch.zeros_like(self.base_pos)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        self.extras = dict()  # extra information for logging

        self.reset()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx] = self.gates_position[self.gate_idx[envs_idx]]
    
    def _at_gate(self):
        R = self.gates_R[self.gate_idx]                       
        c = self.gates_position[self.gate_idx]                
        Rt = R.transpose(1, 2)

        prev = torch.einsum("nij,nj->ni", Rt, self.last_base_pos - c)
        curr = torch.einsum("nij,nj->ni", Rt, self.base_pos - c)

        plane_crossed = (prev[:, 1] < 0.0) & (curr[:, 1] >= 0.0)              
        inside_gate = (curr[:, 0].abs() < self.env_cfg["gate_half_width"]) & (curr[:, 2].abs() < self.env_cfg["gate_half_height"])

        return plane_crossed & inside_gate, plane_crossed & ~inside_gate

    def step(self, actions):
        torch.clamp(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"], out=self.actions)

        self.drone.set_propellers_rpm(self.controller.update(self.actions))
        # update target pos
        # if self.target is not None:
        #     self.target.set_pos(self.commands, zero_velocity=True)

        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        # pick a new random gate (different from the current one) for envs that reached their target
        self.gate_success, self.gate_crash = self._at_gate()
        torch.where(self.gate_success, (self.gate_idx + 1) % self.num_gates, self.gate_idx, out=self.gate_idx)
        self.commands.copy_(self.gates_position[self.gate_idx])
        torch.sub(self.commands, self.base_pos, out=self.rel_pos)
        torch.sub(self.commands, self.last_base_pos, out=self.last_rel_pos)

        # check termination and reset
        self.termination_conditions = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.base_pos[:, 0]) > self.env_cfg["arena_half_x"])
            | (torch.abs(self.base_pos[:, 1]) > self.env_cfg["arena_half_y"])
            | (self.base_pos[:, 2] > self.env_cfg["arena_z_max"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
            | self.gate_crash
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
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])
    
    # ------------ Reset Environment ----------------
    def _sample_spawn(self, envs_idx):
        """Sample poses behind each env's assigned gate. envs_idx: index tensor."""
        n = len(envs_idx)
        R = self.gates_R[self.gate_idx[envs_idx]]        # (n,3,3) gate -> world
        c = self.gates_position[self.gate_idx[envs_idx]] # (n,3)

        def u(lo, hi):
            return torch.empty((n,), device=gs.device, dtype=gs.tc_float).uniform_(lo, hi)

        lo, hi = self.env_cfg["spawn_back_dist"]
        local = torch.stack(
            [
                u(-self.env_cfg["spawn_lateral"], self.env_cfg["spawn_lateral"]),
                -u(lo, hi),                              # negative y = behind the plane
                u(-self.env_cfg["spawn_vertical"], self.env_cfg["spawn_vertical"]),
            ],
            dim=-1,
        )
        pos = c + torch.einsum("nij,nj->ni", R, local)
        pos[:, 2] = pos[:, 2].clamp(min=self.env_cfg["spawn_min_height"])

        fwd = R[:, :, 1]                                
        gate_yaw = torch.atan2(fwd[:, 1], fwd[:, 0])      
        yaw = gate_yaw - math.pi / 2 + u(-math.pi/6, math.pi/6)
        yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))  

        rpy = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), yaw], dim=-1)
        quat = xyz_to_quat(rpy, rpy=True, degrees=False)

        return pos, quat
    
    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        
        self.gate_idx[envs_idx] = torch.randint(0, self.num_gates, (len(envs_idx),), device=gs.device)
        pos, quat = self._sample_spawn(envs_idx)

        # reset base
        self.base_pos[envs_idx] = pos
        self.last_base_pos[envs_idx] = pos
        self.base_quat[envs_idx] = quat
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.controller.reset_idx(envs_idx)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]) / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)
        torch.sub(self.commands, self.base_pos, out=self.rel_pos)
        torch.sub(self.commands, self.last_base_pos, out=self.last_rel_pos)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        self._update_observation()
        return self.get_observations()

    # ------------ reward functions----------------
    def _reward_progress(self):
        progress_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        return torch.where(self.gate_success, torch.zeros_like(progress_rew), progress_rew)

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  
        yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
        return yaw_rew

    def _reward_angular(self):
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=1)
        return angular_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.termination_conditions] = 1
        return crash_rew
