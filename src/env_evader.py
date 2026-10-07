"""evader (was robust_goto on des-inf-dom-rand): one design-informed general drone goes to waypoints, encoder on the
drone parameters, with, all optional (hydra_configs/task/evader.yaml):

    env.upset:        {fraction, heights, ang_vel}  a fraction of the episodes starts at a uniformly random
                      attitude (upside down included) with random body rates, at one of heights [m]
    env.rerandomize:  {fraction}  a fraction of the episodes gets a new drone at a uniformly random step, and the
                      encoder its new parameters
    obs.params_noise: {physical, delay}  the encoder sees the physical parameters x U(1 -+ physical) and the
                      delay + U(-+ delay) [s], fixed per episode (as an inexact rl_goto drone block); the critic
                      gets the true ones in the params_true group
"""
import torch
import math
import copy
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    xyz_to_quat,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)
from src.controller import build_controller
from utils.design_informed_dr import DesignInformedDR


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class EvaderEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        # control step [s]; use a multiple of the deployment loop period (4 ms in gazebo) so the
        # policy can run there with identical hold times
        self.dt = env_cfg.get("dt", 0.01)
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        # drone (hydra_configs/drone) with design-informed randomization of its parameters
        self.drone_cfg = env_cfg["drone"]
        self.domain_rand = DesignInformedDR(self.drone_cfg, env_cfg["domain_rand"], self.num_envs)

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.rendered_env_num))),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                # per-env mass / inertia / arm, only when randomized since it slows the solver down
                batch_links_info=self.domain_rand.batch_links_info,
            ),
            show_viewer=show_viewer,
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
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
            )
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
            # align=False: Genesis refuses to write the inertial properties of an aligned body at runtime
            gs.morphs.Drone(file=self.drone_cfg["urdf"],
                            propellers_spin=tuple(self.drone_cfg["propellers_spin"]), align=False),
        )

        # build_controller, on the middle design (the per-env values are set after the build)
        self.controller = build_controller(
            self.env_cfg["controller_type"], drone=self.drone, num_envs=self.num_envs, dt=self.dt,
            cfg={**self.env_cfg, "drone": self.domain_rand.nominal_cfg()},
        )

        # build scene
        self.scene.build(n_envs=num_envs)
        self.domain_rand.build(self.drone)
        self.controller.max_rpm = self.domain_rand.max_rpm  # per-env max rpm of the sampled drones

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        # motor lag: first-order rpm response with separate spin-up / spin-down time constants
        # (domain_rand.motor_tau), like Gazebo's MulticopterMotorModel. tau = 0 means the rpm follows
        # the command instantly.
        shape = (self.num_envs, 1)
        self.motor_alpha_up = torch.zeros(shape, device=gs.device, dtype=gs.tc_float)
        self.motor_k_up = torch.zeros_like(self.motor_alpha_up)
        self.motor_alpha_down = torch.zeros_like(self.motor_alpha_up)
        self.motor_k_down = torch.zeros_like(self.motor_alpha_up)
        self.motor_rpm = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)

        # initial state randomization (half-ranges; lin vel [m/s], ang vel [rad/s], tilt/yaw [deg])
        self.init_lin_vel = env_cfg.get("init_lin_vel", 0.0)
        self.init_ang_vel = env_cfg.get("init_ang_vel", 0.0)
        self.init_tilt = env_cfg.get("init_tilt", 0.0)
        self.init_yaw = env_cfg.get("init_yaw", 0.0)

        # upset starts and mid-episode drone changes (module docstring)
        self.upset = env_cfg.get("upset", {})
        self.rerand_fraction = env_cfg.get("rerandomize", {}).get("fraction", 0.0)
        self.rerand_step = torch.full((self.num_envs,), -1, device=gs.device, dtype=gs.tc_int)

        # gaussian observation noise in physical units (before scaling), e.g. {"ang_vel": 0.1}
        self.obs_noise = obs_cfg.get("obs_noise", {})

        # correlated state-estimation errors (like PX4 EKF2 vs ground truth), all optional:
        #   att_std_deg / att_tau:   roll/pitch/yaw error, Ornstein-Uhlenbeck (slowly varying) [deg, s]
        #   yaw_offset_deg:          constant heading error per episode, uniform half-range [deg]
        #   vel_std / vel_std_z / vel_tau: world-frame velocity bias, OU [m/s, s]
        #   pos_offset:              constant position error per episode, uniform half-range [m] (x, y, z)
        # The observed body velocity is rotated with the erroneous attitude, as an estimator would.
        self.obs_bias = obs_cfg.get("obs_bias", {})

        # observation history: the policy sees the current frame followed by the previous history-1
        # frames (newest first, one per env step); after a reset it is filled with the first frame
        self.obs_history = obs_cfg.get("history", 1)
        self.frame_size = 13 + self.num_actions
        self.obs_hist = torch.zeros((self.num_envs, self.obs_history, self.frame_size), device=gs.device, dtype=gs.tc_float)
        self.obs_hist_empty = torch.ones((self.num_envs,), device=gs.device, dtype=torch.bool)

        # privileged critic: an extra "critic" observation group with the noise-free state, the motor
        # speeds and the randomized parameters (use obs_groups critic: [policy, critic])
        self.privileged = obs_cfg.get("privileged", False)

        # drone parameters: an extra "params" observation group, set at every reset, for an encoder in the
        # model (src/models.py EncoderMLPModel)
        self.use_params = obs_cfg.get("params", False)
        self.params_buf = torch.zeros((self.num_envs, 17), device=gs.device, dtype=gs.tc_float)
        self.params_noise = obs_cfg.get("params_noise", {})
        self.params_err = torch.zeros_like(self.params_buf)  # added to params_buf for the encoder
        self.params_obs = self.params_buf
        self.att_err = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)  # OU part [deg]
        self.yaw_offset = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)  # [deg]
        self.vel_bias = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.pos_offset = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        # action delay (domain_rand.action_delay): models the odometry -> policy -> motor latency of
        # the real pipeline, 0 applies every action immediately
        n_hist = int(math.floor(self.domain_rand.max_action_delay / self.dt)) + 2
        self.action_hist = torch.zeros((self.num_envs, n_hist, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.action_hist_empty = torch.ones((self.num_envs,), device=gs.device, dtype=torch.bool)

        self.extras = dict()  # extra information for logging

        self.reset()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), gs.device)

    def _motor_lag_coeffs(self, tau):
        """(alpha, k) for a first-order lag over one env step with the command held, tau: tensor [s].
        alpha: fraction of the error left at the end of the step, exp(-dt/tau)
        k:     fraction of the error left on average over the step, tau/dt * (1 - alpha)
        tau <= 0 gives (0, 0): the rpm follows the command instantly."""
        tau = tau.clamp(min=1e-9)
        alpha = torch.exp(-self.dt / tau)
        k = tau / self.dt * (1.0 - alpha)
        instant = tau <= 1e-9
        return alpha.masked_fill(instant, 0.0), k.masked_fill(instant, 0.0)

    def _motor_lag(self, rpm_cmd):
        """Advance the motors one step towards rpm_cmd and return the rpm to apply this step.

        The time constant is chosen per motor from the direction of the change (spin up if the
        command is above the current rpm), as in Gazebo. The returned rpm is the mean over the
        step rather than the end value, because Genesis holds it constant for the whole step."""
        spin_up = rpm_cmd > self.motor_rpm
        alpha = torch.where(spin_up, self.motor_alpha_up, self.motor_alpha_down)
        k = torch.where(spin_up, self.motor_k_up, self.motor_k_down)
        error = self.motor_rpm - rpm_cmd
        rpm_mean = rpm_cmd + k * error
        self.motor_rpm = rpm_cmd + alpha * error
        return rpm_mean * self.domain_rand.rpm_scale

    def _delay_actions(self, actions):
        """Push actions into the history and return the actions the motors see this step.

        With delay D = (n + f) * dt, the command active during the step is the one from n+1 steps
        ago for the first f*dt and the one from n steps ago for the rest; the mean of the two is
        returned since Genesis applies one command per step."""
        self.action_hist = torch.roll(self.action_hist, 1, dims=1)
        self.action_hist[:, 0] = actions
        # after a reset there is no history yet: pretend the first action has always been applied
        empty = self.action_hist_empty
        self.action_hist[empty] = actions[empty].unsqueeze(1).expand(-1, self.action_hist.shape[1], -1)
        self.action_hist_empty[:] = False

        steps = self.domain_rand.action_delay / self.dt
        n = steps.floor().long()
        f = (steps - n).unsqueeze(1)
        idx = torch.arange(self.num_envs, device=gs.device)
        return (1.0 - f) * self.action_hist[idx, n] + f * self.action_hist[idx, n + 1]

    def _at_target(self):
        return (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )

    def step(self, actions):
        # self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        # action range: [0, clip_actions] for SRT (motor commands), action_clip overrides
        lo, hi = self.env_cfg.get("action_clip", [0.0, self.env_cfg["clip_actions"]])
        self.actions = torch.clip(actions, lo, hi)

        self.drone.set_propellers_rpm(self._motor_lag(self.controller.update(self._delay_actions(self.actions))))

        # print(self.actions)

        # update target pos
        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        # mid-episode drone change
        if self.rerand_fraction > 0.0:
            self._rerandomize((self.episode_length_buf == self.rerand_step).nonzero(as_tuple=False).reshape((-1,)))

        # resample commands
        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

        # check termination and reset
        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
        )
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # the observation carries the action just applied, as rl_goto.cpp does (it used to carry the one
        # before, one step older than at deployment)
        self.last_actions[:] = self.actions[:]
        self.last_actions[self.reset_buf.bool()] = 0.0  # a new episode starts from zero, like rl_goto on activation

        # compute observations
        self._step_obs_bias()
        self._update_observation()

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def _ou_step(self, x, std, tau):
        """One dt step of a zero-mean Ornstein-Uhlenbeck process with stationary std and time constant tau."""
        a = math.exp(-self.dt / tau)
        return a * x + std * math.sqrt(1.0 - a * a) * torch.randn_like(x)

    def _step_obs_bias(self):
        b = self.obs_bias
        if not b:
            return
        att_std = b.get("att_std_deg", 0.0)
        if att_std > 0.0:
            self.att_err = self._ou_step(self.att_err, att_std, b.get("att_tau", 2.0))
        vel_std, vel_std_z = b.get("vel_std", 0.0), b.get("vel_std_z", 0.0)
        if vel_std > 0.0 or vel_std_z > 0.0:
            std = torch.tensor([vel_std, vel_std, vel_std_z], device=gs.device, dtype=gs.tc_float)
            self.vel_bias = self._ou_step(self.vel_bias, std, b.get("vel_tau", 1.0))

    def _reset_obs_bias(self, envs_idx):
        b = self.obs_bias
        if not b:
            return
        n = len(envs_idx)
        # start the OU processes from their stationary distribution
        self.att_err[envs_idx] = b.get("att_std_deg", 0.0) * torch.randn((n, 3), device=gs.device, dtype=gs.tc_float)
        yaw = b.get("yaw_offset_deg", 0.0)
        self.yaw_offset[envs_idx] = gs_rand_float(-yaw, yaw, (n,), gs.device)
        std = torch.tensor([b.get("vel_std", 0.0), b.get("vel_std", 0.0), b.get("vel_std_z", 0.0)], device=gs.device, dtype=gs.tc_float)
        self.vel_bias[envs_idx] = std * torch.randn((n, 3), device=gs.device, dtype=gs.tc_float)
        off = torch.tensor(b.get("pos_offset", [0.0, 0.0, 0.0]), device=gs.device, dtype=gs.tc_float)
        self.pos_offset[envs_idx] = (2.0 * torch.rand((n, 3), device=gs.device, dtype=gs.tc_float) - 1.0) * off

    def _update_observation(self):
        rel_pos, quat, lin_vel, ang_vel = self.rel_pos, self.base_quat, self.base_lin_vel, self.base_ang_vel
        if self.obs_bias:
            # estimate = truth + correlated error: world-frame attitude error, velocity bias, position offset
            err = self.att_err.clone()
            err[:, 2] += self.yaw_offset
            quat = transform_quat_by_quat(quat, xyz_to_quat(err, rpy=True, degrees=True))
            vel_world = transform_by_quat(self.base_lin_vel, self.base_quat) + self.vel_bias
            lin_vel = transform_by_quat(vel_world, inv_quat(quat))
            rel_pos = rel_pos - self.pos_offset
        if self.obs_noise:
            noisy = lambda x, key: x + self.obs_noise.get(key, 0.0) * torch.randn_like(x)
            rel_pos, lin_vel, ang_vel = noisy(rel_pos, "rel_pos"), noisy(lin_vel, "lin_vel"), noisy(ang_vel, "ang_vel")
            quat = noisy(quat, "quat")
            quat = quat / quat.norm(dim=1, keepdim=True)
        frame = self._frame(rel_pos, quat, lin_vel, ang_vel)

        if self.obs_history > 1:
            self.obs_hist = torch.roll(self.obs_hist, 1, dims=1)
            self.obs_hist[:, 0] = frame
            empty = self.obs_hist_empty
            self.obs_hist[empty] = frame[empty].unsqueeze(1).expand(-1, self.obs_history, -1)
            self.obs_hist_empty[:] = False
            self.obs_buf = self.obs_hist.reshape(self.num_envs, -1)
        else:
            self.obs_buf = frame

        if self.privileged:
            dr = self.domain_rand
            self.priv_buf = torch.cat(
                [
                    self._frame(self.rel_pos, self.base_quat, self.base_lin_vel, self.base_ang_vel),
                    self.motor_rpm / self.controller.max_rpm,
                    dr.mass.unsqueeze(1) / dr.nominal("mass") - 1.0,
                    dr.inertia / dr.nominal("inertia") - 1.0,
                    dr.kf / dr.nominal("kf") - 1.0,
                    dr.motor_tau / dr.nominal("motor_tau") - 1.0,
                    dr.action_delay.unsqueeze(1) / 0.02,
                    dr.motor_eff - 1.0,
                ],
                axis=-1,
            )

    def _frame(self, rel_pos, quat, lin_vel, ang_vel):
        return torch.cat(
            [
                torch.clip(rel_pos * self.obs_scales["rel_pos"], -1, 1),
                quat,
                torch.clip(lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(ang_vel * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

    def _update_params(self, envs_idx):
        """Drone parameters, log ratios to the middle design (c = 0.5) and the per-episode ones scaled to ~[-1, 1]."""
        dr = self.domain_rand
        log_ratio = lambda name, x: torch.log(x / dr.nominal(name))
        self.params_buf[envs_idx] = torch.cat(
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

    def params_reference(self):
        """Values the drone parameters are normalized with (log ratios): mass, inertia (3), arm (4), kf, max_rpm,
        motor_tau (2) of the middle design."""
        dr = self.domain_rand
        names = ("mass", "inertia", "arm", "kf", "max_rpm", "motor_tau")
        return torch.cat([dr.nominal(n).reshape(-1) for n in names])

    def _reset_params_err(self, envs_idx):
        """Per-episode error of the parameters the encoder sees (obs.params_noise), as log ratios: physical ones
        x U(1 -+ physical) (one factor for the 4 arms, rl_goto gives one arm), delay + U(-+ delay) [s]."""
        if not self.params_noise:
            return
        n = len(envs_idx)
        a = self.params_noise.get("physical", 0.0)
        err = torch.log(1.0 + a * (2.0 * torch.rand((n, 9), device=gs.device, dtype=gs.tc_float) - 1.0))
        self.params_err[envs_idx, 0:4] = err[:, 0:4]  # mass, inertia
        self.params_err[envs_idx, 4:8] = err[:, 4:5]  # arms
        self.params_err[envs_idx, 8:12] = err[:, 5:9]  # kf, max_rpm, motor_tau
        d = self.params_noise.get("delay", 0.0)
        self.params_err[envs_idx, 16] = gs_rand_float(-d, d, (n,), gs.device) / 0.02

    def _set_motor_lag(self, envs_idx):
        tau = self.domain_rand.motor_tau[envs_idx]
        self.motor_alpha_up[envs_idx], self.motor_k_up[envs_idx] = self._motor_lag_coeffs(tau[:, 0:1])
        self.motor_alpha_down[envs_idx], self.motor_k_down[envs_idx] = self._motor_lag_coeffs(tau[:, 1:2])

    def _rerandomize(self, envs_idx):
        """A new drone for envs_idx mid-flight: the motors keep their rpm, the encoder gets the new parameters."""
        if len(envs_idx) == 0:
            return
        self.domain_rand.resample(envs_idx)
        self._set_motor_lag(envs_idx)
        self._update_params(envs_idx)

    def _upset_start(self, envs_idx):
        """Start a fraction of envs_idx at a uniformly random attitude with random body rates, at one of the upset
        heights (above or below the target plane)."""
        u = self.upset
        envs_idx = envs_idx[torch.rand(len(envs_idx), device=gs.device) < u.get("fraction", 0.0)]
        n = len(envs_idx)
        if n == 0:
            return
        heights = torch.tensor(u["heights"], device=gs.device, dtype=gs.tc_float)
        pos = self.base_init_pos.expand(n, -1).clone()
        pos[:, 2] = heights[torch.randint(len(heights), (n,), device=gs.device)]
        quat = torch.randn((n, 4), device=gs.device, dtype=gs.tc_float)
        quat = quat / quat.norm(dim=1, keepdim=True)  # uniform on SO(3)
        self.base_pos[envs_idx] = pos
        self.last_base_pos[envs_idx] = pos
        self.base_quat[envs_idx] = quat
        self.drone.set_pos(pos, zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(quat, zero_velocity=True, envs_idx=envs_idx)
        lin_vel = gs_rand_float(-self.init_lin_vel, self.init_lin_vel, (n, 3), gs.device)  # world
        ang_vel = gs_rand_float(-u["ang_vel"], u["ang_vel"], (n, 3), gs.device)
        self.drone.set_dofs_velocity(torch.cat([lin_vel, ang_vel], dim=1), dofs_idx_local=list(range(6)), envs_idx=envs_idx)
        inv_q = inv_quat(quat)
        self.base_lin_vel[envs_idx] = transform_by_quat(lin_vel, inv_q)
        self.base_ang_vel[envs_idx] = transform_by_quat(self.drone.get_ang()[envs_idx], inv_q)

    def get_observations(self):
        obs = {"policy": self.obs_buf}
        if self.privileged:
            obs["critic"] = self.priv_buf
        if self.use_params:
            obs["params"] = self.params_buf + self.params_err if self.params_noise else self.params_buf
            if self.params_noise:
                obs["params_true"] = self.params_buf
        return TensorDict(obs, batch_size=[self.num_envs])

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        # randomized initial attitude and velocities (e.g. RLGoto takes over while still climbing)
        if self.init_tilt > 0.0 or self.init_yaw > 0.0 or self.init_lin_vel > 0.0 or self.init_ang_vel > 0.0:
            n = len(envs_idx)
            rpy = torch.stack(
                [
                    gs_rand_float(-self.init_tilt, self.init_tilt, (n,), gs.device),
                    gs_rand_float(-self.init_tilt, self.init_tilt, (n,), gs.device),
                    gs_rand_float(-self.init_yaw, self.init_yaw, (n,), gs.device),
                ],
                dim=1,
            )
            quat = transform_quat_by_quat(xyz_to_quat(rpy, rpy=True, degrees=True), self.base_init_quat.reshape(1, -1).expand(n, -1))
            self.base_quat[envs_idx] = quat
            self.drone.set_quat(quat, zero_velocity=True, envs_idx=envs_idx)
            lin_vel = gs_rand_float(-self.init_lin_vel, self.init_lin_vel, (n, 3), gs.device)  # world
            ang_vel = gs_rand_float(-self.init_ang_vel, self.init_ang_vel, (n, 3), gs.device)
            self.drone.set_dofs_velocity(torch.cat([lin_vel, ang_vel], dim=1), dofs_idx_local=list(range(6)), envs_idx=envs_idx)
            inv_q = inv_quat(quat)
            self.base_lin_vel[envs_idx] = transform_by_quat(lin_vel, inv_q)
            self.base_ang_vel[envs_idx] = transform_by_quat(self.drone.get_ang()[envs_idx], inv_q)

        if self.upset:
            self._upset_start(envs_idx)

        self.controller.reset_idx(envs_idx)
        self.domain_rand.reset_idx(envs_idx)
        self._update_params(envs_idx)
        self._reset_params_err(envs_idx)
        self._set_motor_lag(envs_idx)
        self.motor_rpm[envs_idx] = self.domain_rand.hover_rpm[envs_idx]  # episodes start mid-air
        if self.rerand_fraction > 0.0:
            n = len(envs_idx)
            step = torch.randint(1, self.max_episode_length + 1, (n,), device=gs.device, dtype=gs.tc_int)
            self.rerand_step[envs_idx] = torch.where(torch.rand(n, device=gs.device) < self.rerand_fraction, step, -1)
        self.action_hist_empty[envs_idx] = True
        self.obs_hist_empty[envs_idx] = True
        self._reset_obs_bias(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        self._update_observation()
        return self.get_observations()

    # ------------ reward functions----------------
    def _reward_target(self):
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        return target_rew

    def _reward_effort(self):
        # squared action
        return torch.sum(torch.square(self.actions), dim=1)

    def _reward_tilt(self):
        # 1 - cos(tilt) = 2 (qx^2 + qy^2)
        return 2.0 * (torch.square(self.base_quat[:, 1]) + torch.square(self.base_quat[:, 2]))

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # use rad for yaw_reward
        yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
        return yaw_rew

    def _reward_angular(self):
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=1)
        return angular_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew
