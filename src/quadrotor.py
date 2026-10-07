import math

import torch

import genesis as gs
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat

from src.controller import build_controller
from utils.domain_rand import DomainRand


class Quadrotor:
    """One drone of a batched scene with the actuation of the goto task: domain randomization of its parameters
    (utils/domain_rand.py), action delay, controller and first-order motor lag, plus its state.

    Create it before the scene (the scene needs batch_links_info), add it with add_to(scene, ...) and call build()
    after scene.build(). domain_rand replaces the DomainRand of drone_cfg, e.g. a DesignInformedDR (per-env max_rpm)."""

    def __init__(self, drone_cfg, dr_cfg, env_cfg, num_envs, dt, domain_rand=None):
        self.drone_cfg = drone_cfg
        self.env_cfg = env_cfg
        self.num_envs = num_envs
        self.dt = dt
        self.domain_rand = domain_rand or DomainRand(drone_cfg, dr_cfg, num_envs)

        def buf(*shape):
            return torch.zeros((num_envs, *shape), device=gs.device, dtype=gs.tc_float)

        self.pos = buf(3)
        self.quat = buf(4)
        self.euler = buf(3)    # [deg]
        self.lin_vel = buf(3)  # body frame
        self.ang_vel = buf(3)  # body frame

        # motor lag: first-order rpm response with spin-up / spin-down time constants (domain_rand.motor_tau)
        self.motor_alpha_up, self.motor_k_up = buf(1), buf(1)
        self.motor_alpha_down, self.motor_k_down = buf(1), buf(1)
        self.motor_rpm = buf(4)

        # action delay (domain_rand.action_delay)
        n_hist = int(math.floor(self.domain_rand.max_action_delay / dt)) + 2
        self.action_hist = buf(n_hist, 4)
        self.action_hist_empty = torch.ones((num_envs,), device=gs.device, dtype=torch.bool)
        self.envs = torch.arange(num_envs, device=gs.device)

    def add_to(self, scene, urdf, pos):
        # align=False: Genesis refuses to write the inertial properties of an aligned body at runtime
        self.entity = scene.add_entity(
            gs.morphs.Drone(file=urdf, pos=pos, propellers_spin=tuple(self.drone_cfg["propellers_spin"]), align=False)
        )
        nominal = self.domain_rand.nominal_cfg() if hasattr(self.domain_rand, "nominal_cfg") else self.drone_cfg
        self.controller = build_controller(
            self.env_cfg["controller_type"], drone=self.entity, num_envs=self.num_envs, dt=self.dt,
            cfg={**self.env_cfg, "drone": nominal},
        )
        return self.entity

    def build(self):
        self.domain_rand.build(self.entity)
        if hasattr(self.domain_rand, "max_rpm"):  # per-env max rpm of the sampled drones
            self.controller.max_rpm = self.domain_rand.max_rpm
        # per-env hover command (SRTHover) from each drone's mass and kf
        self.hover_per_env = self.domain_rand.hover_noise is not None
        if self.hover_per_env and hasattr(self.controller, "hover_u"):
            self.controller.hover_u = self.domain_rand.hover_u

    def apply(self, actions):
        """Delay, controller and motor lag, then the rpm of this step."""
        self.entity.set_propellers_rpm(self._motor_lag(self.controller.update(self._delay(actions))))

    def update_state(self):
        self.pos[:] = self.entity.get_pos()
        self.quat[:] = self.entity.get_quat()
        self.euler[:] = quat_to_xyz(self.quat, rpy=True, degrees=True)
        inv_q = inv_quat(self.quat)
        self.lin_vel[:] = transform_by_quat(self.entity.get_vel(), inv_q)
        self.ang_vel[:] = transform_by_quat(self.entity.get_ang(), inv_q)

    def reset_idx(self, envs_idx):
        self.controller.reset_idx(envs_idx)
        self.domain_rand.reset_idx(envs_idx)
        tau = self.domain_rand.motor_tau[envs_idx]
        self.motor_alpha_up[envs_idx], self.motor_k_up[envs_idx] = self._motor_lag_coeffs(tau[:, 0:1])
        self.motor_alpha_down[envs_idx], self.motor_k_down[envs_idx] = self._motor_lag_coeffs(tau[:, 1:2])
        # episodes start mid-air
        if self.hover_per_env:
            self.motor_rpm[envs_idx] = self.domain_rand.hover_rpm[envs_idx]
        else:
            self.motor_rpm[envs_idx] = float(self.controller.hover_rpm)
        self.action_hist_empty[envs_idx] = True

    def _motor_lag_coeffs(self, tau):
        """(alpha, k) of a first-order lag over one step with the command held: alpha = exp(-dt/tau) is the error
        left at the end of the step, k = tau/dt (1 - alpha) on average over it. tau <= 0 follows instantly."""
        tau = tau.clamp(min=1e-9)
        alpha = torch.exp(-self.dt / tau)
        k = tau / self.dt * (1.0 - alpha)
        instant = tau <= 1e-9
        return alpha.masked_fill(instant, 0.0), k.masked_fill(instant, 0.0)

    def _motor_lag(self, rpm_cmd):
        """Spin-up or spin-down time constant per motor, as Gazebo; returns the mean rpm over the step since
        Genesis holds it for the whole step."""
        spin_up = rpm_cmd > self.motor_rpm
        alpha = torch.where(spin_up, self.motor_alpha_up, self.motor_alpha_down)
        k = torch.where(spin_up, self.motor_k_up, self.motor_k_down)
        error = self.motor_rpm - rpm_cmd
        rpm_mean = rpm_cmd + k * error
        self.motor_rpm = rpm_cmd + alpha * error
        return rpm_mean * self.domain_rand.rpm_scale

    def _delay(self, actions):
        """With delay D = (n + f) dt the command of n+1 steps ago is active for the first f dt of the step and the
        one of n steps ago for the rest; returns their mean."""
        self.action_hist = torch.roll(self.action_hist, 1, dims=1)
        self.action_hist[:, 0] = actions
        empty = self.action_hist_empty  # no history after a reset: the first action has always been applied
        self.action_hist[empty] = actions[empty].unsqueeze(1).expand(-1, self.action_hist.shape[1], -1)
        self.action_hist_empty[:] = False

        steps = self.domain_rand.action_delay / self.dt
        n = steps.floor().long()
        f = (steps - n).unsqueeze(1)
        return (1.0 - f) * self.action_hist[self.envs, n] + f * self.action_hist[self.envs, n + 1]
