import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import numpy as np

import genesis as gs
from genesis.utils.geom import (
    transform_by_quat,
    inv_quat,
)

from utils.domain_rand import load_drone_cfg

if TYPE_CHECKING:
    from genesis.engine.entities.drone_entity import DroneEntity

def drone_params(urdf: str, drone_cfg: dict | None = None) -> dict:
    """Nominal mass, kf, km, max_rpm, hover_rpm and thrust2weight of a drone, from its config (given, or
    hydra_configs/drone/<urdf name>.yaml). Urdfs without a config (bros300, bambi) give mass, kf, km and
    thrust2weight in their <properties>."""
    name = os.path.splitext(os.path.basename(urdf))[0]
    if drone_cfg is None and os.path.exists(f"hydra_configs/drone/{name}.yaml"):
        drone_cfg = load_drone_cfg(name)
    if drone_cfg is not None:
        p = {k: drone_cfg[k]["nominal"] for k in ("mass", "kf", "km")}
        p["max_rpm"] = drone_cfg["max_rpm"]
    else:
        root = ET.parse(urdf).getroot()
        p = {k: float(v) for k, v in root.find("properties").attrib.items()}
        p.setdefault("mass", sum(float(m.get("value")) for m in root.iter("mass")))

    p["hover_rpm"] = np.sqrt(9.81 * p["mass"] / 4.0 / p["kf"])
    if "max_rpm" in p:
        p["thrust2weight"] = (p["max_rpm"] / p["hover_rpm"]) ** 2
    else:
        p["max_rpm"] = p["hover_rpm"] * np.sqrt(p["thrust2weight"])
    return p


class BaseController(ABC):

    def __init__(self, drone: "DroneEntity", num_envs: int, dt: float, cfg: dict):
        self.drone = drone
        self.num_envs = num_envs
        self.dt = dt
        self.cfg = cfg

        params = drone_params(drone.morph.file, cfg.get("drone"))
        self.KF = params["kf"]
        self.KM = params["km"]
        self.mass = params["mass"]
        self.hover_rpm = params["hover_rpm"]
        self.max_rpm = params["max_rpm"]
        self.min_rpm    = 3200.0 # TODO: Hard coded for now.... 
        self.hover_cmd  = (self.hover_rpm - self.min_rpm) / (self.max_rpm - self.min_rpm)

    @abstractmethod
    def update(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: (num_envs, num_actions) -> rpms: (num_envs, 4)"""

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        """Override to for env reset"""

class SRT(BaseController):

    def __init__(self, drone, num_envs, dt, cfg):
        super().__init__(drone, num_envs, dt, cfg)

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.max_rpm


# class SRT(BaseController):
#     """Single Rotor Thrust: actions are per-motor thrust deltas around hover RPM."""
#
#     def __init__(self, drone, num_envs, dt, cfg):
#         super().__init__(drone, num_envs, dt, cfg)
#         # self.hover_rpm = cfg.get("hover_rpm", 15502.5)
#         self.action_scale = cfg.get("action_scale", 0.8)
#
#     def update(self, actions: torch.Tensor) -> torch.Tensor:
#         return (1 + actions * self.action_scale) * self.hover_rpm

class SRTHover(BaseController):
    """Single rotor thrust around hover: actions in [-1, 1] are relative deltas of the motor command,
    u = hover_cmd * (1 + action_scale * a), rpm = u * max_rpm (u = SRT's action), so a = 0 hovers.
    rl_goto.cpp: action_mode "hover_delta" with the same hover_cmd and action_scale."""

    def __init__(self, drone, num_envs, dt, cfg):
        super().__init__(drone, num_envs, dt, cfg)
        self.action_scale = cfg.get("action_scale", 1.0)
        self.hover_u = self.hover_rpm / self.max_rpm  # sqrt(1 / thrust2weight)

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        u = torch.clamp(self.hover_u * (1.0 + self.action_scale * actions), 0.0, 1.0)
        return u * self.max_rpm


class px4CTBR(BaseController):
    """PX4-style rate controller:
      [collective_thrust, roll_rate_sp, pitch_rate_sp, yaw_rate_sp]."""

    def __init__(self, drone, num_envs, dt, cfg):
        super().__init__(drone, num_envs, dt, cfg)
        device, ft = gs.device, gs.tc_float

        def t(key, default):
            return torch.tensor(cfg.get(key, default), device=device, dtype=ft)

        self.max_rates = t("max_rates", (6.0, 6.0, 3.0))          
        # self.max_rates = t("max_rates", (3.0, 3.0, 1.5))          

        K = t("rate_k", (1.0, 1.0, 1.0))
        # self.gain_p = K * t("rate_p", (0.042, 0.042, 0.2))
        # self.gain_i = K * t("rate_i", (0.08, 0.08, 0.1))
        # self.gain_d = K * t("rate_d", (0.0015, 0.0015, 0.0))
        self.gain_p = K * t("rate_p", (0.02, 0.02, 0.2))    
        self.gain_i = K * t("rate_i", (0.0, 0.0, 0.0))
        self.gain_d = K * t("rate_d", (0.0, 0.0, 0.0))
        self.gain_ff = t("rate_ff", (0.0, 0.0, 0.0))
        self.lim_int = t("rate_int_limit", (0.30, 0.30, 0.30))
        self.i_factor_norm = np.radians(400.0)

        mixer = torch.tensor(cfg["ctbr_mixer"], device=device, dtype=ft)
        self.thrust_scale = torch.sign(mixer[0])
        self.roll_scale = torch.sign(mixer[1])
        self.pitch_scale = torch.sign(mixer[2])
        self.yaw_scale = torch.sign(mixer[3])

        self.rate_int = torch.zeros((num_envs, 3), device=device, dtype=ft)
        self.prev_rate = torch.zeros((num_envs, 3), device=device, dtype=ft)

    def reset_idx(self, envs_idx):
        self.rate_int[envs_idx] = 0.0
        self.prev_rate[envs_idx] = 0.0

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        rate = transform_by_quat(self.drone.get_ang(), inv_quat(self.drone.get_quat()))

        a = actions[:, 0:1]

        # Centering actions around hover so that:
        #  a= -1 is 3200 rpm; a=0 is hover  (8120 rpm); and a=1 is max throttle (21400 rpm) -> a300 drone
        throttle = torch.where(
            a < 0,
            self.hover_cmd * (1.0 + a),                         
            self.hover_cmd + a * (1.0 - self.hover_cmd),      
        )
        rate_sp = actions[:, 1:4] * self.max_rates
               
        angular_accel = (rate - self.prev_rate) / self.dt
        self.prev_rate.copy_(rate)

        rate_error = rate_sp - rate
        torque = (self.gain_p * rate_error
                  + self.rate_int
                  - self.gain_d * angular_accel
                  + self.gain_ff * rate_sp)               
        self._update_integral(rate_error)

        motor_cmd = self._mixer_px4(throttle, torque)        # [0, 1] 

        return self.min_rpm + motor_cmd * (self.max_rpm - self.min_rpm)

    def _update_integral(self, rate_error):
        i_factor = rate_error / self.i_factor_norm
        i_factor = torch.clamp(1.0 - i_factor * i_factor, min=0.0)
        rate_i = self.rate_int + i_factor * self.gain_i * rate_error * self.dt
        rate_i = rate_i.clamp(-self.lim_int, self.lim_int)
        self.rate_int = torch.where(torch.isfinite(rate_i), rate_i, self.rate_int)

    def _compute_desat_gain(self, outputs, desat, min_o, max_o, eps=1e-6):
        valid = desat.abs() >= eps
        desat_sfe = torch.where(valid, desat, torch.ones_like(desat))
        below = outputs < min_o
        above = outputs > max_o
        k_low = (min_o - outputs) / desat_sfe
        k_high = (max_o - outputs) / desat_sfe
        cand = torch.cat([k_low, k_high], dim=-1)
        cond = torch.cat([below & valid, above & valid], dim=-1)
        k_min = torch.minimum(torch.zeros_like(outputs[:, :1]),
                              torch.where(cond, cand, float("inf")).amin(-1, keepdim=True))
        k_max = torch.maximum(torch.zeros_like(outputs[:, :1]),
                              torch.where(cond, cand, -float("inf")).amax(-1, keepdim=True))
        return k_min + k_max

    def _minimize_saturation(self, outputs, desat, min_o=0.0, max_o=1.0, reduce_only=False):
        k1 = self._compute_desat_gain(outputs, desat, min_o, max_o)
        if reduce_only:
            k1 = torch.where(k1 > 0.0, torch.zeros_like(k1), k1)
        outputs = outputs + k1 * desat
        k2 = 0.5 * self._compute_desat_gain(outputs, desat, min_o, max_o)
        if reduce_only:
            k2 = torch.where(k1 == 0.0, torch.zeros_like(k2), k2)
        return outputs + k2 * desat

    def _mix_yaw(self, outputs, yaw):
        outputs = outputs + yaw * self.yaw_scale
        outputs = self._minimize_saturation(outputs, self.yaw_scale, 0.0, 1.15, reduce_only=False)
        outputs = self._minimize_saturation(outputs, self.thrust_scale, 0.0, 1.0, reduce_only=True)
        return outputs

    def _mixer_px4(self, throttle, torque):
        roll = torque[:, 0:1].clamp(-1.0, 1.0)
        pitch = torque[:, 1:2].clamp(-1.0, 1.0)
        yaw = torque[:, 2:3].clamp(-1.0, 1.0)
        thr = throttle.clamp(0.0, 1.0)

        outputs = (roll * self.roll_scale
                   + pitch * self.pitch_scale
                   + thr * self.thrust_scale)

        # airmode = 0
        outputs = self._minimize_saturation(outputs, self.thrust_scale, 0.0, 1.0, reduce_only=True)
        outputs = self._minimize_saturation(outputs, self.roll_scale, 0.0, 1.0, reduce_only=False)
        outputs = self._minimize_saturation(outputs, self.pitch_scale, 0.0, 1.0, reduce_only=False)
        outputs = self._mix_yaw(outputs, yaw)

        return outputs.clamp(0.0, 1.0)


CONTROLLERS = {
    "SRT": SRT,
    "SRTHover": SRTHover,
    "CTBR": px4CTBR, 
}


def build_controller(name: str, drone: "DroneEntity", num_envs: int, dt: float, cfg: dict) -> BaseController:
    try:
        controller_cls = CONTROLLERS[name]
    except KeyError:
        raise ValueError(f"Unknown controller_type '{name}'. Available: {list(CONTROLLERS)}") from None
    return controller_cls(drone, num_envs, dt, cfg)
