from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from genesis.engine.entities.drone_entity import DroneEntity


def drone_params(drone_cfg: dict) -> dict:
    """Mass, kf, km, max_rpm, hover_rpm and thrust2weight of a drone config with {mass, kf, km: {nominal}, max_rpm}
    (DesignInformedDR.nominal_cfg)."""
    p = {k: drone_cfg[k]["nominal"] for k in ("mass", "kf", "km")}
    p["max_rpm"] = drone_cfg["max_rpm"]
    p["hover_rpm"] = np.sqrt(9.81 * p["mass"] / 4.0 / p["kf"])
    p["thrust2weight"] = (p["max_rpm"] / p["hover_rpm"]) ** 2
    return p


class BaseController(ABC):

    def __init__(self, drone: "DroneEntity", num_envs: int, dt: float, cfg: dict):
        self.drone = drone
        self.num_envs = num_envs
        self.dt = dt
        self.cfg = cfg

        params = drone_params(cfg["drone"])
        self.KF = params["kf"]
        self.KM = params["km"]
        self.mass = params["mass"]
        self.hover_rpm = params["hover_rpm"]
        self.max_rpm = params["max_rpm"]  # the envs replace it by the per-env max rpm of the sampled drones

    @abstractmethod
    def update(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: (num_envs, num_actions) -> rpms: (num_envs, 4)"""

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        """Override to for env reset"""


class SRT(BaseController):
    """Single rotor thrust: actions in [0, 1] are the motor commands, rpm = action * max_rpm."""

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.max_rpm


CONTROLLERS = {
    "SRT": SRT,
}


def build_controller(name: str, drone: "DroneEntity", num_envs: int, dt: float, cfg: dict) -> BaseController:
    try:
        controller_cls = CONTROLLERS[name]
    except KeyError:
        raise ValueError(f"Unknown controller_type '{name}'. Available: {list(CONTROLLERS)}") from None
    return controller_cls(drone, num_envs, dt, cfg)
